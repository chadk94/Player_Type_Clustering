"""Private last-successful sportsbook responses, separate from live price tracking."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import os
import tempfile
from threading import RLock
from zoneinfo import ZoneInfo


SOURCE = '_sportsbook_source'
SAVED = 'Saved quote — not live'
# Share request locks across browser sessions; only one caller pays for a cache miss.
_REQUEST_LOCKS = [RLock() for _ in range(64)]


def utc(value):
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        raise ValueError('Timestamp needs a timezone')
    return stamp.astimezone(timezone.utc)


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _events(payload):
    return payload if isinstance(payload, list) else [payload]


def _valid(payload, path, params):
    listing = path in ('/events', '/odds')
    if not isinstance(payload, list if listing else dict):
        return False
    seen = set()
    try:
        for event in _events(payload):
            if not isinstance(event, dict) or not event['id'] or event['id'] in seen:
                return False
            seen.add(event['id'])
            if not event['home_team'] or not event['away_team'] or event['home_team'] == event['away_team']:
                return False
            utc(event['commence_time'])
            if not listing and path != f"/events/{event['id']}/odds":
                return False
            if path != '/events':
                books = event['bookmakers']
                if not isinstance(books, list):
                    return False
                wanted = set(params.get('bookmakers', '').split(','))
                markets = set(params.get('markets', '').split(','))
                for book in books:
                    if book['key'] not in wanted or not isinstance(book['markets'], list):
                        return False
                    for market in book['markets']:
                        if market['key'] not in markets or not isinstance(market['outcomes'], list):
                            return False
                        utc(market.get('last_update', book.get('last_update', '')))
        return True
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def _annotate(payload, saved_at, fallback, persisted=True, reused=False):
    result = deepcopy(payload)
    for event in _events(result):
        event[SOURCE] = dict(saved_at=saved_at, fallback=fallback, persisted=persisted, reused=reused)
    return result


def _write(path, record):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # One atomic envelope keeps the timestamp, request identity and checksum together.
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.lines-')
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(_encoded(record))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _destination(path, api_key, params, cache_dir):
    root = Path(cache_dir or os.environ.get('SPORTSBOOK_LINES_CACHE_DIR') or
                Path(tempfile.gettempdir()) / 'nba-prop-sportsbook-lines-v1')
    identity = sha256(_encoded(['nba-v4', path, params, sha256(api_key.encode()).hexdigest()])).hexdigest()
    return root / (identity + '.json'), identity


def _read_record(destination, identity, path, params, now):
    try:
        record = json.loads(destination.read_text())
        digest = record.pop('sha256')
        if (digest != sha256(_encoded(record)).hexdigest() or record['request'] != identity or
                utc(record['saved_at']) > now or
                utc(record.get('last_attempt_at', record['saved_at'])) > now or
                not _valid(record['payload'], path, params)):
            return None
        return record
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _restore(record, now, *, fallback=False):
    payload = record['payload']
    future = [e for e in _events(payload) if now < utc(e['commence_time'])]
    if not isinstance(payload, list) and not future:
        return None
    return _annotate(future if isinstance(payload, list) else future[0],
                     record['saved_at'], fallback or record.get('refresh_failed', False), reused=True)


def cached_feed(path, api_key, *, cache_dir=None, now=None, max_age_seconds=None, **params):
    """Read an exact request's verified saved response without any network access."""
    now = now or datetime.now(timezone.utc)
    destination, identity = _destination(path, api_key, params, cache_dir)
    record = _read_record(destination, identity, path, params, now)
    if record is None or (max_age_seconds is not None and
            (now - utc(record['saved_at'])).total_seconds() >= max_age_seconds):
        return None
    return _restore(record, now)


def read_saved_feed(path, api_key, fetch, *, cache_dir=None, now=None,
                    refresh_after=0, force=False, **params):
    """Reuse verified saved prices until due; None means explicit refresh only.

    First access still retrieves missing data. Browser reloads and Streamlit cache
    clears cannot bypass this disk-backed policy. Failed refreshes retain the same
    timestamped backup. Credentials and request URLs are never persisted.
    """
    from sportsbook_lines import OddsFeedError
    fixed_now = now
    destination, identity = _destination(path, api_key, params, cache_dir)
    with _REQUEST_LOCKS[int(identity[:2], 16) % len(_REQUEST_LOCKS)]:
        now = fixed_now or datetime.now(timezone.utc)
        record = _read_record(destination, identity, path, params, now)
        saved = _restore(record, now) if record is not None else None
        if saved is not None and not force and (refresh_after is None or
                (now - utc(record.get('last_attempt_at', record['saved_at']))).total_seconds() < refresh_after):
            return saved
        try:
            payload = fetch(path, api_key, **params)
            if not _valid(payload, path, params):
                raise OddsFeedError('Sportsbook response could not be verified.')
        except OddsFeedError as exc:
            if record is not None:
                backup = _restore(record, now, fallback=True)
                if backup is not None:
                    # Back off failed hourly refreshes without changing the quote's age.
                    record.update(last_attempt_at=now.isoformat(), refresh_failed=True)
                    try:
                        record['sha256'] = sha256(_encoded(record)).hexdigest()
                        _write(destination, record)
                    except (OSError, ValueError, TypeError):
                        pass
                    return backup
            raise OddsFeedError(f'{exc} No saved lines are available for this request.') from None
        now = fixed_now or datetime.now(timezone.utc)
        record = dict(request=identity, saved_at=now.isoformat(), payload=payload)
        persisted = True
        try:
            record['sha256'] = sha256(_encoded(record)).hexdigest()
            _write(destination, record)
        except (OSError, ValueError, TypeError):
            persisted = False
        return _annotate(payload, now.isoformat(), False, persisted)


def saved_eligible(payload, now=None):
    """Only locally timestamped, still-pregame snapshots can be used as saved quotes."""
    now = now or datetime.now(timezone.utc)
    try:
        return utc(payload[SOURCE]['saved_at']) <= now < utc(payload['commence_time'])
    except (ValueError, TypeError, KeyError, AttributeError):
        return False


def quote_is_saved(payload, updated, now=None, max_age_seconds=900):
    now = now or datetime.now(timezone.utc)
    if not saved_eligible(payload, now):
        return False
    try:
        stamp = utc(updated)
        return (stamp <= utc(payload[SOURCE]['saved_at']) and
                (payload[SOURCE]['fallback'] or payload[SOURCE].get('reused', False) or
                 (now - stamp).total_seconds() > max_age_seconds))
    except (ValueError, TypeError, KeyError, AttributeError):
        return False


def uses_saved_lines(payload, now=None):
    now = now or datetime.now(timezone.utc)
    if not saved_eligible(payload, now):
        return False
    return bool(payload[SOURCE]['fallback'] or payload[SOURCE].get('reused', False) or any(
        quote_is_saved(payload, m.get('last_update', b.get('last_update', '')), now)
        for b in payload.get('bookmakers', []) for m in b.get('markets', [])))


def _age(stamp, now):
    minutes = max(0, int((now - stamp).total_seconds() // 60))
    if minutes < 1:
        return 'less than a minute ago'
    if minutes < 60:
        return f'{minutes} min ago'
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f'{hours} hr {minutes} min ago'
    days, hours = divmod(hours, 24)
    return f"{days} {'day' if days == 1 else 'days'} {hours} hr ago"


def source_notice(payload, now=None):
    """Return a short visible label; age is recomputed at render, never re-stamped."""
    now = now or datetime.now(timezone.utc)
    try:
        source = payload[SOURCE]
        saved = utc(source['saved_at'])
        times = [utc(m.get('last_update', b.get('last_update', '')))
                 for b in payload.get('bookmakers', []) for m in b.get('markets', [])]
        stamp = saved.astimezone(ZoneInfo('America/Denver'))
        mode = 'Saved lines — not live' if uses_saved_lines(payload, now) else 'Sportsbook lines'
        action = 'Last saved' if source.get('persisted', True) else 'Retrieved'
        text = f'{mode} · {action} {stamp:%b %d, %Y, %I:%M %p %Z} ({_age(saved, now)}).'
        if times:
            oldest = min(times)
            local = oldest.astimezone(ZoneInfo('America/Denver'))
            text += f' Oldest price update: {local:%b %d, %I:%M %p %Z} ({_age(oldest, now)}).'
        if uses_saved_lines(payload, now):
            text += ' Comparisons use saved prices; current availability is unconfirmed.'
        if not source.get('persisted', True):
            text += ' Backup could not be saved; a later outage may leave lines unavailable.'
        return text
    except (ValueError, TypeError, KeyError, AttributeError):
        return ''


def render_source(ui, payload, label='', now=None):
    text = source_notice(payload, now)
    if text:
        draw = ui.warning if uses_saved_lines(payload, now) else ui.caption
        draw((label + ' · ' if label else '') + text)
