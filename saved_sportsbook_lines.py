"""Private last-successful sportsbook responses, separate from live price tracking."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import os
import tempfile
from zoneinfo import ZoneInfo


SOURCE = '_sportsbook_source'
SAVED = 'Saved quote — not live'


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


def _annotate(payload, saved_at, fallback, persisted=True):
    result = deepcopy(payload)
    for event in _events(result):
        event[SOURCE] = dict(saved_at=saved_at, fallback=fallback, persisted=persisted)
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


def read_saved_feed(path, api_key, fetch, *, cache_dir=None, now=None, **params):
    """Try the provider first; only its exact request's verified backup can substitute.

    No keys, request URLs or exception text are persisted. Cloud temporary storage
    survives ordinary reruns/restarts, but a persistent mount is needed for redeploys.
    Empty successful responses are authoritative; they do not revive withdrawn lines.
    """
    from sportsbook_lines import OddsFeedError
    fixed_now = now
    now = now or datetime.now(timezone.utc)
    root = Path(cache_dir or os.environ.get('SPORTSBOOK_LINES_CACHE_DIR') or
                Path(tempfile.gettempdir()) / 'nba-prop-sportsbook-lines-v1')
    # Partition accounts without storing the credential in either file names or bodies.
    identity = sha256(_encoded(['nba-v4', path, params, sha256(api_key.encode()).hexdigest()])).hexdigest()
    destination = root / (identity + '.json')
    try:
        payload = fetch(path, api_key, **params)
        if not _valid(payload, path, params):
            raise OddsFeedError('Sportsbook response could not be verified.')
    except OddsFeedError as exc:
        try:
            record = json.loads(destination.read_text())
            digest = record.pop('sha256')
            if digest != sha256(_encoded(record)).hexdigest() or record['request'] != identity:
                raise ValueError('Saved lines failed verification')
            saved_at = utc(record['saved_at'])
            payload = record['payload']
            if saved_at > now or not _valid(payload, path, params):
                raise ValueError('Saved lines failed verification')
            # Never turn an old or in-game quote into a new pregame price.
            future = [e for e in _events(payload) if now < utc(e['commence_time'])]
            if not future:
                raise ValueError('No saved upcoming lines')
            payload = future if isinstance(payload, list) else future[0]
            return _annotate(payload, record['saved_at'], True)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            raise OddsFeedError(f'{exc} No saved lines are available for this request.') from None
    now = fixed_now or datetime.now(timezone.utc)
    record = dict(request=identity, saved_at=now.isoformat(), payload=payload)
    persisted = True
    try:
        record['sha256'] = sha256(_encoded(record)).hexdigest()
        _write(destination, record)
    except (OSError, ValueError, TypeError):
        persisted = False  # A cache failure must not hide good live prices.
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
                (payload[SOURCE]['fallback'] or (now - stamp).total_seconds() > max_age_seconds))
    except (ValueError, TypeError, KeyError, AttributeError):
        return False


def uses_saved_lines(payload, now=None):
    now = now or datetime.now(timezone.utc)
    if not saved_eligible(payload, now):
        return False
    return bool(payload[SOURCE]['fallback'] or any(
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
