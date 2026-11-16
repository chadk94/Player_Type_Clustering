"""Freeze the selected side of a fresh sportsbook quote for Jae's card."""
from collections import defaultdict
from datetime import datetime, timezone
import math
import re
from zoneinfo import ZoneInfo

from live_projections import name_key
from play_identity import prop_id
from sportsbook_lines import BOOKS, MARKETS, TEAM_NAMES, OddsFeedError, american_probability, game_quotes, read_feed

DENVER = ZoneInfo('America/Denver')
CONFIDENCE_TIERS = {5: 'Low', 10: 'Medium', 15: 'High'}


def worst_price(odds, cents=5):
    """Move the selected number of price ticks against the bettor, counting even money once."""
    if type(odds) is not int or abs(odds) < 100:
        raise ValueError('Invalid American odds.')
    if type(cents) is not int or cents not in CONFIDENCE_TIERS:
        raise ValueError('Choose Low, Medium or High confidence.')
    ticks = odds - 100 if odds > 0 else odds + 100
    ticks -= cents
    return ticks + 100 if ticks >= 0 else ticks - 100


def current_event(event_id, stats, over_stats):
    from FanduelScrape import get_odds_api_key
    base = [key for key, stat in MARKETS.items() if stat in stats]
    alts = [key + '_alternate' for key, stat in MARKETS.items() if stat in over_stats]
    api_key = get_odds_api_key()

    def fetch(markets):
        return read_feed(f'/events/{event_id}/odds', api_key, bookmakers='fanduel',
                         markets=','.join(markets), oddsFormat='american')

    if alts:
        try:
            return fetch(base + alts)
        except OddsFeedError:
            pass  # Keep the confirmed primary playable if the alternate feed fails.
    return fetch(base)


def alternate_overs(event, now):
    """Index current one-sided FanDuel ladders separately from paired main markets."""
    prices = defaultdict(lambda: defaultdict(set))
    for book in event.get('bookmakers', []):
        if book.get('key') != 'fanduel':
            continue
        for market in book.get('markets', []):
            key = market.get('key', '')
            if not key.endswith('_alternate') or (stat := MARKETS.get(key[:-10])) is None:
                continue
            try:
                updated = market.get('last_update') or book.get('last_update')
                age = (now - datetime.fromisoformat(updated.replace('Z', '+00:00'))).total_seconds()
                if not -60 <= age <= 900:
                    continue
            except (AttributeError, TypeError, ValueError):
                continue
            for outcome in market.get('outcomes', []):
                if outcome.get('name') != 'Over' or not outcome.get('description'):
                    continue
                try:
                    line, odds = float(outcome['point']), float(outcome['price'])
                    if not math.isfinite(line) or line < 0 or not odds.is_integer() or abs(odds) < 100:
                        continue
                except (TypeError, ValueError, KeyError):
                    continue
                prices[name_key(outcome['description']), stat][line].add(int(odds))
    return prices


def next_over(pick, prices):
    if pick['side'] != 'Over':
        return None
    ladder = prices.get((name_key(pick['player']), pick['stat']), {})
    if pick['stat'] == '3PM':
        target = pick['line'] + 1
    else:
        available = sorted(line for line, odds in ladder.items() if line > pick['line'] and len(odds) == 1)
        step = {'PTS': 5, 'REB': 2}.get(pick['stat'])
        preferred = [line for line in available if step and (line + 0.5) % step == 0]
        target = next(iter(preferred or available), None)
    odds = ladder.get(target, set())
    return {'line': target, 'odds': next(iter(odds)) if len(odds) == 1 else None}


def card_quotes(picks, fetch=current_event, clock=lambda: datetime.now(timezone.utc)):
    """Requote every confirmed play at FanDuel at its exact saved line."""
    groups = defaultdict(list)
    for p in picks:
        if p['status'] == 'Confirmed':
            groups[p['event']].append(p)
    rows, skipped = [], []
    for event_id, group in groups.items():
        try:
            if not re.fullmatch(r'[a-fA-F0-9]{32}', event_id):
                raise ValueError('unrecognized event')
            event = fetch(event_id, {p['stat'] for p in group}, {p['stat'] for p in group if p['side'] == 'Over'})
            now = clock()
            start = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
            if event['id'] != event_id or start <= now:
                raise ValueError('event unavailable or already started')
            game = f"{TEAM_NAMES[event['away_team']]} @ {TEAM_NAMES[event['home_team']]}"
            quotes = game_quotes(event, [p['player'] for p in group], 'fanduel', now)
            ladders = alternate_overs(event, now)
        except (OddsFeedError, ValueError, TypeError, KeyError, AttributeError):
            skipped.extend(f"{p['player']} {p['side']} {p['line']:g} {p['stat']} (FanDuel) — game or quote unavailable." for p in group)
            continue
        for p in group:
            label = f"{p['player']} {p['side']} {p['line']:g} {p['stat']} (FanDuel)"
            if game != p['game'] or start.astimezone(DENVER).date().isoformat() != p['day']:
                skipped.append(f'{label} — game changed.')
                continue
            quote = quotes[name_key(p['player'])].set_index('Stat').loc[p['stat']]
            if quote['Market status'] != 'Current' or quote['Book line'] != p['line']:
                skipped.append(f'{label} — line moved or no fresh price. Review it on Edges.')
                continue
            raw_odds = quote[p['side'] + ' odds']
            if not float(raw_odds).is_integer():
                skipped.append(f'{label} — unsupported price.')
                continue
            price = int(raw_odds)
            worst = worst_price(price, p.get('confidence_cents', 5))
            rows.append(dict(id=p['id'], day=p['day'], start=start.isoformat(), game=game,
                             player=p['player'], stat=p['stat'], side=p['side'], line=p['line'],
                             book='FanDuel', alt=next_over(p, ladders), odds=price, worst=worst, breakeven=american_probability(worst),
                             quoted=quote['Updated UTC'], sent=now.isoformat()))
    return rows, skipped


def sheet_request(connection, action, day, *, ids=(), rows=(), edits=()):
    """Use the private write key only for the owner's selected Google deployment."""
    import requests
    endpoint, separator, key = connection.partition('#')
    if (not separator or not re.fullmatch(r'https://script\.google\.com/macros/s/[\w-]{20,200}/exec', endpoint)
            or not re.fullmatch(r'[a-f0-9]{64}', key)):
        raise ValueError('Paste the connection from the Jae card menu in your Google Sheet.')
    body = dict(key=key, action=action, day=day, ids=list(ids), rows=list(rows), edits=list(edits))
    try:
        response = requests.post(endpoint, json=body, timeout=(4,25), allow_redirects=False)
        if response.status_code in (302, 303):
            location = response.headers.get('Location','')
            if not location.startswith('https://script.googleusercontent.com/'):
                raise ValueError('Google needs the sheet owner to finish the connection setup.')
            response = requests.get(location, timeout=(4,20), allow_redirects=False)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, requests.JSONDecodeError):
        raise ValueError('Google did not confirm delivery. Retry to check the sheet for existing rows.') from None
    if not isinstance(data, dict) or data.get('ok') is not True:
        raise ValueError('The sheet rejected the request. Check its connection and daily headers.')
    return data


def send_card(day, picks, connection, request=sheet_request, fetch=current_event):
    # The same pick can be shortlisted at more than one book. Use its FanDuel
    # identity on the card so repeat sends and cross-book selections agree.
    unique = {}
    source_ids = defaultdict(list)
    # Prefer an explicit edit. If both are edits, use the tighter limit.
    def priority(item):
        return (not item.get('confidence_edited', False), item.get('confidence_cents', 5))

    for p in picks:
        if p['status'] == 'Confirmed':
            source_id = p['id']
            if p['book'] != 'FanDuel':
                p = dict(p, book='FanDuel')
                p['id'] = prop_id(p)
            source_ids[p['id']].append((source_id,p.get('confidence_cents',5)))
            prior = unique.get(p['id'])
            if prior is None or priority(p) < priority(prior):
                unique[p['id']] = p
    confirmed = list(unique.values())
    if not confirmed:
        raise ValueError('Confirm at least one play for this slate.')
    # Recognize immutable rows sent under another book before the FanDuel-only switch.
    aliases = {p['id']: {p['id']} | {prop_id(dict(p, book=b)) for b in BOOKS} for p in confirmed}
    all_ids = sorted(set().union(*aliases.values()))
    check = request(connection, 'check', day, ids=all_ids)
    sent = set(check['existing'])
    existing = {pid for pid, keys in aliases.items() if keys & sent}
    rows, skipped = card_quotes([p for p in confirmed if p['id'] not in existing], fetch)
    new_ids = {r['id'] for r in rows}
    delivered = existing | new_ids
    edits = []
    for p in confirmed:
        if not p.get('confidence_edited') or p['id'] not in delivered:
            continue
        matching = aliases[p['id']] & sent
        if len(matching)>1:
            raise ValueError('Multiple sent rows match this prop. Check the daily card before changing its limit.')
        edits.append(dict(id=next(iter(matching),p['id']), cents=p['confidence_cents']))
    if edits and not check.get('confidenceEdits'):
        raise ValueError('The connected sheet needs the confidence update before sending these changes.')
    added = updated = 0
    if rows or edits:
        reply = request(connection, 'append', day, rows=rows, **({'edits':edits} if edits else {}))
        added = reply['added']
        if type(added) is not int or not 0 <= added <= len(rows):
            raise ValueError('Google did not confirm the row count. Retry to check the sheet.')
        updated = reply.get('updated',0) if not edits else reply.get('updated')
        if type(updated) is not int or not 0 <= updated <= len(edits):
            raise ValueError('Google did not confirm the confidence updates. Retry to check the sheet.')
    synced = {source:cents for pid in delivered for source,cents in source_ids[pid]}
    message = f'{added} added to {day} · {updated} updated · {len(delivered)-added-updated} already sent.'
    return message, skipped, synced
