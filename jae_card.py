"""Freeze the selected side of a fresh sportsbook quote for Jae's card."""
from collections import defaultdict
from datetime import datetime, timezone
import re
from zoneinfo import ZoneInfo

from live_projections import name_key
from sportsbook_lines import BOOKS, MARKETS, TEAM_NAMES, OddsFeedError, american_probability, game_quotes, read_feed

DENVER = ZoneInfo('America/Denver')


def worst_price(odds):
    """Move five American price ticks against the bettor, counting even money once."""
    if type(odds) is not int or abs(odds) < 100:
        raise ValueError('Invalid American odds.')
    ticks = odds - 100 if odds > 0 else odds + 100
    ticks -= 5
    return ticks + 100 if ticks >= 0 else ticks - 100


def current_event(event_id, book, stats):
    from FanduelScrape import get_odds_api_key
    return read_feed(f'/events/{event_id}/odds', get_odds_api_key(), bookmakers=book,
                     markets=','.join(key for key, stat in MARKETS.items() if stat in stats),
                     oddsFormat='american')


def card_quotes(picks, fetch=current_event, clock=lambda: datetime.now(timezone.utc)):
    """One fresh request per event/book. Leave changed or unavailable lines unsent."""
    groups = defaultdict(list)
    for p in picks:
        if p['status'] == 'Confirmed':
            groups[p['event'], BOOKS[p['book']]].append(p)
    rows, skipped = [], []
    for (event_id, book), group in groups.items():
        try:
            if not re.fullmatch(r'[a-fA-F0-9]{32}', event_id):
                raise ValueError('unrecognized event')
            event = fetch(event_id, book, {p['stat'] for p in group})
            now = clock()
            start = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
            if event['id'] != event_id or start <= now:
                raise ValueError('event unavailable or already started')
            game = f"{TEAM_NAMES[event['away_team']]} @ {TEAM_NAMES[event['home_team']]}"
            quotes = game_quotes(event, [p['player'] for p in group], book, now)
        except (OddsFeedError, ValueError, TypeError, KeyError, AttributeError):
            skipped.extend(f"{p['player']} {p['side']} {p['line']:g} {p['stat']} ({p['book']}) — game or quote unavailable." for p in group)
            continue
        for p in group:
            label = f"{p['player']} {p['side']} {p['line']:g} {p['stat']} ({p['book']})"
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
            worst = worst_price(price)
            rows.append(dict(id=p['id'], day=p['day'], start=start.isoformat(), game=game,
                             player=p['player'], stat=p['stat'], side=p['side'], line=p['line'],
                             book=p['book'], odds=price, worst=worst, breakeven=american_probability(worst),
                             quoted=quote['Updated UTC'], sent=now.isoformat()))
    return rows, skipped


def sheet_request(connection, action, day, *, ids=(), rows=()):
    """Use the private write key only for the owner's selected Google deployment."""
    import requests
    endpoint, separator, key = connection.partition('#')
    if (not separator or not re.fullmatch(r'https://script\.google\.com/macros/s/[\w-]{20,200}/exec', endpoint)
            or not re.fullmatch(r'[a-f0-9]{64}', key)):
        raise ValueError('Paste the connection from the Jae card menu in your Google Sheet.')
    body = dict(key=key, action=action, day=day, ids=list(ids), rows=list(rows))
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
    confirmed = [p for p in picks if p['status'] == 'Confirmed']
    ids = {p['id'] for p in confirmed}
    if not ids:
        raise ValueError('Confirm at least one play for this slate.')
    existing = set(request(connection, 'check', day, ids=sorted(ids))['existing']) & ids
    rows, skipped = card_quotes([p for p in confirmed if p['id'] not in existing], fetch)
    added = 0
    if rows:
        reply = request(connection, 'append', day, rows=rows)
        added = reply['added']
        if type(added) is not int or not 0 <= added <= len(rows):
            raise ValueError('Google did not confirm the row count. Retry to check the sheet.')
    return f'{added} added to {day} · {len(existing) + len(rows) - added} already sent.', skipped
