"""Freeze the selected side of a fresh sportsbook quote for Jae's card."""
from collections import defaultdict
from datetime import datetime, timezone
import math
import re
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo

from live_projections import name_key
from play_identity import prop_id
from sportsbook_lines import BOOKS, MARKETS, TEAM_NAMES, OddsFeedError, american_probability, game_quotes, read_feed, sportsbook_player_key

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


def current_event(event_id, stats, alternate_stats):
    from FanduelScrape import get_odds_api_key
    base = [key for key, stat in MARKETS.items() if stat in stats]
    alts = [key + '_alternate' for key, stat in MARKETS.items() if stat in alternate_stats]
    if alternate_stats:
        alts.append('player_double_double')
    api_key = get_odds_api_key()

    def fetch(markets):
        source_links = {'includeLinks': 'true'} if any(
            market in markets for market in ('player_points_alternate', 'player_rebounds_alternate')) else {}
        return read_feed(f'/events/{event_id}/odds', api_key, bookmakers='fanduel',
                         markets=','.join(markets), oddsFormat='american', **source_links)

    if alts:
        try:
            return fetch(base + alts)
        except OddsFeedError:
            pass  # Keep the confirmed primary playable if the alternate feed fails.
    return fetch(base)


def _fanduel_ids(outcome):
    """Read the native market and selection IDs without visiting the betslip link."""
    try:
        url = urlsplit(outcome.get('link') or '')
        if url.scheme != 'https' or not (url.hostname or '').endswith('.fanduel.com'):
            return None, None
        params = parse_qs(url.query)
        return params.get('marketId', [None])[0], params.get('selectionId', [None])[0]
    except (TypeError, ValueError):
        return None, None


def _milestone_prices(markets, stat):
    # The feed combines two native sections. In Player Points/Rebounds X+, a market is
    # one threshold and its selections are players. A player's ordinary ladder
    # is one market with a different selection for each threshold.
    fixed = [rows for rows in markets.values()
             if len({line for _, _, line, _, _ in rows}) == 1
             and all(side == 'Over' and line % 1 == 0.5 for _, side, line, _, _ in rows)]
    runner_lines = defaultdict(set)
    for rows in fixed:
        for player, _, line, _, runner in rows:
            if runner:
                runner_lines[player, runner].add(line)
    preferred = defaultdict(lambda: defaultdict(set))
    for rows in fixed:
        multiple_players = len({row[0] for row in rows}) > 1
        for player, side, line, odds, runner in rows:
            if multiple_players or (runner and len(runner_lines[player, runner]) > 1):
                preferred[player, stat, side][line].add(odds)
    return preferred


def alternate_prices(event, now):
    """Collect fresh FanDuel alternate quotes by player, stat, side, and exact line."""
    prices = defaultdict(lambda: defaultdict(set))
    milestone_markets = defaultdict(lambda: defaultdict(list))
    for book in event.get('bookmakers', []):
        if book.get('key') != 'fanduel':
            continue
        for market in book.get('markets', []):
            key = market.get('key', '')
            stat = 'DD' if key == 'player_double_double' else MARKETS.get(key.removesuffix('_alternate'))
            if not stat or (stat != 'DD' and not key.endswith('_alternate')):
                continue
            try:
                updated = market.get('last_update') or book.get('last_update')
                age = (now - datetime.fromisoformat(updated.replace('Z', '+00:00'))).total_seconds()
                if not -60 <= age <= 900:
                    continue
            except (AttributeError, TypeError, ValueError):
                continue
            for outcome in market.get('outcomes', []):
                side = outcome.get('name')
                if side not in (('Yes',) if stat == 'DD' else ('Over','Under')) or not outcome.get('description'):
                    continue
                try:
                    line = 0.0 if stat == 'DD' else float(outcome['point'])
                    odds = float(outcome['price'])
                    if not math.isfinite(line) or line < 0 or not odds.is_integer() or abs(odds) < 100:
                        continue
                except (TypeError, ValueError, KeyError):
                    continue
                player = sportsbook_player_key(outcome['description'])
                prices[player, stat, side][line].add(int(odds))
                if stat in ('PTS', 'REB'):
                    market_id, runner = _fanduel_ids(outcome)
                    if market_id:
                        milestone_markets[stat][market_id].append((player, side, line, int(odds), runner))
    for stat, markets in milestone_markets.items():
        for key, milestones in _milestone_prices(markets, stat).items():
            prices[key].update(milestones)
    return prices


def next_over(pick, prices):
    if pick['side'] != 'Over':
        return None
    ladder = prices.get((sportsbook_player_key(pick['player']), pick['stat'], 'Over'), {})
    if pick['stat'] == '3PM':
        target = pick['line'] + 1
    else:
        available = sorted(line for line, odds in ladder.items() if line > pick['line'] and len(odds) == 1)
        step = {'PTS': 5, 'REB': 2}.get(pick['stat'])
        preferred = [line for line in available if step and (line + 0.5) % step == 0]
        target = next(iter(preferred or available), None)
    odds = ladder.get(target, set())
    return {'line': target, 'odds': next(iter(odds)) if len(odds) == 1 else None}


def shopping_prices(pick, ladders):
    name, stat, side = sportsbook_player_key(pick['player']), pick['stat'], pick['side']
    ladder = ladders.get((name, stat, side), {})
    offers = [{'line':line, 'odds':next(iter(prices))} for line, prices in sorted(ladder.items())
              if len(prices) == 1 and (line > pick['line'] if side == 'Over' else line < pick['line'])]
    double = ladders.get((name, 'DD', 'Yes'), {}).get(0, set()) if side == 'Over' else set()
    return {'offers': offers, 'double': next(iter(double)) if len(double) == 1 else None}


def valid_alt_choice(pick):
    choice = pick.get('alt_choice', 'auto')
    if choice in ('auto', 'none'):
        return choice
    if choice == 'dd' and pick['side'] == 'Over':
        return choice
    if isinstance(choice, str) and re.fullmatch(r'line:\d+(?:\.5)?', choice):
        line = float(choice[5:])
        if (line > pick['line'] if pick['side'] == 'Over' else line < pick['line']):
            return choice
    raise ValueError('Choose a valid alternate for this prop.')


def alternate_quotes(picks, fetch=current_event, clock=lambda: datetime.now(timezone.utc), preview=False):
    """Quote alternate choices without requiring or changing the original main price."""
    rows, skipped = [], []
    groups = defaultdict(list)
    for p in picks:
        groups[p['event']].append(p)
    for event_id, group in groups.items():
        try:
            if not re.fullmatch(r'[a-fA-F0-9]{32}', event_id):
                raise ValueError('Event unavailable.')
            event = fetch(event_id, {p['stat'] for p in group}, {p['stat'] for p in group})
            now = clock()
            start = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
            if event['id'] != event_id or start <= now:
                raise ValueError('Game has started or is unavailable.')
            game = f"{TEAM_NAMES[event['away_team']]} @ {TEAM_NAMES[event['home_team']]}"
            ladders = alternate_prices(event, now)
        except (ValueError, KeyError, OddsFeedError):
            skipped.extend(f"{p['player']}: alternate prices unavailable; existing card unchanged." for p in group)
            continue
        for p in group:
            if game != p['game'] or start.astimezone(DENVER).date().isoformat() != p['day']:
                skipped.append(f"{p['player']}: alternate game/date mismatch.")
                continue
            shopping = shopping_prices(p, ladders)
            choice = valid_alt_choice(p)
            if not preview and ((choice == 'dd' and shopping['double'] is None) or
                    (choice.startswith('line:') and not any(o['line'] == float(choice[5:]) for o in shopping['offers']))):
                skipped.append(f"{p['player']}: selected alternate has no fresh FanDuel quote; existing card unchanged.")
                continue
            rows.append({**p, 'start': start.isoformat(), 'quoted': now.isoformat(),
                         'shopping': shopping, 'alt_choice': choice})
    return rows, skipped


def preview_alternates(picks, fetch=current_event):
    rows, skipped = alternate_quotes(picks, fetch, preview=True)
    result = {}
    for p in rows:
        options = []
        step = {'PTS': 5, 'REB': 2}.get(p['stat'], 1)
        offers = sorted(p['shopping']['offers'], key=lambda o: (
            (o['line'] + .5) % step != 0, abs(o['line'] - p['line'])))
        for o in offers:
            line = o['line']
            label = f"{line + .5:g}+ {p['stat']}" if p['side'] == 'Over' and line % 1 == .5 else f"{p['side']} {line:g} {p['stat']}"
            options.append({'value': f'line:{line:g}', 'label': f"{label} · FD {o['odds']:+d}"})
        if p['shopping']['double'] is not None:
            options.append({'value': 'dd', 'label': f"Double-double · FD {p['shopping']['double']:+d}"})
        result[p['id']] = {'options': options, 'quoted': p['quoted']}
    return result, skipped


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
            event = fetch(event_id, {p['stat'] for p in group}, {p['stat'] for p in group})
            now = clock()
            start = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
            if event['id'] != event_id or start <= now:
                raise ValueError('event unavailable or already started')
            game = f"{TEAM_NAMES[event['away_team']]} @ {TEAM_NAMES[event['home_team']]}"
            quotes = game_quotes(event, [p['player'] for p in group], 'fanduel', now)
            ladders = alternate_prices(event, now)
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
            choice = valid_alt_choice(p)
            shopping = shopping_prices(p, ladders)
            if ((choice == 'dd' and shopping['double'] is None) or
                    (choice.startswith('line:') and not any(o['line'] == float(choice[5:]) for o in shopping['offers']))):
                skipped.append(f"{p['player']}: selected alternate has no fresh FanDuel quote; play not sent.")
                continue
            rows.append(dict(id=p['id'], day=p['day'], start=start.isoformat(), game=game,
                             player=p['player'], stat=p['stat'], side=p['side'], line=p['line'],
                             book='FanDuel', alt=next_over(p, ladders), shopping=shopping, alt_choice=choice, odds=price, worst=worst, breakeven=american_probability(worst),
                             quoted=quote['Updated UTC'], sent=now.isoformat()))
    return rows, skipped


def sheet_request(connection, action, day, *, ids=(), rows=(), edits=(), alt_edits=()):
    """Use the private write key only for the owner's selected Google deployment."""
    import requests
    endpoint, separator, key = connection.partition('#')
    if (not separator or not re.fullmatch(r'https://script\.google\.com/macros/s/[\w-]{20,200}/exec', endpoint)
            or not re.fullmatch(r'[a-f0-9]{64}', key)):
        raise ValueError('Paste the connection from the Jae card menu in your Google Sheet.')
    body = dict(key=key, action=action, day=day, ids=list(ids), rows=list(rows), edits=list(edits), alt_edits=list(alt_edits))
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
        return (not item.get('alt_edited', False), not item.get('confidence_edited', False), item.get('confidence_cents', 5))

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
    alt_pending, alt_synced = [], {}
    for p in confirmed:
        if not p.get('alt_edited') or p['id'] not in existing:
            continue
        matching = aliases[p['id']] & sent
        if len(matching) != 1:
            raise ValueError('Multiple sent rows match this prop. Check the card before changing its alternate.')
        stored_id = next(iter(matching))
        if not check.get('alternateEdits'):
            raise ValueError('The sheet needs the alternate update before sending these changes.')
        if check.get('altChoices', {}).get(stored_id) == valid_alt_choice(p):
            alt_synced[p['id']] = valid_alt_choice(p)
        else:
            alt_pending.append(dict(p, id=stored_id))
    alt_edits, alt_skipped = alternate_quotes(alt_pending, fetch)
    skipped.extend(alt_skipped)
    added = updated = alt_updated = 0
    if rows or edits or alt_edits:
        reply = request(connection, 'append', day, rows=rows, **({'edits':edits} if edits else {}),
                        **({'alt_edits': alt_edits} if alt_edits else {}))
        added = reply['added']
        if type(added) is not int or not 0 <= added <= len(rows):
            raise ValueError('Google did not confirm the row count. Retry to check the sheet.')
        updated = reply.get('updated',0) if not edits else reply.get('updated')
        if type(updated) is not int or not 0 <= updated <= len(edits):
            raise ValueError('Google did not confirm the confidence updates. Retry to check the sheet.')
        alt_updated = reply.get('altUpdated', 0)
        if type(alt_updated) is not int or not 0 <= alt_updated <= len(alt_edits):
            raise ValueError('Google did not confirm the alternate updates. Retry to check the sheet.')
        accepted = set(reply.get('altAccepted', []))
        for p in confirmed:
            if aliases[p['id']] & accepted:
                alt_synced[p['id']] = valid_alt_choice(p)
        skipped.extend(reply.get('altSkipped', []))
    for p in confirmed:
        if p['id'] in new_ids:
            alt_synced[p['id']] = valid_alt_choice(p)
    alt_synced = {source: alt_synced[pid] for pid in alt_synced for source, _ in source_ids[pid]
                  if unique[pid].get('alt_choice', 'auto') == next(p.get('alt_choice', 'auto') for p in picks if p['id'] == source)}
    synced = {source:cents for pid in delivered for source,cents in source_ids[pid]}
    message = f'{added} added to {day} · {updated} updated · {len(existing)} already sent.'
    if alt_updated:
        message += f' {alt_updated} alternate(s) updated.'
    return message, skipped, synced, alt_synced
