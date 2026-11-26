"""Current same-book, same-event player props and model-implied stat means."""
from datetime import datetime, timezone
from functools import lru_cache
import math
import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq
from scipy.stats import poisson
from nba_api.stats.static.teams import get_teams
from live_projections import name_key
from saved_sportsbook_lines import SOURCE, SAVED, quote_is_saved

BASE = 'https://api.the-odds-api.com/v4/sports/basketball_nba'
BOOKS = {'FanDuel':'fanduel','Propbuilder (BetOnline)':'betonlineag',
         'DraftKings':'draftkings','BetMGM':'betmgm'}
# Keep retired books valid for saved plays and sent-card deduplication.
# Only connected, enabled books belong in the projection-page menus.
SELECTABLE_BOOKS = {'FanDuel':'fanduel','Propbuilder (BetOnline)':'betonlineag'}
MARKETS = {'player_points':'PTS','player_rebounds':'REB','player_assists':'AST',
           'player_threes':'3PM','player_steals':'STL','player_blocks':'BLK',
           'player_turnovers':'TOV','player_points_rebounds_assists':'PRA'}
STAT_ORDER = ['PTS','REB','AST','3PM','STL','BLK','TOV','PRA']
FANDUEL_THREES_OVER_ONLY = 'FanDuel threes milestone — Over only'
PLAY_THRESHOLD_PCT = 5.
MIN_STEALS_BLOCKS_UNDER_ODDS = -160
# Schedule and book feeds can disagree by 20 minutes (BOS–DEN, Nov 18, 2026).
# Share this bounded window across page matching and completed-box grading.
EVENT_TIME_TOLERANCE = pd.Timedelta(minutes=30)
TEAM_NAMES = {team['full_name']:team['abbreviation'] for team in get_teams()}
TEAM_NAMES.update({'Los Angeles Clippers':'LAC','LA Clippers':'LAC'})
# Verified FanDuel aliases from the November 20 league-wide quote audit.
# Only book lookups use these; roster names, NBA IDs and saved picks stay intact.
SPORTSBOOK_PLAYER_ALIASES = {
    'herb jones': 'herbert jones',          # NBA 1630529
    'paul reed jr': 'paul reed',           # NBA 1630194
    'derrick jones': 'derrick jones jr',    # NBA 1627884
}


def sportsbook_player_key(name):
    """Resolve reviewed aliases without fuzzy matching or stripping all suffixes."""
    normalized = name_key(name)
    return SPORTSBOOK_PLAYER_ALIASES.get(normalized, normalized)


class OddsFeedError(Exception):
    pass


def read_feed(path, api_key, **params):
    # Never expose request URLs/exceptions: credentials are query parameters.
    try:
        response = requests.get(BASE+path,params={'apiKey':api_key,**params},timeout=(3,10))
        if not response.ok:
            raise OddsFeedError(f'Sportsbook feed unavailable (HTTP {response.status_code}).')
        return response.json()
    except (requests.RequestException,ValueError):
        raise OddsFeedError('Sportsbook feed could not be reached.') from None


def next_matchup(events, team, opponent, now=None, scheduled_start=None,
                 time_tolerance=EVENT_TIME_TOLERANCE):
    now = now or datetime.now(timezone.utc)
    scheduled_start = pd.Timestamp(scheduled_start) if scheduled_start is not None else None
    matches=[]
    for event in events:
        try:
            start = datetime.fromisoformat(event['commence_time'].replace('Z','+00:00'))
            if scheduled_start is not None and abs(pd.Timestamp(start)-scheduled_start) > time_tolerance:
                continue
            pair={TEAM_NAMES.get(event['home_team']),TEAM_NAMES.get(event['away_team'])}
            if pair=={team,opponent} and start>now:
                matches.append(event)
        except (KeyError,ValueError,TypeError):
            continue
    return min(matches,key=lambda e:e['commence_time']) if matches else None


def american_probability(price):
    price=float(price)
    if not np.isfinite(price) or abs(price)<100:
        raise ValueError('Expected valid American odds (at least +100 or at most -100)')
    return -price/(100-price) if price<0 else 100/(price+100)


def fair_over_probability(over, under):
    p,q=american_probability(over),american_probability(under)
    return p/(p+q)


def projection_markets(bookmaker):
    """Include FD's X+ threes offers, which may be absent from its main market."""
    return ','.join([*MARKETS, *(['player_threes_alternate'] if bookmaker == 'fanduel' else [])])


@lru_cache(maxsize=4096)
def implied_total(line, over, under):
    """Solve for a Poisson mean using paired, proportional no-vig prices.

    On integer lines, over/under prices describe probabilities conditional on no
    push. Half-point lines use P(X > floor(line)). Mean is model-dependent; one
    threshold and two prices do not uniquely identify a real-world distribution.
    """
    line=float(line)
    if not np.isfinite(line) or line<0 or abs(line*2-round(line*2))>1e-8:
        raise ValueError('Expected a nonnegative integer or half-point prop line')
    if line==0:
        raise ValueError('A zero integer line cannot identify an over/under mean')
    return _implied_total(line, fair_over_probability(over,under))


@lru_cache(maxsize=4096)
def _implied_total(line, target):
    integer=line.is_integer()
    k=math.floor(line)
    def over_probability(mu):
        if integer:
            # P(over | no push), evaluated from the two tails for stability.
            upper=poisson.sf(k,mu); lower=poisson.cdf(k-1,mu)
            return upper/(upper+lower)
        return poisson.sf(k,mu)
    high=max(10.,line*2+10)
    while over_probability(high)<target:
        high*=2
    return float(brentq(lambda mu:over_probability(mu)-target,1e-9,high))


def _empty_quotes():
    return {stat:{'Stat':stat,'Book line':np.nan,'Over odds':np.nan,'Under odds':np.nan,
                'Book implied total':np.nan,'Fair over probability':np.nan,'Updated UTC':'',
                'Market status':'No posted line','Saved UTC':'','Quote basis':''} for stat in STAT_ORDER}


def _fanduel_threes_fallback(payload, rows, targets, now, max_age_seconds):
    """Use one missing-main Over offer, never create an unquoted Under.

    Select the threshold closest to even money without consulting projections.
    Retain the quoted margin in the implied mean: a one-sided offer cannot be
    de-vigged. This makes the Over comparison more conservative than removing vig.
    """
    offers = {}
    for book in payload.get('bookmakers', []):
        if book.get('key') != 'fanduel':
            continue
        for market in book.get('markets', []):
            if market.get('key') != 'player_threes_alternate':
                continue
            updated = market.get('last_update', book.get('last_update', ''))
            for outcome in market.get('outcomes', []):
                matches = targets.get(sportsbook_player_key(outcome.get('description', '')), [])
                if len(matches) != 1 or outcome.get('name') != 'Over':
                    continue
                try:
                    line, price = float(outcome['point']), float(outcome['price'])
                    american_probability(price)
                    # X+ offers translate to positive half-point lines only.
                    if not np.isfinite(line) or line < .5 or not (line + .5).is_integer():
                        continue
                    offers.setdefault(matches[0], {}).setdefault(line, set()).add((price, updated))
                except (KeyError, TypeError, ValueError):
                    continue
    for player, lines in offers.items():
        row = rows[player]['3PM']
        if row['Market status'] != 'No posted line':
            continue  # Never replace a main line, even an ambiguous or stale one.
        if any(len(quotes) != 1 for quotes in lines.values()):
            row['Market status'] = 'Ambiguous alternate prices'
            continue
        ranked = sorted((abs(american_probability(price) - .5), line, price, updated)
                        for line, quotes in lines.items() for price, updated in quotes)
        if len(ranked) > 1 and abs(ranked[0][0] - ranked[1][0]) < 1e-10:
            row['Market status'] = 'Ambiguous alternate lines'
            continue
        _, line, price, updated = ranked[0]
        row.update({'Book line':line, 'Over odds':price, 'Updated UTC':updated,
                    'Quote basis':FANDUEL_THREES_OVER_ONLY})
        try:
            age = (now - datetime.fromisoformat(updated.replace('Z', '+00:00'))).total_seconds()
            saved = quote_is_saved(payload, updated, now, max_age_seconds)
            if (age < -60 or (age > max_age_seconds and not saved) or
                    (payload.get(SOURCE, {}).get('fallback') and not saved)):
                row['Market status'] = 'Stale quote — comparison unavailable'
                continue
            row['Book implied total'] = _implied_total(line, american_probability(price))
            row['Market status'] = SAVED if saved else 'Current'
            row['Saved UTC'] = payload.get(SOURCE, {}).get('saved_at', '')
        except (ValueError, TypeError, OverflowError):
            row['Market status'] = 'Unusable quote — comparison unavailable'


def game_quotes(payload, player_names, book_key, now=None, max_age_seconds=900):
    """Parse each outcome once and apply the freshness check at this render's clock."""
    now = now or datetime.now(timezone.utc)
    rows = {name_key(name): _empty_quotes() for name in player_names}
    targets = {}
    for player in rows:
        canonical = sportsbook_player_key(player)
        targets.setdefault(canonical, []).append(player)
    for book in payload.get('bookmakers',[]):
        if book.get('key')!=book_key:
            continue
        for market in book.get('markets',[]):
            stat=MARKETS.get(market.get('key'))
            if stat is None:
                continue
            by_player={}
            for outcome in market.get('outcomes',[]):
                player = name_key(outcome.get('description',''))
                candidates = targets.get(sportsbook_player_key(player), [])
                # Never choose between two roster profiles sharing an alias.
                if len(candidates) != 1:
                    continue
                player = candidates[0]
                if outcome.get('name') not in ['Over','Under']:
                    continue
                try:
                    point=float(outcome['point']); price=float(outcome['price'])
                    american_probability(price)
                    candidates = by_player.setdefault(player, {})
                    candidates.setdefault(point,{}).setdefault(outcome['name'],[]).append(price)
                except (ValueError,KeyError,TypeError):
                    continue
            for player, candidates in by_player.items():
                complete=[(point,sides) for point,sides in candidates.items()
                          if set(sides)=={'Over','Under'} and all(len(v)==1 for v in sides.values())]
                # Do not merge alternate/mismatched lines or silently pick between them.
                if len(complete)!=1:
                    if candidates:
                        rows[player][stat]['Market status']='Needs matching over/under prices' if not complete else 'Ambiguous lines'
                    continue
                line,sides=complete[0]
                over,under=sides['Over'][0],sides['Under'][0]
                row=rows[player][stat]
                row.update({'Book line':line,'Over odds':over,'Under odds':under,
                            'Quote basis':'Paired Over/Under — no vig',
                            'Updated UTC':market.get('last_update',book.get('last_update',''))})
                try:
                    updated=datetime.fromisoformat(row['Updated UTC'].replace('Z','+00:00'))
                    age=(now-updated).total_seconds()
                    saved = quote_is_saved(payload, row['Updated UTC'], now, max_age_seconds)
                    if (age < -60 or (age > max_age_seconds and not saved) or
                            (payload.get(SOURCE, {}).get('fallback') and not saved)):
                        row['Market status']='Stale quote — comparison unavailable'
                        continue
                    row['Book implied total']=implied_total(line,over,under)
                    row['Fair over probability']=fair_over_probability(over,under)
                    row['Market status']=SAVED if saved else 'Current'
                    row['Saved UTC']=payload.get(SOURCE, {}).get('saved_at', '')
                except (ValueError,TypeError,OverflowError):
                    row['Market status']='Unusable quote — comparison unavailable'
    if book_key == 'fanduel':
        _fanduel_threes_fallback(payload, rows, targets, now, max_age_seconds)
    return {player: pd.DataFrame(markets.values()) for player, markets in rows.items()}


def player_quotes(payload, player_name, book_key, now=None, max_age_seconds=900):
    return game_quotes(payload, [player_name], book_key, now, max_age_seconds)[name_key(player_name)]


def signed_edge_pct(projected, implied):
    """Use the displayed two-decimal edge for play eligibility."""
    return np.round((projected / implied - 1) * 100, 2)


def compare_with_book(preview, quotes):
    comparison=preview.merge(quotes,on='Stat',how='left',validate='one_to_one')
    implied=comparison['Book implied total']
    valid=implied.gt(0) & np.isfinite(implied) & np.isfinite(comparison.Projected)
    if 'Quote basis' in comparison:
        over_only = comparison['Quote basis'].eq(FANDUEL_THREES_OVER_ONLY)
        valid &= ~over_only | comparison.Projected.ge(implied)
    # An Under uses the selected book's Under price, even when the table
    # displays the Over price beside the line. Keep the projection and quote.
    stocks_under = comparison.Stat.isin(['STL', 'BLK']) & comparison.Projected.lt(implied)
    valid &= ~stocks_under | (comparison.get('Under odds', np.nan) >= MIN_STEALS_BLOCKS_UNDER_ODDS)
    comparison['Signed edge %']=np.nan
    comparison.loc[valid,'Signed edge %']=signed_edge_pct(comparison.loc[valid,'Projected'], implied[valid])
    comparison['Edge %']=comparison['Signed edge %'].abs()
    comparison['Play']=comparison['Signed edge %'].map(
        lambda edge: 'Over' if edge>PLAY_THRESHOLD_PCT else 'Under' if edge < -PLAY_THRESHOLD_PCT else '')
    return comparison


def comparison_style(row):
    edge=row['Edge %']
    if pd.isna(edge) or abs(edge)<=5:
        return ['' for _ in row.index]
    # Continuous scale: pale at the 5% cutoff, saturated at 25% and beyond.
    strength=float(np.clip((abs(edge)-5)/20,0,1))
    is_over=row['Play']=='Over'
    pale=np.array([224,244,230] if is_over else [253,230,231])
    deep=np.array([25,135,65] if is_over else [194,40,56])
    rgb=np.rint(pale+(deep-pale)*strength).astype(int)
    background='#'+''.join(f'{value:02x}' for value in rgb)
    foreground='#ffffff' if strength>=0.65 else '#143c24' if is_over else '#701823'
    style=f'background-color: {background}; color: {foreground}'
    return [style if column in ['Edge %','Play'] else '' for column in row.index]
