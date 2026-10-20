"""Current same-book, same-event player props and model-implied stat means."""
from datetime import datetime, timezone
import math
import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq
from scipy.stats import poisson
from nba_api.stats.static.teams import get_teams
from live_projections import name_key

BASE = 'https://api.the-odds-api.com/v4/sports/basketball_nba'
BOOKS = {'FanDuel':'fanduel','DraftKings':'draftkings','BetMGM':'betmgm'}
MARKETS = {'player_points':'PTS','player_rebounds':'REB','player_assists':'AST',
           'player_threes':'3PM','player_steals':'STL','player_blocks':'BLK',
           'player_turnovers':'TOV','player_points_rebounds_assists':'PRA'}
STAT_ORDER = ['PTS','REB','AST','3PM','STL','BLK','TOV','PRA']
TEAM_NAMES = {team['full_name']:team['abbreviation'] for team in get_teams()}
TEAM_NAMES.update({'Los Angeles Clippers':'LAC','LA Clippers':'LAC'})


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
                 time_tolerance=pd.Timedelta(minutes=15)):
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
    target=fair_over_probability(over,under)
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


def player_quotes(payload, player_name, book_key, now=None, max_age_seconds=900):
    now=now or datetime.now(timezone.utc)
    rows={stat:{'Stat':stat,'Book line':np.nan,'Over odds':np.nan,'Under odds':np.nan,
                'Book implied total':np.nan,'Fair over probability':np.nan,'Updated UTC':'',
                'Market status':'No posted line'} for stat in STAT_ORDER}
    for book in payload.get('bookmakers',[]):
        if book.get('key')!=book_key:
            continue
        for market in book.get('markets',[]):
            stat=MARKETS.get(market.get('key'))
            if stat is None:
                continue
            candidates={}
            for outcome in market.get('outcomes',[]):
                if name_key(outcome.get('description',''))!=name_key(player_name):
                    continue
                if outcome.get('name') not in ['Over','Under']:
                    continue
                try:
                    point=float(outcome['point']); price=float(outcome['price'])
                    american_probability(price)
                    candidates.setdefault(point,{}).setdefault(outcome['name'],[]).append(price)
                except (ValueError,KeyError,TypeError):
                    continue
            complete=[(point,sides) for point,sides in candidates.items()
                      if set(sides)=={'Over','Under'} and all(len(v)==1 for v in sides.values())]
            # Do not merge alternate/mismatched lines or silently pick between them.
            if len(complete)!=1:
                if candidates:
                    rows[stat]['Market status']='Needs matching over/under prices' if not complete else 'Ambiguous lines'
                continue
            line,sides=complete[0]
            over,under=sides['Over'][0],sides['Under'][0]
            row=rows[stat]
            row.update({'Book line':line,'Over odds':over,'Under odds':under,
                        'Updated UTC':market.get('last_update',book.get('last_update',''))})
            try:
                updated=datetime.fromisoformat(row['Updated UTC'].replace('Z','+00:00'))
                age=(now-updated).total_seconds()
                if not -60<=age<=max_age_seconds:
                    row['Market status']='Stale quote — comparison unavailable'
                    continue
                row['Book implied total']=implied_total(line,over,under)
                row['Fair over probability']=fair_over_probability(over,under)
                row['Market status']='Current'
            except (ValueError,TypeError,OverflowError):
                row['Market status']='Unusable quote — comparison unavailable'
    return pd.DataFrame(rows.values())


def compare_with_book(preview, quotes):
    comparison=preview.merge(quotes,on='Stat',how='left',validate='one_to_one')
    implied=comparison['Book implied total']
    valid=implied.gt(0) & np.isfinite(implied) & np.isfinite(comparison.Projected)
    comparison['Signed edge %']=np.nan
    comparison.loc[valid,'Signed edge %']=(comparison.loc[valid,'Projected']/implied[valid]-1)*100
    # Round for consistent display and threshold decisions: exactly 5% is not >5%.
    comparison['Signed edge %']=comparison['Signed edge %'].round(2)
    comparison['Edge %']=comparison['Signed edge %'].abs()
    comparison['Play']=comparison['Signed edge %'].map(
        lambda edge: 'Over' if edge>5 else 'Under' if edge < -5 else '')
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
