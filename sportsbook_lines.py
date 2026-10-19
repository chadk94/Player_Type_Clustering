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


def next_matchup(events, team, opponent, now=None):
    now = now or datetime.now(timezone.utc)
    matches=[]
    for event in events:
        try:
            start = datetime.fromisoformat(event['commence_time'].replace('Z','+00:00'))
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
    comparison['vs implied']=comparison.Projected-comparison['Book implied total']
    return comparison


def comparison_style(row):
    edge=row['vs implied']
    color='' if pd.isna(edge) else 'background-color: #c8eed2; color: #155b2a' if edge>0 else 'background-color: #f6d2d6; color: #8b1f32' if edge<0 else 'background-color: #eeeeee; color: #333333'
    return [color if column in ['Projected','vs implied'] else '' for column in row.index]
