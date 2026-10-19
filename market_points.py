"""Full market team-total matching for points only; PRA follows PTS; other components stay intact."""
from datetime import datetime, timezone
import math
from sportsbook_lines import TEAM_NAMES, american_probability
import pandas as pd


def team_targets(total, home_spread, home, away):
    if not all(math.isfinite(x) for x in [total,home_spread]) or total <= abs(home_spread):
        raise ValueError('Valid positive game total and home-team spread required')
    return {home:(total-home_spread)/2,away:(total+home_spread)/2}


def match_points(projections, targets):
    adjusted=projections.copy(deep=True)
    summaries=[]
    for team,target in targets.items():
        if adjusted.loc[adjusted.Team.eq(team),'Projected minutes'].isna().any():
            raise ValueError(f'{team}: unavailable projected minutes')
        mask=adjusted.Team.eq(team) & adjusted['Projected minutes'].gt(0)
        before=adjusted.loc[mask,'PTS']
        if before.empty or before.isna().any() or not before.map(math.isfinite).all() or before.lt(0).any() or before.sum()<=0:
            raise ValueError(f'{team}: incomplete team points coverage; no scaling applied')
        if not math.isfinite(target) or target <= 0:
            raise ValueError('Positive market-implied team total required')
        factor=target/before.sum()
        if not {'REB','AST'}.issubset(adjusted.columns):
            raise ValueError(f'{team}: rebounds and assists required to update PRA')
        adjusted.loc[mask,'PTS']=before*factor
        adjusted.loc[mask,'PRA']=adjusted.loc[mask,'PTS']+adjusted.loc[mask,'REB']+adjusted.loc[mask,'AST']
        summaries.append({'Team':team,'Original PTS':before.sum(),'Market team total':target,'Factor':factor,'Adjusted PTS':adjusted.loc[mask,'PTS'].sum(),'Projected minutes':adjusted.loc[mask,'Projected minutes'].sum()})
    return adjusted,pd.DataFrame(summaries)


def market_targets(payload, event, bookmaker, now=None):
    """Require fresh matching spread/total markets for this exact event and book."""
    now=now or datetime.now(timezone.utc)
    if payload.get('id') != event['id'] or any(payload.get(k)!=event[k] for k in ['home_team','away_team']):
        raise ValueError('Game market event mismatch')
    start=datetime.fromisoformat(event['commence_time'].replace('Z','+00:00'))
    if start <= now:
        raise ValueError('Game has already started')
    books=[b for b in payload.get('bookmakers',[]) if b.get('key')==bookmaker]
    if len(books)!=1:
        raise ValueError('Selected sportsbook has no game lines')
    markets={}
    for key in ['spreads','totals']:
        found=[m for m in books[0].get('markets',[]) if m.get('key')==key]
        if len(found)!=1:
            raise ValueError('A matching spread and total are required')
        market=found[0]
        updated=datetime.fromisoformat(market.get('last_update',books[0].get('last_update','')).replace('Z','+00:00'))
        if not -60 <= (now-updated).total_seconds() <= 900:
            raise ValueError('Game lines are stale')
        if len(market.get('outcomes',[]))!=2:
            raise ValueError('Ambiguous game lines')
        for outcome in market['outcomes']:
            american_probability(outcome['price'])
        markets[key]={x['name']:float(x['point']) for x in market['outcomes']}
    spread,total=markets['spreads'],markets['totals']
    if set(spread)!={event['home_team'],event['away_team']} or set(total)!={'Over','Under'}:
        raise ValueError('Unpaired game lines')
    if not math.isclose(spread[event['home_team']],-spread[event['away_team']]) or total['Over']!=total['Under']:
        raise ValueError('Game lines do not match')
    return team_targets(total['Over'],spread[event['home_team']],TEAM_NAMES[event['home_team']],TEAM_NAMES[event['away_team']])


def apply_market_points(projections, targets):
    """Fail safely per team; retain opponent-only PTS for rating and audit."""
    result=projections.copy(deep=True)
    result['Pre-market PTS']=result.PTS
    result['Pre-market PRA']=result.PRA
    result['Market PTS factor']=1.
    result['Market team total']=float('nan')
    summaries=[]
    issues=[]
    for team in projections.Team.unique():
        if team not in targets:
            issues.append(f'{team}: no market team total')
            continue
        mask=projections.Team.eq(team)
        try:
            scaled,summary=match_points(projections.loc[mask],{team:targets[team]})
            result.loc[mask,'PTS']=scaled.PTS
            result.loc[mask,'PRA']=scaled.PRA
            result.loc[mask,'Market PTS factor']=summary.iloc[0].Factor
            result.loc[mask,'Market team total']=targets[team]
            summaries.append(summary)
        except ValueError as exc:
            issues.append(str(exc))
    return result,pd.concat(summaries,ignore_index=True) if summaries else pd.DataFrame(),issues
