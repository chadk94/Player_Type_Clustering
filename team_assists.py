"""Team assist targets with normalized, opponent-adjusted player shares."""
import math

import numpy as np
import pandas as pd


def historical_assist_rates(logs, expected_games=None):
    """Aggregate every appearance; do not apply the archetype five-minute filter."""
    required = ['GAME_ID', 'Team', 'Opponent', 'AST', 'FGM', 'PTS']
    if logs.empty or not set(required).issubset(logs):
        raise ValueError('Historical team assist data unavailable')
    if logs[required].isna().any().any():
        raise ValueError('Incomplete historical assist data')
    values = logs[['AST', 'FGM', 'PTS']]
    if not np.isfinite(values).all().all() or values.lt(0).any().any():
        raise ValueError('Invalid historical assist data')
    boxes = logs.groupby(['GAME_ID', 'Team', 'Opponent'], as_index=False)[['AST', 'FGM', 'PTS']].sum()
    if not boxes.groupby('GAME_ID').size().eq(2).all():
        raise ValueError('Historical games must have both teams')
    reverse = boxes[['GAME_ID', 'Team', 'Opponent']].rename(columns={'Team': 'Opponent', 'Opponent': 'Team'})
    paired = boxes.merge(reverse, on=['GAME_ID', 'Team', 'Opponent'], validate='one_to_one')
    if len(paired) != len(boxes) or boxes.Team.eq(boxes.Opponent).any():
        raise ValueError('Historical opponent pairs do not match')
    if boxes.FGM.le(0).any() or boxes.PTS.le(0).any() or boxes.AST.gt(boxes.FGM).any():
        raise ValueError('Invalid historical team totals')
    counts = boxes.groupby('Team').size()
    if expected_games is not None and not counts.eq(expected_games).all():
        raise ValueError('Incomplete historical season')
    own = boxes.groupby('Team')[['AST', 'FGM', 'PTS']].sum()
    allowed = boxes.groupby('Opponent')[['AST', 'FGM', 'PTS']].sum()
    return pd.DataFrame({
        'Assist rate': own.AST / own.FGM,
        'Allowed assist rate': allowed.AST / allowed.FGM,
        'Points per FGM': own.PTS / own.FGM,
        'Allowed points per FGM': allowed.PTS / allowed.FGM,
        'Games': counts,
    })


def assist_target(rates, team, opponent, market_points):
    """50/50 own/allowed rates, using total points per FGM including free throws."""
    if team == opponent or team not in rates.index or opponent not in rates.index:
        raise ValueError('Historical team/opponent assist rates unavailable')
    own, allowed = rates.loc[team], rates.loc[opponent]
    rates_to_blend = [own['Assist rate'], allowed['Allowed assist rate']]
    conversions = [own['Points per FGM'], allowed['Allowed points per FGM']]
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in rates_to_blend):
        raise ValueError('Invalid assist rates')
    if not all(math.isfinite(x) and x >= 2 for x in conversions):
        raise ValueError('Invalid points-to-field-goals ratios')
    if not math.isfinite(market_points) or market_points <= 0:
        raise ValueError('Valid market team points required')
    rate = sum(rates_to_blend) / 2
    field_goals = market_points / (sum(conversions) / 2)
    return rate * field_goals, rate, field_goals


def apply_team_assists(projections, point_targets, rates, opponents):
    """Normalize existing archetype-adjusted AST to one team target, exactly once.

    A common opponent multiplier cancels in each player's AST / team AST share.
    Relative archetype differences survive. No second opponent multiplier follows.
    """
    result = projections.copy(deep=True)
    result['Pre-team AST'] = result.AST
    result['Team AST factor'] = 1.
    for column in ['Team AST target', 'Team assist rate', 'Expected team FGM', 'Assist share']:
        result[column] = np.nan
    summaries, issues = [], []
    for team in result.Team.unique():
        try:
            if team not in point_targets or team not in opponents:
                raise ValueError('No verified market team total/opponent')
            team_mask = result.Team.eq(team)
            minutes = result.loc[team_mask, 'Projected minutes']
            if not np.isfinite(minutes).all() or minutes.lt(0).any():
                raise ValueError('Incomplete projected minutes')
            mask = team_mask & result['Projected minutes'].gt(0)
            active = result.loc[mask]
            if 'Opponent' in active and not active.Opponent.eq(opponents[team]).all():
                raise ValueError('Projection opponent does not match game')
            values = active[['AST', 'PTS', 'REB']]
            if active.empty or not np.isfinite(values).all().all() or values.lt(0).any().any():
                raise ValueError('Incomplete player assist/points/rebounds coverage')
            original = active.AST.sum()
            if original <= 0:
                raise ValueError('No positive player assist production to allocate')
            target, rate, field_goals = assist_target(rates, team, opponents[team], point_targets[team])
            shares = active.AST / original
            result.loc[mask, 'AST'] = shares * target
            result.loc[mask, 'PRA'] = result.loc[mask, 'PTS'] + result.loc[mask, 'REB'] + result.loc[mask, 'AST']
            result.loc[mask, 'Team AST factor'] = target / original
            result.loc[mask, 'Team AST target'] = target
            result.loc[mask, 'Team assist rate'] = rate
            result.loc[mask, 'Expected team FGM'] = field_goals
            result.loc[mask, 'Assist share'] = shares
            summaries.append({'Team': team, 'Original AST': original, 'Team AST target': target,
                              'Factor': target / original, 'Team assist rate': rate,
                              'Expected team FGM': field_goals})
        except ValueError as exc:
            issues.append(f'{team}: {exc}')
    return result, pd.DataFrame(summaries), issues
