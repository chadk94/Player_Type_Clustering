"""Team three-point targets with normalized archetype shares and fixed points."""
import numpy as np
import pandas as pd

from team_assists import historical_assist_rates


def historical_three_rates(logs, expected_games=None):
    """Pooled 3PA/FGA and PTS/FGA, including low/rounded-zero-minute rows."""
    columns = ['GAME_ID', 'Team', 'Opponent', 'FG3A', 'FG3M', 'FGA', 'FGM', 'PTS']
    if not set(columns).issubset(logs):
        raise ValueError('Historical three-point data unavailable')
    values = logs[['FG3A','FG3M','FGA','FGM','PTS']]
    if (not np.isfinite(values).all().all() or values.lt(0).any().any()
            or logs.FG3M.gt(logs.FG3A).any() or logs.FG3A.gt(logs.FGA).any()
            or logs.FG3M.gt(logs.FGM).any() or logs.FGM.gt(logs.FGA).any()):
        raise ValueError('Invalid historical shooting totals')
    rates = historical_assist_rates(logs[['GAME_ID','Team','Opponent','FG3A','FGA','PTS']]
                                   .rename(columns={'FG3A':'AST','FGA':'FGM'}), expected_games)
    return rates.rename(columns={'Assist rate':'Three attempt rate','Allowed assist rate':'Allowed three attempt rate',
                                 'Points per FGM':'Points per FGA','Allowed points per FGM':'Allowed points per FGA'})


def three_target(rates, team, opponent, market_points):
    """Return team attempts, 3PA/FGA share and expected FGA; never a makes target."""
    if team == opponent or team not in rates.index or opponent not in rates.index:
        raise ValueError('Team/opponent three-attempt rates unavailable')
    shares = [rates.loc[team,'Three attempt rate'],rates.loc[opponent,'Allowed three attempt rate']]
    conversions = [rates.loc[team,'Points per FGA'],rates.loc[opponent,'Allowed points per FGA']]
    if (not all(np.isfinite(x) and 0 <= x <= 1 for x in shares)
            or not all(np.isfinite(x) and x > 0 for x in conversions)
            or not np.isfinite(market_points) or market_points <= 0):
        raise ValueError('Invalid attempt rates or market points')
    rate = sum(shares)/2
    attempts = market_points/(sum(conversions)/2)
    return attempts*rate, rate, attempts


def apply_team_threes(projections, point_targets, rates, opponents):
    """Allocate the team target using current minute/archetype-adjusted shares.

    Equivalent to team-first baseline shares followed by archetype modifiers
    divided by their baseline-attempt-weighted team average. Shared effects cancel.
    Allocate attempts, retaining each player's expected 3P%. PTS/PRA stay fixed.
    """
    result = projections.copy(deep=True)
    result['Pre-team 3PM'] = result.FG3M
    result['Pre-team 3PA'] = result.FG3A
    result['Team 3PM factor'] = 1.
    for column in ['Team 3PM target', 'Team 3PA target', 'Team three attempt rate', 'Expected team FGA', 'Three-point share']:
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
            values = active[['FG3M', 'FG3A', 'PTS']]
            if active.empty or not np.isfinite(values).all().all() or values.lt(0).any().any():
                raise ValueError('Incomplete player threes/attempts/points coverage')
            if active.FG3M.gt(active.FG3A).any():
                raise ValueError('Three-point makes exceed attempts')
            original = active.FG3M.sum()
            original_attempts = active.FG3A.sum()
            if original_attempts <= 0:
                raise ValueError('No positive player three-point attempts to allocate')
            target, rate, field_goals = three_target(rates, team, opponents[team], point_targets[team])
            shares = active.FG3A / original_attempts
            factor = target / original_attempts
            makes = active.FG3M * factor
            if (3 * makes).gt(active.PTS + 1e-9).any():
                raise ValueError('Team target would put a player’s three-point scoring above total points')
            result.loc[mask, 'FG3M'] = makes
            result.loc[mask, 'FG3A'] = shares * target
            result.loc[mask, 'Team 3PM factor'] = factor
            result.loc[mask, 'Team 3PM target'] = makes.sum()
            result.loc[mask, 'Team 3PA target'] = target
            result.loc[mask, 'Team three attempt rate'] = rate
            result.loc[mask, 'Expected team FGA'] = field_goals
            result.loc[mask, 'Three-point share'] = shares
            summaries.append({'Team': team, 'Original 3PM': original, 'Team 3PM target': makes.sum(),
                              'Original 3PA':original_attempts, 'Team 3PA target':target,
                              'Factor': factor, 'Team three attempt rate': rate,
                              'Expected team FGA': field_goals})
        except (ValueError, KeyError, TypeError) as exc:
            issues.append(f'{team}: {exc}')
    return result, pd.DataFrame(summaries), issues
