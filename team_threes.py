"""Team three-point targets with normalized archetype shares and fixed points."""
import numpy as np
import pandas as pd

from team_assists import historical_assist_rates, assist_target


def historical_three_rates(logs, expected_games=None):
    """Use the same complete-game validation and ratio aggregation as assists.

    Here the numerator is three-point makes, not assists. All appearances count.
    """
    columns = ['GAME_ID', 'Team', 'Opponent', 'FG3M', 'FGM', 'PTS']
    if not set(columns).issubset(logs):
        raise ValueError('Historical three-point data unavailable')
    if logs.FG3M.gt(logs.FGM).any():
        raise ValueError('Historical three-point makes exceed field goals')
    rates = historical_assist_rates(logs[columns].rename(columns={'FG3M': 'AST'}), expected_games)
    return rates.rename(columns={'Assist rate': 'Three rate', 'Allowed assist rate': 'Allowed three rate'})


def three_target(rates, team, opponent, market_points):
    """Expected FGM times the 50/50 own/allowed share of made baskets from three."""
    return assist_target(rates.rename(columns={
        'Three rate': 'Assist rate', 'Allowed three rate': 'Allowed assist rate',
    }), team, opponent, market_points)


def apply_team_threes(projections, point_targets, rates, opponents):
    """Allocate the team target using current minute/archetype-adjusted shares.

    Equivalent to team-first baseline shares followed by archetype modifiers
    divided by their baseline-three-weighted team average. Shared effects cancel.
    Supporting attempts scale equally to preserve conversion. PTS/PRA stay fixed.
    """
    result = projections.copy(deep=True)
    result['Pre-team 3PM'] = result.FG3M
    result['Pre-team 3PA'] = result.FG3A
    result['Team 3PM factor'] = 1.
    for column in ['Team 3PM target', 'Team three rate', 'Three-point share']:
        result[column] = np.nan
    if 'Expected team FGM' not in result:
        result['Expected team FGM'] = np.nan
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
            if original <= 0:
                raise ValueError('No positive player three-point production to allocate')
            target, rate, field_goals = three_target(rates, team, opponents[team], point_targets[team])
            shares = active.FG3M / original
            makes = shares * target
            if (3 * makes).gt(active.PTS + 1e-9).any():
                raise ValueError('Team target would put a player’s three-point scoring above total points')
            result.loc[mask, 'FG3M'] = makes
            result.loc[mask, 'FG3A'] = active.FG3A * (target / original)
            result.loc[mask, 'Team 3PM factor'] = target / original
            result.loc[mask, 'Team 3PM target'] = target
            result.loc[mask, 'Team three rate'] = rate
            result.loc[mask, 'Expected team FGM'] = field_goals
            result.loc[mask, 'Three-point share'] = shares
            summaries.append({'Team': team, 'Original 3PM': original, 'Team 3PM target': target,
                              'Factor': target / original, 'Team three rate': rate,
                              'Expected team FGM': field_goals})
        except ValueError as exc:
            issues.append(f'{team}: {exc}')
    return result, pd.DataFrame(summaries), issues
