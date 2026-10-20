"""Credited rebound pools with separately normalized offensive/defensive shares."""
import numpy as np
import pandas as pd

REBOUND_METADATA = [
    'Pre-team OREB', 'Pre-team DREB', 'Pre-team REB', 'Pre-rebound PRA',
    'Team OREB target', 'Team DREB target', 'Team REB target',
    'Team OREB factor', 'Team DREB factor', 'OREB share', 'DREB share',
    'Rebound expected FGM', 'Expected FG misses', 'Expected FT misses',
    'Player credited fraction', 'Player rebound pool', 'Offensive rebound share',
    'Opponent rebound pool',
]


def historical_rebound_rates(logs, expected_games=None):
    """Pool all appearances, including low-minute players and free-throw misses."""
    fields = ['FGM', 'FGA', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'PTS']
    required = ['GAME_ID', 'Team', 'Opponent', *fields]
    if logs.empty or not set(required).issubset(logs) or logs[required].isna().any().any():
        raise ValueError('Incomplete historical rebound data')
    if not np.isfinite(logs[fields]).all().all() or logs[fields].lt(0).any().any():
        raise ValueError('Invalid historical rebound data')
    if (logs.FGM.gt(logs.FGA).any() or logs.FTM.gt(logs.FTA).any()
            or not np.allclose(logs.REB, logs.OREB + logs.DREB)):
        raise ValueError('Historical shot/rebound components do not match')
    boxes = logs.groupby(['GAME_ID', 'Team', 'Opponent'], as_index=False)[fields].sum()
    if not boxes.groupby('GAME_ID').size().eq(2).all() or boxes.Team.eq(boxes.Opponent).any():
        raise ValueError('Historical games must have two opposing teams')
    if expected_games is not None and not boxes.groupby('Team').size().eq(expected_games).all():
        raise ValueError('Incomplete historical season')
    reverse = boxes.rename(columns={'Team': 'Opponent', 'Opponent': 'Team',
                                     **{field: 'Opponent ' + field for field in fields}})
    paired = boxes.merge(reverse, on=['GAME_ID', 'Team', 'Opponent'], validate='one_to_one')
    if len(paired) != len(boxes):
        raise ValueError('Historical opponent pairs do not match')
    paired['FG misses'] = paired.FGA - paired.FGM
    paired['FT misses'] = paired.FTA - paired.FTM
    paired['Misses'] = paired['FG misses'] + paired['FT misses']
    paired['Player rebound pool'] = paired.OREB + paired['Opponent DREB']
    if paired['Player rebound pool'].gt(paired.Misses + 1e-9).any():
        raise ValueError('Historical player rebounds exceed missed shots')
    sums = ['FGM', 'PTS', 'FG misses', 'FT misses', 'Misses', 'Player rebound pool', 'OREB']

    def ratios(frame):
        if frame[['FGM', 'Misses', 'Player rebound pool']].le(0).any().any():
            raise ValueError('Historical rebound rates require positive denominators')
        return pd.DataFrame({
            'Points per FGM': frame.PTS / frame.FGM,
            'FG misses per FGM': frame['FG misses'] / frame.FGM,
            'FT misses per FGM': frame['FT misses'] / frame.FGM,
            'Player credited fraction': frame['Player rebound pool'] / frame.Misses,
            'Offensive rebound share': frame.OREB / frame['Player rebound pool'],
        })
    return ratios(paired.groupby('Team')[sums].sum()), ratios(paired.groupby('Opponent')[sums].sum())


def game_rebound_targets(own, allowed, opponents, point_targets):
    """Use both market team totals, even when previewing only one team's roster."""
    rows = []
    for team, opponent in opponents.items():
        if (team == opponent or opponents.get(opponent) != team
                or team not in own.index or opponent not in allowed.index):
            raise ValueError('Verified reciprocal opponents and historical rates required')
        if team not in point_targets or not np.isfinite(point_targets[team]) or point_targets[team] <= 0:
            raise ValueError('Both market team point totals are required')
        source = pd.DataFrame([own.loc[team], allowed.loc[opponent]])
        required = ['Points per FGM', 'FG misses per FGM', 'FT misses per FGM',
                    'Player credited fraction', 'Offensive rebound share']
        if (not set(required).issubset(source) or not np.isfinite(source[required]).all().all()
                or source[required].lt(0).any().any() or source['Points per FGM'].lt(2).any()
                or source[['Player credited fraction', 'Offensive rebound share']].gt(1).any().any()):
            raise ValueError('Invalid historical rebound rates')
        rates = (own.loc[team] + allowed.loc[opponent]) / 2
        fgm = point_targets[team] / rates['Points per FGM']
        fg_misses, ft_misses = fgm * rates['FG misses per FGM'], fgm * rates['FT misses per FGM']
        pool = (fg_misses + ft_misses) * rates['Player credited fraction']
        share = rates['Offensive rebound share']
        rows.append({'Team': team, 'Rebound expected FGM': fgm,
                     'Expected FG misses': fg_misses, 'Expected FT misses': ft_misses,
                     'Player credited fraction': rates['Player credited fraction'],
                     'Player rebound pool': pool, 'Offensive rebound share': share,
                     'Team OREB target': pool * share, 'Opponent DREB target': pool * (1 - share)})
    targets = pd.DataFrame(rows).set_index('Team')
    targets['Team DREB target'] = [targets.loc[opponents[t], 'Opponent DREB target'] for t in targets.index]
    targets['Opponent rebound pool'] = [targets.loc[opponents[t], 'Player rebound pool'] for t in targets.index]
    targets['Team REB target'] = targets['Team OREB target'] + targets['Team DREB target']
    return targets


def apply_team_rebounds(projections, point_targets, own, allowed, opponents):
    """Keep relative archetype effects within OREB and DREB; update REB and PRA."""
    result = projections.copy(deep=True)
    for stat in ['OREB', 'DREB', 'REB']:
        result[f'Pre-team {stat}'] = result[stat]
    result['Pre-rebound PRA'] = result.PRA
    for field in REBOUND_METADATA[4:]:
        result[field] = 1. if field.endswith('factor') else np.nan
    summaries, issues = [], []
    for team in result.Team.unique():
        try:
            opponent = opponents.get(team)
            if opponent is None or opponents.get(opponent) != team:
                raise ValueError('No verified market opponent')
            targets = game_rebound_targets(own, allowed, {team: opponent, opponent: team}, point_targets)
            team_mask = result.Team.eq(team)
            minutes = result.loc[team_mask, 'Projected minutes']
            if not np.isfinite(minutes).all() or minutes.lt(0).any():
                raise ValueError('Incomplete projected minutes')
            mask = team_mask & result['Projected minutes'].gt(0)
            active = result.loc[mask]
            if 'Opponent' in active and not active.Opponent.eq(opponent).all():
                raise ValueError('Projection opponent does not match game')
            values = active[['OREB', 'DREB', 'REB', 'PTS', 'AST']]
            if active.empty or not np.isfinite(values).all().all() or values.lt(0).any().any():
                raise ValueError('Incomplete player rebound/points/assist coverage')
            if not np.allclose(active.REB, active.OREB + active.DREB):
                raise ValueError('Player rebound components do not match')
            totals = active[['OREB', 'DREB']].sum()
            if totals.le(0).any():
                raise ValueError('No positive rebound production to allocate')
            # All validation precedes either component write, so fallback is atomic per team.
            for stat in ['OREB', 'DREB']:
                target = targets.loc[team, f'Team {stat} target']
                shares = active[stat] / totals[stat]
                result.loc[mask, stat] = shares * target
                result.loc[mask, f'Team {stat} factor'] = target / totals[stat]
                result.loc[mask, f'{stat} share'] = shares
            result.loc[mask, 'REB'] = result.loc[mask, 'OREB'] + result.loc[mask, 'DREB']
            result.loc[mask, 'PRA'] = result.loc[mask, 'PTS'] + result.loc[mask, 'REB'] + result.loc[mask, 'AST']
            for field in targets.columns:
                if field in REBOUND_METADATA:
                    result.loc[mask, field] = targets.loc[team, field]
            summaries.append({'Team': team, 'Original OREB': totals.OREB, 'Original DREB': totals.DREB,
                              'Original REB': active.REB.sum(), **targets.loc[team].to_dict()})
        except (ValueError, KeyError, TypeError) as exc:
            issues.append(f'{team}: {exc}')
    return result, pd.DataFrame(summaries), issues
