"""How far a single player's minutes can move while a posted play still qualifies."""
import numpy as np
import pandas as pd

from sportsbook_lines import PLAY_THRESHOLD_PCT, signed_edge_pct
from team_threes import exceeds_projected_points


def _minutes_projections(team, index, minutes):
    """One scenario per row, one active teammate per column; vary only index."""
    scale = np.ones((len(minutes), len(team)))
    scale[:, index] = minutes / team.iloc[index]['Projected minutes']
    player = team.iloc[index]

    def scaled(stat):
        if stat not in team:
            return np.full(scale.shape, np.nan)
        return scale * team[stat].to_numpy(dtype=float)

    def allocate(stat, before, target_field):
        target = player.get(target_field, np.nan)
        if pd.isna(target):
            return scaled(stat)
        raw = scaled(before)
        return raw * (target / raw.sum(axis=1, keepdims=True))

    points = allocate('PTS', 'Pre-market PTS', 'Market team total')
    assists = allocate('AST', 'Pre-team AST', 'Team AST target')[:, index]
    if pd.notna(player.get('Team REB target')):
        rebounds = (allocate('OREB', 'Pre-team OREB', 'Team OREB target')[:, index]
                    + allocate('DREB', 'Pre-team DREB', 'Team DREB target')[:, index])
    else:
        rebounds = scaled('REB')[:, index]

    threes = scaled('FG3M')[:, index]
    attempt_target = player.get('Candidate team 3PA', np.nan)
    if pd.notna(attempt_target):
        raw_makes = scaled('Pre-team 3PM')
        attempts = scaled('Pre-team 3PA').sum(axis=1, keepdims=True)
        makes = raw_makes * (attempt_target / attempts)
        # The production model falls back for the whole team if any shooter
        # would score more points from threes than their points projection.
        fallback = exceeds_projected_points(makes, points).any(axis=1)
        threes = np.where(fallback, raw_makes[:, index], makes[:, index])

    return {'PTS': points[:, index], 'AST': assists, 'REB': rebounds, '3PM': threes,
            'PRA': points[:, index] + rebounds + assists,
            'STL': scaled('STL')[:, index], 'BLK': scaled('BLK')[:, index]}


def add_minutes_cushion(ranked, matchups):
    result = ranked.copy()
    result['Good-to minutes'] = np.nan
    result['Minutes cushion'] = np.nan
    plays = result[result.Play.isin(['Over', 'Under'])]
    if plays.empty:
        return result
    active = matchups[matchups['Projected minutes'].gt(0)]
    grid = np.arange(0.5, 48.5, 0.5)
    for team_name, team_plays in plays.groupby('Team', sort=False):
        team = active[active.Team.eq(team_name)].set_index('PLAYER_ID', verify_integrity=True)
        for player_id, edges in team_plays.groupby('PLAYER_ID', sort=False):
            index = team.index.get_loc(player_id)
            current = team.iloc[index]['Projected minutes']
            # Include exact imported minutes even when they fall between slider steps.
            minutes = np.union1d(grid, [current])
            anchor = minutes.searchsorted(current)
            forecasts = _minutes_projections(team, index, minutes)
            for row_id, row in edges.iterrows():
                values = forecasts[row.Stat]
                if not np.isclose(values[anchor], row.Projected):
                    continue
                over = row.Play == 'Over'
                steps = np.arange(anchor, -1, -1) if over else np.arange(anchor, len(minutes))
                gaps = signed_edge_pct(values[steps], row['Book implied total'])
                passes = gaps > PLAY_THRESHOLD_PCT if over else gaps < -PLAY_THRESHOLD_PCT
                # Stop at the first failure, even if the 3PM fallback creates
                # another qualifying interval farther from today's projection.
                stop = np.flatnonzero(~passes)
                count = stop[0] if stop.size else len(steps)
                if count:
                    limit = minutes[steps[count - 1]]
                    result.loc[row_id, ['Good-to minutes', 'Minutes cushion']] = limit, abs(limit - current)
    return result
