"""Rank the existing per-game player projections without recalculating them."""
import pandas as pd

from team_projection_view import TEAM_GAME_RETENTION

PLAYER_SORTS = {
    'Points (PTS)': 'PTS', 'Rebounds (REB)': 'REB', 'Assists (AST)': 'AST',
    'Threes made (3PM)': 'FG3M', 'Steals (STL)': 'STL', 'Blocks (BLK)': 'BLK',
    'Turnovers (TOV)': 'TOV', 'Points + rebounds + assists (PRA)': 'PRA',
    'Offensive rebounds (OREB)': 'OREB', 'Defensive rebounds (DREB)': 'DREB',
    'Personal fouls (PF)': 'PF', 'Projected minutes': 'Projected minutes',
}


def rank_slate_players(frame, stat, include_zero=False, now=None):
    """Sort full-precision values; keep unknowns last and never invent zeroes."""
    if stat not in PLAYER_SORTS.values():
        raise ValueError('Unknown projection category')
    rows = frame.copy()
    if rows.empty:
        return rows
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    starts = pd.to_datetime(rows['GameTimeUTC'], utc=True)
    rows = rows.loc[starts + TEAM_GAME_RETENTION > now]
    if not include_zero:
        rows = rows.loc[rows['Projected minutes'].gt(0) | rows['Projected minutes'].isna()]
    rows[stat] = pd.to_numeric(rows.get(stat, pd.Series(float('nan'), index=rows.index)), errors='coerce')
    rows = rows.sort_values([stat, 'Player', 'Team', 'Game key'],
                            ascending=[False, True, True, True], na_position='last', kind='stable')
    rows.insert(0, 'Rank', range(1, len(rows) + 1))
    return rows.reset_index(drop=True)
