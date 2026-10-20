"""Team summaries of the existing player model; no separate projection model."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from import_minutes import TEAM_IDS

TEAM_STATS = {'PTS': 'PTS', 'REB': 'REB', 'AST': 'AST', '3PM': 'FG3M',
              'STL': 'STL', 'BLK': 'BLK', 'TOV': 'TOV', 'PRA': 'PRA'}
TARGET_COLUMNS = {'PTS': 'Market team total', 'AST': 'Team AST target',
                  '3PM': 'Team 3PM target', 'REB': 'Team REB target'}


def upcoming_team_games(schedule, now=None):
    """Show scheduled games even when no sportsbook event/lines are available."""
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    abbreviations = {value: key for key, value in TEAM_IDS.items()}
    games = {}
    for _, row in schedule.iterrows():
        try:
            start = pd.Timestamp(row.GameTimeUTC)
            home, away = abbreviations[row.TEAM_ID], row.Opponent
            if (row.Venue != 'home' or away not in TEAM_IDS or home == away
                    or start.tzinfo is None or start <= now
                    or any(word in str(row.GameStatus).lower()
                           for word in ['postpon', 'cancel', 'suspend', 'delay', 'final'])):
                continue
            key = f'{home}_{away}_{start.isoformat()}'
            games[key] = {'key': key, 'Home': home, 'Away': away, 'GameTimeUTC': start}
        except (KeyError, ValueError, TypeError):
            continue
    return sorted(games.values(), key=lambda game: (game['GameTimeUTC'], game['Away']))


def _total(frame, column):
    """Never turn an incomplete rotation into an apparently complete team total."""
    minutes = frame['Projected minutes']
    if frame.empty or not np.isfinite(minutes).all() or minutes.lt(0).any():
        return np.nan
    active = frame[minutes.gt(0)]
    if active.empty or column not in active:
        return np.nan
    values = active[column]
    return values.sum() if np.isfinite(values).all() and values.ge(0).all() else np.nan


def _target(frame, column):
    if column not in frame:
        return np.nan
    values = frame.loc[frame['Projected minutes'].gt(0), column]
    if values.empty or not np.isfinite(values).all() or values.nunique() != 1:
        return np.nan
    return values.iloc[0]


def team_projection_tables(baselines, matchups, adjusted, team_order):
    """Sum full rosters before rounding; separate archetype and team adjustments."""
    totals, breakdown, issues = [], [], []
    for team in team_order:
        base = baselines[baselines.Team.eq(team)]
        before = matchups[matchups.Team.eq(team)]
        final = adjusted[adjusted.Team.eq(team)]
        if (set(base.PLAYER_ID) != set(before.PLAYER_ID)
                or set(base.PLAYER_ID) != set(final.PLAYER_ID)
                or any(frame.PLAYER_ID.duplicated().any() for frame in [base, before, final])):
            raise ValueError('Team summary requires the same complete roster at every stage')
        row = {'Team': team, 'Minutes': _total(final, 'Projected minutes')}
        row.update({stat: _total(final, column) for stat, column in TEAM_STATS.items()})
        missing = [stat for stat in TEAM_STATS if pd.isna(row[stat])]
        if pd.isna(row['Minutes']):
            issues.append(f'{team}: projected minutes are missing or invalid; team totals stay blank.')
        elif missing:
            issues.append(f'{team}: incomplete player coverage for {", ".join(missing)}; those totals stay blank.')
        totals.append(row)
        for stat, target_column in TARGET_COLUMNS.items():
            column = TEAM_STATS[stat]
            original, projected = _total(before, column), row[stat]
            change = (projected / original - 1) * 100 if original > 0 else np.nan
            breakdown.append({'Team': team, 'Stat': stat, 'Baseline': _total(base, column),
                              'After archetypes': original, 'Team target': _target(final, target_column),
                              'Final projection': projected, 'Team adjustment %': change})
    return pd.DataFrame(totals), pd.DataFrame(breakdown), issues
