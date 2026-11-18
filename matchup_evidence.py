"""Display-only peer game history; never used to fit matchup effects."""
import numpy as np
import pandas as pd

from live_projections import name_key

EVIDENCE_STATS = ('PTS', 'AST', 'OREB', 'DREB', 'STL', 'BLK', 'FG3A', 'FG3M', 'FGA')


def recent_peer_games(logs):
    """Index actual appearances and strictly preceding player rates, once per refresh.

    Keep this separate from matchup_effects: these rows explain recent history,
    not the blended/shrunk rating. Missing boxes are never filled with zero.
    """
    required = {'ESPN_PLAYER_ID', 'PLAYER_NAME', 'GAME_DATE', 'GAME_ID', 'MIN', 'Opponent'}
    if logs.empty or not required.issubset(logs):
        return {}
    frame = logs.copy()
    frame['_date'] = pd.to_datetime(frame.GAME_DATE, format='mixed', errors='coerce', utc=True)
    frame['MIN'] = pd.to_numeric(frame.MIN, errors='coerce')
    frame = frame.loc[frame.MIN.ge(5) & np.isfinite(frame.MIN) & frame._date.notna()
                      & frame.ESPN_PLAYER_ID.notna() & frame.PLAYER_NAME.notna()].copy()
    frame = frame.drop_duplicates(['ESPN_PLAYER_ID', 'GAME_ID'], keep=False)
    frame = frame.sort_values(['_date', 'GAME_ID', 'ESPN_PLAYER_ID']).reset_index(drop=True)
    for stat in EVIDENCE_STATS:
        values = pd.to_numeric(frame.reindex(columns=[stat])[stat], errors='coerce')
        frame[stat] = values.where(np.isfinite(values) & values.ge(0))
        # Only earlier appearances with this statistic enter its usual rate.
        available = frame[stat].notna()
        groups = frame.ESPN_PLAYER_ID
        totals = frame[stat].fillna(0).groupby(groups).cumsum() - frame[stat].fillna(0)
        minutes = frame.MIN.where(available, 0)
        prior_minutes = minutes.groupby(groups).cumsum() - minutes
        prior_count = available.astype(int).groupby(groups).cumsum() - available.astype(int)
        frame['usual_' + stat] = (totals / prior_minutes * frame.MIN).where(prior_count.ge(3))
    rows = []
    for _, row in frame.iterrows():
        raw_date = str(row.GAME_DATE)
        day = row._date if len(raw_date) == 10 else row._date.tz_convert('America/New_York')
        rows.append(dict(player=str(row.PLAYER_NAME), key=name_key(row.PLAYER_NAME),
                         date=day.strftime('%b %d'), timestamp=row._date.isoformat(),
                         game=str(row.GAME_ID), minutes=float(row.MIN),
                         stats={s: float(row[s]) if pd.notna(row[s]) else None for s in EVIDENCE_STATS},
                         usual={s: float(row['usual_' + s]) if pd.notna(row['usual_' + s]) else None
                                for s in EVIDENCE_STATS}))
    result = {}
    for side in ('Off', 'Def'):
        column = side + 'Cluster'
        if column not in frame:
            continue
        for (cluster, opponent), group in frame.dropna(subset=[column, 'Opponent']).groupby([column, 'Opponent']):
            result[(side, int(cluster), opponent)] = [rows[i] for i in reversed(group.index.tolist())]
    return result


def peer_panels(stat, player, opponent, effects, roles, before=None):
    """Select ten most recent *other-player* appearances for each relevant role."""
    panels = []
    own_names = {name_key(str(player.get(k, ''))) for k in ('PLAYER_NAME', 'WorkbookPlayerName')}
    cutoff = pd.to_datetime(before, errors='coerce', utc=True)
    for side, components in roles.items():
        cluster = player.get(side + 'Cluster')
        entry = effects.get((side, int(cluster), opponent), {}) if pd.notna(cluster) else {}
        # Attempts are primary for threes; makes are descriptive only.
        stats = ('FG3A', 'FG3M') if stat == '3PM' else ('PTS', 'FGA') if stat == 'PTS' else components
        labels = [{'FG3A': '3PA', 'FG3M': '3PM'}.get(s, s) for s in stats]
        primary = stats[0]
        rows = []
        for row in entry.get('performances', []):
            if row['key'] in own_names:
                continue
            if pd.notna(cutoff) and pd.Timestamp(row['timestamp']) >= cutoff:
                continue
            if not any(row['stats'].get(s) is not None for s in components):
                continue
            def number(value, decimals=0):
                return '—' if value is None else f'{value:.{decimals}f}'
            rows.append(dict(player=row['player'], date=row['date'],
                             values=[number(row['minutes'], 1),
                                     *[number(row['stats'].get(s)) for s in stats],
                                     number(row['usual'].get(primary), 1)]))
            if len(rows) == 10:
                break
        name = player.get(side + 'Archetype')
        panels.append(dict(title=f"{'Offense' if side == 'Off' else 'Defense'} · {name if pd.notna(name) else 'Unassigned'}",
                           columns=['Player / date', 'MIN', *labels, 'Usual ' + labels[0]], rows=rows))
    return dict(opponent=opponent, panels=panels,
                note='Actual game totals · 5+ minutes played · newest first this season. Usual = that player’s earlier season rate at the minutes shown (at least 3 earlier appearances). — = unavailable. These examples do not change the blended matchup rating.')
