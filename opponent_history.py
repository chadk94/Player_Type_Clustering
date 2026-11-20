"""Informational head-to-head signals; never change a forecast or selection rule."""
import math

import numpy as np
import pandas as pd

MAX_GAMES = 5
MIN_GAMES = 3
MIN_MINUTES = 10
SIGNAL_FRACTION = .75
STATS = {'PTS': ('PTS',), 'REB': ('REB',), 'AST': ('AST',),
         '3PM': ('FG3M',), 'STL': ('STL',), 'BLK': ('BLK',),
         'PRA': ('PTS', 'REB', 'AST')}
UNITS = {'PTS': 'Pts', 'REB': 'Reb', 'AST': 'Ast', '3PM': '3PM',
         'STL': 'Stl', 'BLK': 'Blk', 'PRA': 'PRA'}
POLICY = ('Last 5 available meetings from this season and last season, playing for his current team; '
          'at least 10 minutes played. Either signal needs at least 3 games: green means 75% or more '
          'support today’s recommended side, with the average also supporting it; red means 75% or more '
          'oppose it, with the average also against it. Ties are pushes and count in the sample, '
          'but support neither side. Dates are MM/DD/YY; O = over, U = under, P = push. '
          'These are actual totals compared with today’s line, not the '
          'historical betting lines or a win-probability estimate. Minutes and lineups may differ. '
          'This is informational: projections, rankings and the More robust filter are unchanged.')


def player_key(value):
    """Match only verified provider IDs, including numeric IDs from pandas rows."""
    try:
        number = float(value)
        return str(int(number)) if math.isfinite(number) and number > 0 and number.is_integer() else None
    except (TypeError, ValueError):
        return None


def history_index(*frames):
    """Index public box scores once per data refresh, not once per player or game."""
    keys = ['ESPN_PLAYER_ID', 'GAME_ID', 'GAME_DATE', 'Team', 'Opponent', 'MIN']
    values = sorted({s for components in STATS.values() for s in components})
    usable = [f.reindex(columns=keys + values) for f in frames
              if f is not None and not f.empty and set(keys).issubset(f)]
    if not usable:
        return {}
    frame = pd.concat(usable, ignore_index=True).copy()
    frame['ESPN_PLAYER_ID'] = frame.ESPN_PLAYER_ID.map(player_key)
    frame['_date'] = pd.to_datetime(frame.GAME_DATE, format='mixed', errors='coerce', utc=True)
    frame['_day_only'] = frame.GAME_DATE.astype(str).str.len().eq(10)
    for field in ['MIN', *values]:
        frame[field] = pd.to_numeric(frame[field], errors='coerce')
    frame = frame.dropna(subset=['ESPN_PLAYER_ID', 'GAME_ID', '_date', 'Team', 'Opponent'])
    frame['GAME_ID'] = frame.GAME_ID.astype(str)
    # Exact duplicate imports count once. Conflicting boxes count neither way.
    frame = frame.drop_duplicates().loc[lambda f: ~f.duplicated(['ESPN_PLAYER_ID', 'GAME_ID'], keep=False)]
    frame = frame.loc[np.isfinite(frame.MIN) & frame.MIN.ge(MIN_MINUTES)]
    frame = frame.sort_values(['_date', 'GAME_ID'], ascending=False)
    return {key: group.reset_index(drop=True) for key, group in
            frame.groupby(['ESPN_PLAYER_ID', 'Team'], sort=False)}


def history_check(row, index, observed, deadline):
    result = {'History flag': 'Unavailable', 'History games': 0,
              'History detail': 'Head-to-head history could not be checked.', 'History technical': POLICY}
    side, stat = row.get('Play'), row.get('Stat')
    try:
        line = float(row.get('Book line'))
    except (TypeError, ValueError):
        return result
    if side not in ('Over', 'Under') or stat not in STATS or not math.isfinite(line) or line < 0:
        return result
    pid = player_key(observed.get('ESPN_PLAYER_ID')) if observed is not None else None
    if pid is None:
        return result
    sample = index.get((pid, row.get('Team')))
    if sample is None:
        return dict(result, **{'History flag': 'Insufficient',
                              'History detail': 'Fewer than 3 qualifying meetings are available.'})
    cutoff = pd.to_datetime(deadline, utc=True, errors='coerce')
    if pd.isna(cutoff):
        return result
    slate_day = cutoff.tz_convert('America/New_York').date()
    year = slate_day.year - (slate_day.month < 7)
    start = pd.Timestamp(f'{year - 1}-07-01', tz='UTC')
    sample = sample.loc[sample.Opponent.eq(row.get('Opponent')) & sample._date.ge(start) & sample._date.lt(cutoff)
                        & (~sample._day_only | sample._date.dt.date.lt(slate_day))].head(MAX_GAMES)
    result['History games'] = len(sample)
    if len(sample) < MIN_GAMES:
        return dict(result, **{'History flag': 'Insufficient',
                              'History detail': f'Only {len(sample)} qualifying meetings are available; at least 3 are needed.'})
    components = sample[list(STATS[stat])]
    if not np.isfinite(components.to_numpy(float)).all() or components.lt(0).any().any():
        return dict(result, **{'History detail': 'Some recent head-to-head box-score stats are missing; no warning was inferred.'})
    totals = components.sum(axis=1)
    contrary = totals.lt(line) if side == 'Over' else totals.gt(line)
    supportive = totals.gt(line) if side == 'Over' else totals.lt(line)
    average = float(totals.mean())
    against = average < line if side == 'Over' else average > line
    in_favor = average > line if side == 'Over' else average < line
    warn = contrary.mean() >= SIGNAL_FRACTION and against
    supports = supportive.mean() >= SIGNAL_FRACTION and in_favor
    units = UNITS[stat]
    outcome_side = ('Under' if side == 'Over' else 'Over') if warn else side
    count = int(contrary.sum() if warn else supportive.sum())
    signal = 'Opposes' if warn else 'Supports' if supports else 'Mixed history for'
    summary = (f'{row.get("Player", "Player")} vs {row.get("Opponent")} · {side[0]}{line:g} {units}\n'
               f'{signal} {side} · {count}/{len(sample)} {outcome_side.lower()} {line:g}')
    games = []
    for (_, game), total in zip(sample.iterrows(), totals):
        day = game._date if game._day_only else game._date.tz_convert('America/New_York')
        outcome = 'P' if total == line else 'U' if total < line else 'O'
        games.append(f'{day:%m/%d/%y} · {total:g} {units} · {game.MIN:g} min · {outcome}')
    summary += '\n\nRecent vs today’s line (newest first):\n' + '\n'.join(games)
    return dict(result, **{'History flag': 'HISTORY' if warn else 'SUPPORTS' if supports else 'No warning',
                          'History detail': summary})


def add_opponent_history(edges, index, matches, deadline):
    """Attach explanations without changing model fields or robustness eligibility."""
    out = edges.copy(deep=True)
    checks = [history_check(row, index, matches.get(int(row.PLAYER_ID)), deadline)
              for _, row in out.iterrows()]
    for column in ['History flag', 'History games', 'History detail', 'History technical']:
        out[column] = [check[column] for check in checks]
    if 'Review flags' in out:
        for status, label in [('HISTORY', 'History −'), ('SUPPORTS', 'History +')]:
            mask = out['History flag'].eq(status)
            out.loc[mask, 'Review flags'] = out.loc[mask, 'Review flags'].astype(str) + ' · ' + label
    return out
