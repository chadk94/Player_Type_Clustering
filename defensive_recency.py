"""Comparison-only defensive recency. Never replaces the live projection frame."""
from hashlib import sha256
import json

import numpy as np
import pandas as pd

from team_rate_blend import DEFENSIVE_RATE_COLUMNS, season_rates
from team_assists import apply_team_assists
from team_threes import apply_team_threes
from team_rebounds import apply_team_rebounds

VERSION = 'defensive-team-half10-after8-v1'
STATS = {'AST': 'AST', '3PA': 'FG3A', '3PM': 'FG3M', 'OREB': 'OREB',
         'DREB': 'DREB', 'REB': 'REB', 'PRA': 'PRA'}
TARGETS = ['Market team total', 'Team AST target', 'Team 3PA target',
           'Team OREB target', 'Team DREB target']


def comparison_rates(prior, current, logs):
    """Only three allowed ratios change, with the existing year weights preserved."""
    result = {key: value.copy(deep=True) for key, value in current.items()}
    coverage = result['coverage']
    coverage['Recency status'] = 'Waiting for eight verified games'
    if logs.empty:
        return result
    try:
        season_rates(logs)  # Reuse the same complete-box validation as production.
        if logs.duplicated(['GAME_ID', 'ESPN_PLAYER_ID']).any():
            raise ValueError('Duplicate appearances')
        data = logs.copy()
        data['_date'] = pd.to_datetime(data.GAME_DATE, utc=True, errors='raise')
        if data._date.isna().any() or data.groupby('GAME_ID')._date.nunique().gt(1).any():
            raise ValueError('Unverified game dates')
        fields = ['AST', 'FGM', 'FG3A', 'FGA', 'OREB', 'DREB']
        boxes = data.groupby(['GAME_ID', '_date', 'Team', 'Opponent'], as_index=False)[fields].sum()
        reverse = boxes[['GAME_ID', 'Team', 'DREB']].rename(columns={'Team': 'Opponent', 'DREB': '_def_dreb'})
        boxes = boxes.merge(reverse, on=['GAME_ID', 'Opponent'], validate='many_to_one')
        boxes['_pool'] = boxes.OREB + boxes._def_dreb
    except (ValueError, KeyError, TypeError):
        coverage['Recency status'] = 'Unavailable — game history could not be verified'
        return result
    ratios = {'assists': ('AST', 'FGM'), 'threes': ('FG3A', 'FGA'),
              'rebounds_allowed': ('OREB', '_pool')}
    for team, row in coverage.iterrows():
        games = boxes[boxes.Opponent.eq(team)].sort_values(['_date', 'GAME_ID'])
        n = len(games)
        if not (n == row['Verified games'] == row['Completed games'] and row['Defensive current weight'] > 0):
            coverage.loc[team, 'Recency status'] = 'Unavailable — incomplete current season'
            continue
        if n < 8:
            continue
        # Games, never player appearances; pool weighted opportunities before dividing.
        weights = np.power(.5, np.arange(n - 1, -1, -1) / 10.)
        weight = row['Defensive current weight']
        for kind, (numerator, denominator) in ratios.items():
            rate = np.dot(weights, games[numerator]) / np.dot(weights, games[denominator])
            col = DEFENSIVE_RATE_COLUMNS[kind][0]
            result[kind].loc[team, col] = rate if weight == 1 else (
                (1 - weight) * prior[kind].loc[team, col] + weight * rate)
        coverage.loc[team, 'Recency status'] = 'Ready'
    return result


def compare_game(final, rates, opponents):
    """Reallocate identical existing shares, without touching the input frame."""
    if set(final.Team) != set(opponents) or any(opponents.get(v) != k for k, v in opponents.items()):
        raise ValueError('Both teams are required for the comparison')
    targets = {}
    for team, group in final.groupby('Team'):
        minutes = group['Projected minutes']
        active = group[minutes.gt(0)]
        if (not np.isfinite(minutes).all() or minutes.lt(0).any() or
                not np.isclose(minutes.sum(), 240.) or active.empty):
            raise ValueError(f'{team}: a complete 240-minute rotation is needed')
        if rates['coverage'].loc[team, 'Recency status'] != 'Ready':
            raise ValueError(f"{team}: {rates['coverage'].loc[team, 'Recency status']}")
        for col in TARGETS:
            if col not in active or not np.isfinite(active[col]).all() or active[col].nunique() != 1:
                raise ValueError(f'{team}: a verified {col} is needed')
        targets[team] = float(active['Market team total'].iloc[0])
    trial, _, errors = apply_team_assists(final, targets, rates['assists'], opponents)
    trial, _, three_errors = apply_team_threes(trial, targets, rates['threes'], opponents)
    trial, _, reb_errors = apply_team_rebounds(trial, targets, rates['rebounds_own'], rates['rebounds_allowed'], opponents)
    if errors or three_errors or reb_errors:
        raise ValueError('; '.join(errors + three_errors + reb_errors))
    rows, player_rows = [], []
    for team in sorted(opponents):
        mask = final.Team.eq(team) & final['Projected minutes'].gt(0)
        for label, stat in STATS.items():
            old, new = final.loc[mask, stat].sum(), trial.loc[mask, stat].sum()
            rows.append({'Team': team, 'Stat': label, 'Current': float(old),
                         'Defensive recency': float(new), 'Change': float(new - old)})
    for index, player in final[final['Projected minutes'].gt(0)].iterrows():
        for label in ['AST', '3PM', 'REB', 'PRA']:
            stat = STATS[label]
            player_rows.append({'Player': player.get('Player', str(player.PLAYER_ID)), 'Team': player.Team,
                'Stat': label, 'Current': player[stat], 'Defensive recency': trial.loc[index, stat],
                'Change': trial.loc[index, stat] - player[stat]})
    return pd.DataFrame(rows), pd.DataFrame(player_rows)


def record_id(record):
    return sha256(json.dumps({k: v for k, v in record.items() if k not in ['id', 'captured_at']},
                             sort_keys=True, allow_nan=False).encode()).hexdigest()


def snapshot(table, game, book, logs, final, notes=None, now=None):
    now = pd.Timestamp(now or pd.Timestamp.now(tz='UTC'))
    start = pd.Timestamp(game['GameTimeUTC'])
    if now.tzinfo is None or start.tzinfo is None or now >= start or logs.empty:
        return None
    through = pd.to_datetime(logs.GAME_DATE, utc=True).max()
    if pd.isna(through) or through >= now:
        return None
    # Each pair freezes identical inputs and totals; no changing estimates after tip.
    r = dict(version=VERSION, start=start.isoformat(), captured_at=now.isoformat(),
             history_through=through.isoformat(), home=game['Home'], away=game['Away'], book=book,
             notes=notes or {}, totals=table.drop(columns='Change').to_dict('records'),
             minutes=final[['PLAYER_ID', 'Team', 'Projected minutes']].to_dict('records'),
             saved_lines=bool(final.get('Saved game lines', pd.Series(False, index=final.index)).any()))
    r['id'] = record_id(r)
    return r if valid_record(r) else None


def valid_record(r):
    try:
        if (r['version'] != VERSION or r['id'] != record_id(r) or len(json.dumps(r)) > 50000 or
                r['home'] == r['away'] or not pd.Timestamp(r['history_through']) < pd.Timestamp(r['captured_at']) < pd.Timestamp(r['start'])):
            return False
        if any(pd.Timestamp(r[c]).tzinfo is None for c in ['start', 'captured_at', 'history_through']):
            return False
        totals = pd.DataFrame(r['totals'])
        expected = {(t, s) for t in [r['home'], r['away']] for s in STATS}
        if (len(totals) != len(expected) or set(zip(totals.Team, totals.Stat)) != expected or
                not np.isfinite(totals[['Current', 'Defensive recency']]).all().all() or
                totals[['Current', 'Defensive recency']].lt(0).any().any()):
            return False
        for team, note in r['notes'].items():
            if (team not in [r['home'], r['away']] or not note['change'].strip() or len(note['change']) > 500 or
                    not note['source'].startswith(('https://', 'http://')) or len(note['source']) > 1000):
                return False
        return True
    except (ValueError, TypeError, KeyError, OverflowError, AttributeError):
        return False


def evaluate(records, logs):
    """Latest frozen pregame snapshot per game/book. Never fill absent results with zero."""
    chosen, notes = {}, {}
    for r in sorted((r for r in records if valid_record(r)), key=lambda r: r['captured_at']):
        key = (r['home'], r['away'], r['start'], r['book'])
        notes.setdefault(key, {}).update(r['notes'])
        if key not in chosen or r['captured_at'] > chosen[key]['captured_at']:
            chosen[key] = r
    rows = []
    if logs.empty:
        boxes = pd.DataFrame()
    else:
        boxes = logs.copy()
        boxes['_date'] = pd.to_datetime(boxes.GAME_DATE, utc=True)
    for r in chosen.values():
        start = pd.Timestamp(r['start'])
        pool = boxes if boxes.empty else boxes[boxes.Team.isin([r['home'], r['away']]) &
            boxes.Opponent.isin([r['home'], r['away']]) &
            (boxes['_date'] - start).abs().le(pd.Timedelta(hours=12))]
        actual = {}
        if not pool.empty and pool.GAME_ID.nunique() == 1:
            try:
                season_rates(pool)
                if pool.duplicated(['GAME_ID', 'ESPN_PLAYER_ID']).any():
                    raise ValueError('Duplicate result')
                if set(pool.Team) != {r['home'], r['away']} or pool.groupby('Team').MIN.sum().lt(235).any():
                    raise ValueError('Incomplete result')
                actual = pool.groupby('Team')[list(set(STATS.values()) - {'PRA'})].sum().to_dict('index')
                for team, values in actual.items():
                    pts = pool.loc[pool.Team.eq(team), 'PTS'].sum()
                    values['PRA'] = pts + values['REB'] + values['AST']
            except (ValueError, KeyError, TypeError):
                actual = {}
        for total in r['totals']:
            team, stat = total['Team'], total['Stat']
            opponent = r['away'] if team == r['home'] else r['home']
            note = notes[(r['home'], r['away'], r['start'], r['book'])].get(opponent, {})
            rows.append({**total, 'Opponent': opponent, 'Game time': r['start'], 'Book': r['book'],
                'Captured at': r['captured_at'], 'Actual': actual.get(team, {}).get(STATS[stat], np.nan),
                'Documented opponent change': note.get('change', ''), 'Source': note.get('source', ''),
                'Saved game lines': r['saved_lines']})
    return pd.DataFrame(rows)


def error_summary(history, documented_only=False):
    if history.empty:
        return pd.DataFrame()
    frame = history[history.Actual.notna()]
    if documented_only:
        frame = frame[frame['Documented opponent change'].ne('')]
    rows = []
    for stat, group in frame.groupby('Stat'):
        a = group.Current - group.Actual
        b = group['Defensive recency'] - group.Actual
        rows.append({'Stat': stat, 'Team-games': len(group), 'Current RMSE': np.sqrt((a*a).mean()),
                     'Recency RMSE': np.sqrt((b*b).mean()), 'Current MAE': a.abs().mean(),
                     'Recency MAE': b.abs().mean()})
    return pd.DataFrame(rows)
