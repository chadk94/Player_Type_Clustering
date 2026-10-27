"""Validated season refresh with explicit, provenance-labelled historical prior mode."""
import hashlib
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from archetype_model import apply_reviews, assign_named_roles, validate_named_roles

ROOT = Path(__file__).resolve().parent
SEASON = os.environ.get('NBA_SEASON', '2026-27')
SEASON_ID = 20000 + int(SEASON[:4])
OPENING_GAMES = int(os.environ.get('NBA_OPENING_GAMES', '3'))
if OPENING_GAMES < 0:
    raise ValueError('NBA_OPENING_GAMES must be nonnegative')


def load_rules():
    return json.loads((ROOT / 'archetype_overrides.json').read_text())


def read_season_clusters(path):
    frame = pd.read_csv(path)
    if frame.empty or not frame.SEASON_ID.eq(SEASON_ID).all():
        raise ValueError(f'{path} is not a complete {SEASON} export; run season_pipeline.py')
    side = 'def' if Path(path).name.startswith('def_') else 'off'
    validate_named_roles(frame, side, load_rules())
    return frame


def apply_overrides(frame, side, rules=None):
    """Manual assignment wins after automatic semantic label alignment."""
    rules = rules or load_rules()
    frame = apply_reviews(frame, side, rules)
    overrides = rules.get(f'{side.upper()}_CLUSTER_OVERRIDES', {})
    mask = frame.PLAYER_NAME.isin(overrides)
    frame.loc[mask, 'Cluster'] = frame.loc[mask, 'PLAYER_NAME'].map(overrides)
    names = rules[f'{side.upper()}_CLUSTER_NAMES']
    frame['Archetype'] = frame.Cluster.map(lambda x: names.get(str(int(x)), f'Cluster {int(x)} (unnamed)') if pd.notna(x) else None)
    frame['AssignmentSource'] = np.where(mask, 'manual_override', frame.get('AssignmentSource', 'auto_model'))
    return frame


def choose_minutes(actual, projected, games, window=OPENING_GAMES):
    if window < 0 or int(window) != window or not np.isfinite(games) or games < 0 or int(games) != games:
        raise ValueError('Game counts must be nonnegative integers')
    value = projected if games < window else actual
    if not np.isfinite(value) or not 0 <= value <= 48:
        raise ValueError('Selected minutes must be finite and between 0 and 48')
    return float(value)


def projected_minutes(games, fallback):
    """Dashboard uses full-season team game count, independent of selected date window."""
    if games.empty:
        return fallback
    path = Path(os.environ.get('NBA_MINUTES_FILE', 'projected_minutes.csv'))
    counts_path = Path(os.environ.get('NBA_TEAM_GAMES_FILE', 'team_games.csv'))
    if not path.exists() or not counts_path.exists():
        raise ValueError('Provide projected_minutes.csv and team_games.csv (see README)')
    projections, counts = validate_minutes(path, counts_path)
    player_id = int(games.iloc[0].PLAYER_ID)
    row = projections[projections.PLAYER_ID == player_id]
    if row.empty:
        raise ValueError(f'Missing projected minutes for PLAYER_ID={player_id}')
    row = row.iloc[0]
    return choose_minutes(fallback, row.PROJECTED_MIN, counts.loc[row.TEAM_ID], OPENING_GAMES)


def validate_minutes(minutes_path, games_path):
    minutes = pd.read_csv(minutes_path)
    counts = pd.read_csv(games_path)
    for frame in (minutes, counts):
        if frame.empty or not frame.SEASON_ID.eq(SEASON_ID).all():
            raise ValueError(f'Minutes and team games must be nonempty and scoped to {SEASON}')
    if minutes.PLAYER_ID.duplicated().any() or counts.TEAM_ID.duplicated().any():
        raise ValueError('Duplicate player or team keys')
    if not minutes.PROJECTED_MIN.between(0, 48).all():
        raise ValueError('PROJECTED_MIN must be between 0 and 48')
    if not (counts.GAMES_PLAYED.ge(0) & counts.GAMES_PLAYED.mod(1).eq(0)).all():
        raise ValueError('GAMES_PLAYED must be nonnegative integers')
    counts = counts.set_index('TEAM_ID').GAMES_PLAYED
    if not minutes.TEAM_ID.isin(counts.index).all():
        raise ValueError('Missing team game counts')
    return minutes, counts


def run(input_path, minutes_path, games_path, output, window=OPENING_GAMES, prior_season_id=None):
    data = pd.read_csv(input_path)
    expected_season = SEASON_ID if prior_season_id is None else prior_season_id
    if prior_season_id is not None and prior_season_id != SEASON_ID - 1:
        raise ValueError('Prior mode requires the immediately previous season')
    if data.empty or not data.SEASON_ID.eq(expected_season).all():
        raise ValueError(f'Feature input must be scoped to SEASON_ID={expected_season}')
    data['SourceSeasonID'] = data.SEASON_ID
    data['FeatureDataStatus'] = 'current_season' if prior_season_id is None else 'preseason_historical_prior'
    data['SEASON_ID'] = SEASON_ID
    if data.PLAYER_ID.duplicated().any():
        raise ValueError('Expected one current feature row per PLAYER_ID')
    minutes, counts = validate_minutes(minutes_path, games_path)
    if prior_season_id is not None and not counts.eq(0).all():
        raise ValueError('Preseason prior mode requires zero completed current-season team games')
    if prior_season_id is not None and window == 0:
        raise ValueError('Preseason prior mode requires an opening window for user minutes')
    if 'TEAM_ID' in data:
        supplied = minutes.set_index('PLAYER_ID').TEAM_ID.reindex(data.PLAYER_ID)
        if not np.array_equal(supplied.to_numpy(), data.TEAM_ID.to_numpy()):
            raise ValueError('Current team IDs disagree with user minutes')
        data = data.drop(columns='TEAM_ID')
    data = data.merge(minutes[['PLAYER_ID', 'TEAM_ID', 'PROJECTED_MIN']],
                      on='PLAYER_ID', validate='one_to_one', how='left')
    # Do not change MIN: observed counts and MIN must stay paired for per-36 rates.
    if len(data) != len(pd.read_csv(input_path)) or data.PROJECTED_MIN.isna().any():
        raise ValueError('Missing user minutes for feature players')
    data['EffectiveMinutes'] = [choose_minutes(r.MIN, r.PROJECTED_MIN, counts.loc[r.TEAM_ID], window) for r in data.itertuples()]
    data = data[data.EffectiveMinutes > 10].copy()
    data['Height_IN'] = data.HEIGHT.str.split('-').map(lambda h: int(h[0]) * 12 + int(h[1]))
    for col in ['PTS', 'AST', 'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'paint_shots_per_game', 'corner_3_per_game', 'midrange_per_game', 'above_break_3_per_game']:
        data[col + '_per36'] = data[col] * 36 / data.MIN
    feature_sets = json.loads((ROOT / 'clustering_features.json').read_text())
    exports, audit = {}, {'season': SEASON, 'opening_games': window, 'mode': 'current_season' if prior_season_id is None else 'preseason_historical_prior', 'source_season_id': expected_season, 'inputs': [str(input_path), str(minutes_path), str(games_path)], 'label_maps': {}, 'sha256': {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [input_path, minutes_path, games_path, ROOT / 'archetype_overrides.json', ROOT / 'clustering_features.json']}}
    for side, filename in [('off', 'player_clusters_detailed.csv'), ('def', 'def_player_clusters_detailed.csv')]:
        features = feature_sets[f'cluster_players_{side}']
        if data.empty or not np.isfinite(data[features].to_numpy(dtype=float)).all():
            raise ValueError(f'Missing or nonfinite {side} features; no silent zero imputation')
        reference = pd.read_csv(ROOT / 'references' / filename)
        result, model_info = assign_named_roles(data, reference, features, side,
                                                load_rules(), apply_overrides,
                                                prior_mode=prior_season_id is not None)
        audit.setdefault('archetype_models', {})[side] = model_info
        exports[filename] = result
    audit['archetype_rotation_reviews_sha256'] = hashlib.sha256((ROOT / 'archetype_rotation_reviews.csv').read_bytes()).hexdigest()
    audit['archetype_reference_reviews_sha256'] = hashlib.sha256((ROOT / 'archetype_reference_reviews.csv').read_bytes()).hexdigest()
    audit['archetype_reviews_sha256'] = hashlib.sha256((ROOT / 'archetype_reviews.csv').read_bytes()).hexdigest()
    audit['archetype_league_reviews_sha256'] = hashlib.sha256((ROOT / 'archetype_league_reviews.csv').read_bytes()).hexdigest()
    audit['classification_policy'] = 'Named references only; no new KMeans fit or forced label permutation'
    # Validate both models before writing either export.
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    for filename, frame in exports.items():
        frame.to_csv(destination / filename, index=False)
    (destination / 'season_refresh.json').write_text(json.dumps(audit, indent=2) + '\n')
    return exports



def apply_roster_rules(roster):
    """Enforce reviewed roles and user overrides at the live-app boundary too."""
    result=roster.copy()
    rules=load_rules()
    for kind in ['Off','Def']:
        frame=result[['PLAYER_ID','PLAYER_NAME',f'{kind}Cluster',f'{kind}AssignmentSource']].rename(
            columns={f'{kind}Cluster':'Cluster',f'{kind}AssignmentSource':'AssignmentSource'})
        final=apply_overrides(frame,kind.lower(),rules)
        validate_named_roles(final,kind.lower(),rules)
        result[f'{kind}Cluster']=final.Cluster
        result[f'{kind}Archetype']=final.Archetype
        result[f'{kind}AssignmentSource']=final.AssignmentSource
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--minutes', required=True)
    parser.add_argument('--team-games', required=True)
    parser.add_argument('--opening-games', type=int, default=OPENING_GAMES)
    parser.add_argument('--output', default='.')
    parser.add_argument('--prior-season-id', type=int, help='Explicit preseason-only prior, e.g. 22025; preserves source season metadata')
    args = parser.parse_args()
    run(args.input, args.minutes, args.team_games, args.output, args.opening_games, args.prior_season_id)
