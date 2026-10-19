"""Build provisional target-season archetypes from archived prior-season features."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from season_pipeline import ROOT, SEASON_ID, OPENING_GAMES, apply_overrides, load_rules, run


def bootstrap(minutes_path, output, window=OPENING_GAMES):
    output = Path(output)
    minutes = pd.read_csv(minutes_path)
    required = ['PLAYER_ID', 'TEAM_ID', 'SEASON_ID', 'PROJECTED_MIN', 'PLAYER_NAME', 'TEAM_ABBREVIATION']
    if not set(required).issubset(minutes.columns):
        raise ValueError('Use import_minutes.py to prepare the workbook first')
    counts = minutes[['TEAM_ID', 'SEASON_ID']].drop_duplicates().assign(GAMES_PLAYED=0)
    # User states no current-season games have been played. No historical game counts copied.
    output.mkdir(parents=True, exist_ok=True)
    counts.to_csv(output / 'team_games.csv', index=False)
    minutes.to_csv(output / 'projected_minutes.csv', index=False)
    history_path = ROOT / 'references/player_clusters_detailed.csv'
    history = pd.read_csv(history_path)
    if not history.SEASON_ID.eq(SEASON_ID - 1).all() or history.PLAYER_ID.duplicated().any():
        raise ValueError('Expected unique previous-season reference profiles')
    for side in ['player_clusters_detailed.csv', 'def_player_clusters_detailed.csv']:
        reference = pd.read_csv(ROOT / 'references' / side)
        features = json.loads((ROOT / 'clustering_features.json').read_text())[
            'cluster_players_off' if side == 'player_clusters_detailed.csv' else 'cluster_players_def']
        np.testing.assert_allclose(history[features].to_numpy(), reference[features].to_numpy())
    inputs = history[history.PLAYER_ID.isin(minutes.PLAYER_ID)].drop(columns=['Cluster'])
    inputs.to_csv(output / 'historical_player_data.csv', index=False)
    exports = run(output / 'historical_player_data.csv', output / 'projected_minutes.csv',
                  output / 'team_games.csv', output, window, prior_season_id=SEASON_ID - 1)
    historical = history.drop(columns='Cluster').rename(
        columns={'PLAYER_NAME': 'HistoricalPlayerName', 'SEASON_ID': 'SourceSeasonID'})
    base = minutes.merge(historical, on='PLAYER_ID', how='left', validate='one_to_one')
    base['WorkbookPlayerName'] = base.PLAYER_NAME
    base['PLAYER_NAME'] = base.HistoricalPlayerName.fillna(base.WorkbookPlayerName)
    base['HasHistoricalProfile'] = base.SourceSeasonID.notna()
    base['ModelEligibleMinutes'] = base.PROJECTED_MIN.gt(10)
    base['EffectiveMinutes'] = base.PROJECTED_MIN
    base['FeatureDataStatus'] = np.where(base.HasHistoricalProfile, 'preseason_historical_prior', 'missing_historical_profile')
    report_columns = ['PLAYER_ID', 'PLAYER_NAME', 'WorkbookPlayerName', 'TEAM_ABBREVIATION',
                      'TEAM_ID', 'SEASON_ID', 'PROJECTED_MIN', 'SourceSeasonID',
                      'HasHistoricalProfile', 'ModelEligibleMinutes', 'FeatureDataStatus']
    report = base[report_columns].copy()
    rules = load_rules()
    for side, filename in [('off', 'player_clusters_detailed.csv'), ('def', 'def_player_clusters_detailed.csv')]:
        model = exports[filename].set_index('PLAYER_ID')
        result = base.copy()
        result['Cluster'] = result.PLAYER_ID.map(model.Cluster)
        result['AssignmentSource'] = np.select(
            [result.Cluster.notna(), ~result.ModelEligibleMinutes],
            ['kmeans_aligned', 'outside_minutes_cutoff'], default='missing_historical_profile')
        result = apply_overrides(result, side, rules)
        result['ArchetypeStatus'] = np.where(result.Cluster.notna(), 'assigned_provisional', result.AssignmentSource)
        result.to_csv(output / filename, index=False)
        report[f'{side.title()}Cluster'] = result.Cluster
        report[f'{side.title()}Archetype'] = result.Archetype
        report[f'{side.title()}AssignmentSource'] = result.AssignmentSource
    report.to_csv(output / 'preseason_archetypes.csv', index=False)
    missing = report[~report.HasHistoricalProfile].copy()
    missing.to_csv(output / 'missing_historical_profiles.csv', index=False)
    missing[missing.ModelEligibleMinutes].to_csv(output / 'rotation_players_needing_profiles.csv', index=False)
    totals = minutes.groupby(['TEAM_ABBREVIATION', 'TEAM_ID']).agg(
        Players=('PLAYER_ID', 'size'), RotationPlayers=('PROJECTED_MIN', lambda x: int(x.gt(0).sum())),
        ProjectedMinutes=('PROJECTED_MIN', 'sum')).reset_index()
    totals.to_csv(output / 'team_minutes_check.csv', index=False)
    override_audit = []
    for side in ['off', 'def']:
        for name, cluster in rules.get(f'{side.upper()}_CLUSTER_OVERRIDES', {}).items():
            row = report[report.PLAYER_NAME.eq(name)]
            column = f'{side.title()}Cluster'
            override_audit.append({'Side': side, 'PLAYER_NAME': name, 'ExpectedCluster': cluster,
                                   'PresentInWorkbook': not row.empty,
                                   'ActualCluster': None if row.empty else int(row.iloc[0][column])})
            if not row.empty and not row[column].eq(cluster).all():
                raise AssertionError(f'Override lost: {side} {name}')
    pd.DataFrame(override_audit).to_csv(output / 'override_verification.csv', index=False)
    unresolved = report[report.ModelEligibleMinutes & (report.OffCluster.isna() | report.DefCluster.isna())]
    unresolved.to_csv(output / 'rotation_players_needing_archetypes.csv', index=False)
    source_manifest = Path(minutes_path).parent / 'minutes_import.json'
    workbook_source = json.loads(source_manifest.read_text()) if source_manifest.exists() else None
    if workbook_source:
        (output / 'minutes_import.json').write_text(json.dumps(workbook_source, indent=2) + '\n')
    audit_path = output / 'season_refresh.json'
    audit = json.loads(audit_path.read_text())
    audit.update({'workbook_source': workbook_source, 'roster_players': len(report), 'teams': len(counts),
                  'historical_profile_matches': int(report.HasHistoricalProfile.sum()),
                  'model_eligible_players': int(report.ModelEligibleMinutes.sum()),
                  'modelled_players': len(next(iter(exports.values()))),
                  'rotation_players_missing_profiles': int((~report.HasHistoricalProfile & report.ModelEligibleMinutes).sum()),
                  'rotation_players_missing_archetypes': len(unresolved),
                  'offensive_overrides_stored': len(rules.get('OFF_CLUSTER_OVERRIDES', {})),
                  'defensive_overrides_stored': len(rules['DEF_CLUSTER_OVERRIDES']),
                  'overrides_stored': len(override_audit),
                  'overrides_present': sum(r['PresentInWorkbook'] for r in override_audit),
                  'team_game_count_source': 'User states no games played; all team counts initialized to zero',
                  'roster_source': 'User workbook, not a verified NBA roster',
                  'feature_coverage': 'Repository 2025-26 partial-season snapshots; games_played ranges 0 to 56',
                  'live_stats_check': 'NBA Stats connectivity check timed out; no new features fetched',
                  'reference_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'references').glob('*.csv')}})
    audit_path.write_text(json.dumps(audit, indent=2) + '\n')
    return report, audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--minutes', required=True)
    parser.add_argument('--output', default='.')
    parser.add_argument('--opening-games', type=int, default=OPENING_GAMES)
    args = parser.parse_args()
    _, audit = bootstrap(args.minutes, args.output, args.opening_games)
    print(json.dumps({k: audit[k] for k in ['roster_players', 'teams', 'historical_profile_matches',
          'modelled_players', 'rotation_players_missing_profiles', 'overrides_present']}, indent=2))
