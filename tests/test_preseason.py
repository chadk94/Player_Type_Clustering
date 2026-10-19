import ast
import json
import os
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from import_minutes import normalize_rows, TEAM_IDS
from preseason_bootstrap import bootstrap
from season_pipeline import ROOT, SEASON_ID, apply_overrides, run


class PreseasonTests(unittest.TestCase):
    def test_workbook_mapping_preserves_zero_precision_and_ids(self):
        rows = [{'Team': 'ATL', 'Player': 'Example', 'Minutes': 32.98992617,
                 'Player ID': '1630552'}, {'Team': 'BOS', 'Player': 'Other',
                 'Minutes': 0, 'Player ID': '202681'}]
        result = normalize_rows(rows)
        self.assertEqual(result[0]['PLAYER_ID'], 1630552)
        self.assertEqual(result[0]['PROJECTED_MIN'], 32.98992617)
        self.assertEqual(result[0]['TEAM_ID'], TEAM_IDS['ATL'])
        self.assertEqual(result[1]['PROJECTED_MIN'], 0)
        self.assertEqual(result[1]['SourceRow'], 3)
        with self.assertRaises(ValueError):
            normalize_rows(rows + rows)

    def test_missing_profile_is_not_assigned_a_fake_cluster(self):
        frame = pd.DataFrame({'PLAYER_NAME': ['Scottie Barnes', 'Unknown Rookie'],
                              'Cluster': [np.nan, np.nan],
                              'AssignmentSource': ['missing_historical_profile'] * 2})
        result = apply_overrides(frame, 'def')
        self.assertEqual(result.loc[0, 'Cluster'], 8)
        self.assertEqual(result.loc[0, 'AssignmentSource'], 'manual_override')
        self.assertTrue(pd.isna(result.loc[1, 'Cluster']))
        self.assertTrue(pd.isna(result.loc[1, 'Archetype']))
        self.assertEqual(result.loc[1, 'AssignmentSource'], 'missing_historical_profile')

    def test_explicit_prior_keeps_source_year_and_rejects_silent_relabel(self):
        history = pd.read_csv(ROOT / 'references/player_clusters_detailed.csv')
        minutes = history[['PLAYER_ID', 'SEASON_ID']].assign(
            TEAM_ID=TEAM_IDS['ATL'], PROJECTED_MIN=30)
        minutes.SEASON_ID = SEASON_ID
        counts = pd.DataFrame({'TEAM_ID': [TEAM_IDS['ATL']],
                               'SEASON_ID': [SEASON_ID], 'GAMES_PLAYED': [0]})
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            history.to_csv(tmp / 'history.csv', index=False)
            minutes.to_csv(tmp / 'minutes.csv', index=False)
            counts.to_csv(tmp / 'games.csv', index=False)
            with self.assertRaises(ValueError):
                run(tmp / 'history.csv', tmp / 'minutes.csv', tmp / 'games.csv', tmp / 'silent')
            result = run(tmp / 'history.csv', tmp / 'minutes.csv', tmp / 'games.csv',
                         tmp / 'explicit', prior_season_id=SEASON_ID - 1)
            for frame in result.values():
                self.assertTrue(frame.SourceSeasonID.eq(SEASON_ID - 1).all())
                self.assertTrue(frame.SEASON_ID.eq(SEASON_ID).all())
                self.assertTrue(frame.FeatureDataStatus.eq('preseason_historical_prior').all())
                np.testing.assert_allclose(frame.PTS_per36, frame.PTS * 36 / frame.MIN)
            counts.GAMES_PLAYED = 1
            counts.to_csv(tmp / 'games.csv', index=False)
            with self.assertRaises(ValueError):
                run(tmp / 'history.csv', tmp / 'minutes.csv', tmp / 'games.csv',
                    tmp / 'played', prior_season_id=SEASON_ID - 1)

    def test_actual_workbook_roster_end_to_end(self):
        minutes = pd.read_csv(ROOT / 'data/preseason/projected_minutes.csv')
        with tempfile.TemporaryDirectory() as tmp:
            result, audit = bootstrap(ROOT / 'data/preseason/projected_minutes.csv', tmp)
            self.assertEqual(len(result), len(minutes))
            self.assertEqual(result.PLAYER_ID.nunique(), len(minutes))
            np.testing.assert_allclose(result.PROJECTED_MIN, minutes.PROJECTED_MIN)
            self.assertTrue(minutes.groupby('TEAM_ID').PROJECTED_MIN.sum().sub(240).abs().lt(1e-8).all())
            unresolved = result[~result.HasHistoricalProfile]
            for side in ['Off', 'Def']:
                nonmanual = unresolved[unresolved[f'{side}AssignmentSource'] != 'manual_override']
                self.assertTrue(nonmanual[f'{side}Cluster'].isna().all())
            zero = result[result.PROJECTED_MIN.eq(0)]
            self.assertFalse(zero.ModelEligibleMinutes.any())
            self.assertEqual(audit['mode'], 'preseason_historical_prior')
            self.assertEqual(audit['source_season_id'], SEASON_ID - 1)
            corrections = result.set_index('PLAYER_NAME').loc[['OG Anunoby','Josh Hart']]
            self.assertEqual(corrections.OffCluster.tolist(),[0,6])
            self.assertEqual(corrections.DefCluster.tolist(),[5,5])
            self.assertTrue(corrections.OffAssignmentSource.eq('manual_override').all())
            self.assertTrue(corrections.DefAssignmentSource.eq('manual_override').all())
            approved = pd.read_csv(ROOT / 'data/preseason/approved_archetypes.csv')
            assigned = result.set_index('PLAYER_NAME')
            self.assertEqual(len(approved), 26)
            for row in approved.itertuples():
                for side in ['Off', 'Def']:
                    self.assertEqual(assigned.loc[row.PLAYER_NAME, side + 'Cluster'], getattr(row, side + 'Cluster'))
                    self.assertEqual(assigned.loc[row.PLAYER_NAME, side + 'Archetype'], getattr(row, side + 'Archetype'))
                    self.assertEqual(assigned.loc[row.PLAYER_NAME, side + 'AssignmentSource'], 'manual_override')
                self.assertFalse(assigned.loc[row.PLAYER_NAME, 'HasHistoricalProfile'])
            for name, cluster in [('Kingston Flemings', 6), ('Yaxel Lendeborg', 5),
                                  ('Darryn Peterson', 6), ('Cameron Carr', 4)]:
                self.assertEqual(assigned.loc[name, 'OffCluster'], cluster)
            self.assertEqual(audit['rotation_players_missing_archetypes'], 0)
            # The preseason view must work before NBA game logs exist, without fetching them.
            tree = ast.parse((ROOT / 'main.py').read_text())
            main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
            ui = MagicMock()
            ui.selectbox.return_value = 'All'
            ui.checkbox.return_value = False
            live_loader = MagicMock(side_effect=AssertionError('Preseason must not fetch live games'))
            environment = {'st': ui, 'pd': pd, 'SEASON': audit['season'],
                           'SEASON_ID': SEASON_ID, 'load_data': live_loader}
            exec(compile(ast.Module(body=[main], type_ignores=[]), 'main.py', 'exec'), environment)
            previous = os.getcwd()
            try:
                os.chdir(tmp)
                renderer = MagicMock()
                import types
                fake_module = types.ModuleType('projection_dashboard')
                fake_module.render_live_dashboard = renderer
                with patch.dict('sys.modules', {'projection_dashboard': fake_module}):
                    environment['main']()
            finally:
                os.chdir(previous)
            live_loader.assert_not_called()
            displayed = renderer.call_args.args[1]
            self.assertEqual(len(displayed), len(minutes))


if __name__ == '__main__':
    unittest.main()
