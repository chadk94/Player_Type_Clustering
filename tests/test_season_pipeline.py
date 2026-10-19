import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from season_pipeline import (ROOT, SEASON_ID, apply_overrides,
                             choose_minutes, load_rules, read_season_clusters, run)


class SeasonPipelineTests(unittest.TestCase):
    def test_all_manual_overrides_win(self):
        rules = load_rules()
        overrides = rules['DEF_CLUSTER_OVERRIDES']
        frame = pd.DataFrame({'PLAYER_NAME': list(overrides), 'Cluster': -1})
        result = apply_overrides(frame, 'def')
        self.assertEqual(result.Cluster.tolist(), list(overrides.values()))
        self.assertTrue(result.AssignmentSource.eq('manual_override').all())
        for row in result.itertuples():
            self.assertEqual(row.Archetype, rules['DEF_CLUSTER_NAMES'][str(row.Cluster)])

    def test_unmodified_player_and_offensive_override(self):
        frame = pd.DataFrame({'PLAYER_NAME': ['Example', 'Other'], 'Cluster': [0, 1]})
        rules = load_rules()
        rules['OFF_CLUSTER_OVERRIDES'] = {'Example': 2}
        result = apply_overrides(frame, 'off', rules)
        self.assertEqual(result.Cluster.tolist(), [2, 1])
        self.assertEqual(result.AssignmentSource.tolist(), ['manual_override', 'auto_model'])

    def test_first_x_team_games_and_zero_projection(self):
        self.assertEqual(choose_minutes(25, 32, 0, 10), 32)
        self.assertEqual(choose_minutes(25, 32, 9, 10), 32)
        self.assertEqual(choose_minutes(25, 32, 10, 10), 25)
        self.assertEqual(choose_minutes(25, 0, 9, 10), 0)
        self.assertEqual(choose_minutes(25, 32, 0, 0), 25)
        for value in [-1, np.nan, 49]:
            with self.assertRaises(ValueError):
                choose_minutes(25, value, 1)

    def test_reject_historical_exports(self):
        with self.assertRaises(ValueError):
            read_season_clusters(ROOT / 'references/player_clusters_detailed.csv')

    def test_full_refresh_and_missing_inputs(self):
        # Synthetic fixture verifies plumbing only; never delivered as real season data.
        data = pd.read_csv(ROOT / 'references/player_clusters_detailed.csv')
        data.SEASON_ID = SEASON_ID
        data['TEAM_ID'] = 1
        minutes = data[['PLAYER_ID', 'TEAM_ID', 'SEASON_ID']].copy()
        minutes['PROJECTED_MIN'] = 30.
        counts = pd.DataFrame({'TEAM_ID': [1], 'SEASON_ID': [SEASON_ID], 'GAMES_PLAYED': [0]})
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data.to_csv(tmp / 'features.csv', index=False)
            minutes.to_csv(tmp / 'minutes.csv', index=False)
            counts.to_csv(tmp / 'games.csv', index=False)
            exports = run(tmp / 'features.csv', tmp / 'minutes.csv', tmp / 'games.csv', tmp / 'out')
            for frame in exports.values():
                self.assertTrue(frame.SEASON_ID.eq(SEASON_ID).all())
                self.assertTrue(frame.EffectiveMinutes.eq(30).all())
                np.testing.assert_allclose(frame.PTS_per36, frame.PTS * 36 / frame.MIN)
            defense = exports['def_player_clusters_detailed.csv'].set_index('PLAYER_NAME')
            for name, cluster in load_rules()['DEF_CLUSTER_OVERRIDES'].items():
                if name in defense.index:
                    self.assertEqual(defense.loc[name, 'Cluster'], cluster)
            minutes.iloc[1:].to_csv(tmp / 'minutes.csv', index=False)
            with self.assertRaises(ValueError):
                run(tmp / 'features.csv', tmp / 'minutes.csv', tmp / 'games.csv', tmp / 'missing')
            self.assertFalse((tmp / 'missing').exists())


if __name__ == '__main__':
    unittest.main()
