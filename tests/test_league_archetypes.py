"""Every saved roster slot has roles even when its projected minutes are zero."""
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from archetype_model import load_reviews, named_roles
from preseason_bootstrap import bootstrap
from season_pipeline import ROOT, apply_overrides, apply_roster_rules, load_rules


class LeagueArchetypeTests(unittest.TestCase):
    def test_complete_roster_preserves_every_previously_assigned_side(self):
        before = pd.read_csv(ROOT/'data/audit/league_roster_before_2026-10-27.csv')
        after = pd.read_csv(ROOT/'preseason_archetypes.csv')
        review = pd.read_csv(ROOT/'archetype_league_reviews.csv')
        self.assertEqual(len(after), 541)
        self.assertEqual(len(review), 176)
        missing = before.OffCluster.isna() | before.DefCluster.isna()
        self.assertEqual(set(before.loc[missing, 'PLAYER_ID']), set(review.PLAYER_ID))
        self.assertTrue(before.loc[missing, 'PROJECTED_MIN'].eq(0).all())
        for kind in ['Off','Def']:
            self.assertTrue(after[kind+'Cluster'].isin(named_roles(load_rules(), kind.lower())).all())
            assigned = before[kind+'Cluster'].notna()
            for suffix in ['Cluster','Archetype','AssignmentSource']:
                pd.testing.assert_series_equal(before.loc[assigned,kind+suffix],after.loc[assigned,kind+suffix])
        roles = [kind+suffix for kind in ['Off','Def'] for suffix in ['Cluster','Archetype','AssignmentSource']]
        pd.testing.assert_frame_equal(before.drop(columns=roles),after.drop(columns=roles))
        # Stale exports must be repaired at the live boundary without changing minutes.
        pd.testing.assert_frame_equal(apply_roster_rules(before), after)

    def test_manual_rules_and_reference_observations_unchanged(self):
        manifest = json.loads((ROOT/'data/audit/league_coverage_manifest.json').read_text())
        self.assertEqual(hashlib.sha256((ROOT/'archetype_overrides.json').read_bytes()).hexdigest(),
                         manifest['manual_override_file_sha256'])
        for filename,digest in manifest['reference_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/'references'/filename).read_bytes()).hexdigest(),digest)
        for side,filename in [('Off','player_clusters_detailed.csv'),('Def','def_player_clusters_detailed.csv')]:
            detailed = pd.read_csv(ROOT/filename)
            self.assertTrue(detailed.Cluster.notna().all())
            no_features = detailed[~detailed.HasHistoricalProfile]
            self.assertTrue(no_features.SourceSeasonID.isna().all())
            self.assertTrue(no_features.MIN.isna().all())
        rules = load_rules()
        rules['OFF_CLUSTER_OVERRIDES']['Braden Smith'] = 6
        frame = pd.DataFrame({'PLAYER_ID':[int(pd.read_csv(ROOT/'archetype_league_reviews.csv').set_index('PLAYER_NAME').loc['Braden Smith','PLAYER_ID'])],
                              'PLAYER_NAME':['Braden Smith'],'Cluster':[4],'PROJECTED_MIN':[0]})
        result = apply_overrides(frame,'off',rules)
        self.assertEqual(result.iloc[0].Cluster,6)  # Overrides the new review's 2.
        self.assertEqual(result.iloc[0].AssignmentSource,'manual_override')

    def test_regeneration_keeps_zero_minute_roles_without_fabricating_features(self):
        expected = pd.read_csv(ROOT/'preseason_archetypes.csv').set_index('PLAYER_ID')
        with tempfile.TemporaryDirectory() as folder:
            refreshed,_ = bootstrap(ROOT/'data/preseason/projected_minutes.csv',folder)
        refreshed = refreshed.set_index('PLAYER_ID').loc[expected.index]
        for column in ['OffCluster','DefCluster','PROJECTED_MIN','HasHistoricalProfile']:
            pd.testing.assert_series_equal(refreshed[column],expected[column])

    def test_historical_fallback_itself_has_no_zero_minute_cutoff(self):
        # Remove one new review to exercise the actual historical fallback path.
        reviews = load_reviews()
        reviews = reviews[reviews.PLAYER_NAME.ne('Asa Newell')]
        with patch('archetype_model.load_reviews',return_value=reviews), tempfile.TemporaryDirectory() as folder:
            refreshed,_ = bootstrap(ROOT/'data/preseason/projected_minutes.csv',folder)
        player = refreshed.set_index('PLAYER_NAME').loc['Asa Newell']
        self.assertEqual(player.PROJECTED_MIN,0)
        self.assertTrue(pd.notna(player.OffCluster) and pd.notna(player.DefCluster))
        self.assertEqual(player.OffAssignmentSource,'historical_named_role_low_minutes')


if __name__ == '__main__':
    unittest.main()
