import json
import unittest
import numpy as np
import pandas as pd
from archetype_model import (assign_named_roles, named_roles, validate_named_roles, load_reviews)
from season_pipeline import ROOT, apply_overrides, load_rules
from matchup_projections import assign_game_archetypes, matchup_effects
from audit_archetypes import build_audit


class ArchetypeAuditTests(unittest.TestCase):
    def test_all_100_reviewed_and_named_with_user_overrides_protected(self):
        before = pd.read_csv(ROOT/'data/audit/affected_players_before.csv')
        review = load_reviews()
        roster = pd.read_csv(ROOT/'preseason_archetypes.csv').set_index('PLAYER_ID')
        self.assertEqual(len(before),100)
        self.assertTrue(set(before.PLAYER_ID).issubset(set(review.PLAYER_ID)))
        self.assertEqual(len(pd.read_csv(ROOT/'archetype_reviews.csv')),100)
        rules = load_rules()
        for row in review.itertuples():
            player = roster.loc[row.PLAYER_ID]
            for side in ['Off','Def']:
                manual = rules[f'{side.upper()}_CLUSTER_OVERRIDES'].get(row.PLAYER_NAME)
                expected = manual if manual is not None else getattr(row,f'{side}Cluster')
                self.assertEqual(player[f'{side}Cluster'],expected)
                self.assertIn(expected,named_roles(rules,side.lower()))
                self.assertEqual(player[f'{side}AssignmentSource'],'manual_override' if manual is not None else 'analyst_review')
        self.assertEqual(len(build_audit()),100)

    def test_user_override_wins_even_when_review_disagrees(self):
        rules = load_rules()
        rules['OFF_CLUSTER_OVERRIDES']['Draymond Green']=2
        # Deliberately bad model plus a reviewed role of 6; user override is 2.
        frame = pd.DataFrame({'PLAYER_NAME':['Draymond Green'],'PLAYER_ID':[203110],'Cluster':[3]})
        result = apply_overrides(frame,'off',rules)
        self.assertEqual(result.iloc[0].Cluster,2)
        self.assertEqual(result.iloc[0].AssignmentSource,'manual_override')

    def test_unnamed_categories_block_publication(self):
        rules = load_rules()
        for side,role in [('off',3),('def',2),('def',9),('def',10),('def',13)]:
            with self.assertRaises(ValueError):
                validate_named_roles(pd.DataFrame({'Cluster':[role]}),side,rules)
        # Outside-cutoff / genuinely missing profiles are not fake assignments.
        validate_named_roles(pd.DataFrame({'Cluster':[np.nan]}),'off',rules)

    def test_named_model_ignores_batch_and_outlier_cluster_size(self):
        rules = load_rules()
        reference = pd.read_csv(ROOT/'references/player_clusters_detailed.csv')
        features = json.loads((ROOT/'clustering_features.json').read_text())['cluster_players_off']
        target = reference[reference.PLAYER_NAME.eq('Derrick White')].copy()
        self.assertEqual(len(target),1)
        # New profiles have no identity match: centroids are fixed, not fit to batch.
        target['PLAYER_ID']=999999999
        target['PLAYER_NAME']='New unseen guard'
        alone,_ = assign_named_roles(target,reference,features,'off',rules,apply_overrides)
        crowd = pd.concat([target]*50,ignore_index=True)
        together,_ = assign_named_roles(crowd,reference,features,'off',rules,apply_overrides)
        self.assertTrue(together.Cluster.eq(alone.iloc[0].Cluster).all())
        self.assertNotEqual(alone.iloc[0].Cluster,3)
        self.assertNotIn(alone.iloc[0].Cluster,[1,7,8,9])
        prior,_ = assign_named_roles(target,reference,features,'off',rules,apply_overrides,prior_mode=True)
        self.assertEqual(prior.iloc[0].Cluster,alone.iloc[0].Cluster)

    def test_prior_role_is_stable_across_roster_and_minutes_changes(self):
        rules=load_rules()
        reference=pd.read_csv(ROOT/'references/player_clusters_detailed.csv')
        features=json.loads((ROOT/'clustering_features.json').read_text())['cluster_players_off']
        player=reference[reference.PLAYER_NAME.eq('Derrick White')].copy()
        alone,_=assign_named_roles(player,reference,features,'off',rules,apply_overrides,prior_mode=True)
        batch,_=assign_named_roles(reference,reference,features,'off',rules,apply_overrides,prior_mode=True)
        expected=batch.loc[batch.PLAYER_ID.eq(player.iloc[0].PLAYER_ID),'Cluster'].iloc[0]
        self.assertEqual(alone.iloc[0].Cluster,expected)
        player['PROJECTED_MIN']=12
        changed,_=assign_named_roles(player,reference,features,'off',rules,apply_overrides,prior_mode=True)
        self.assertEqual(changed.iloc[0].Cluster,expected)

    def test_big_role_guardrail_and_human_exception(self):
        rules=load_rules()
        reference=pd.read_csv(ROOT/'references/def_player_clusters_detailed.csv')
        features=json.loads((ROOT/'clustering_features.json').read_text())['cluster_players_def']
        guard=reference[reference.PLAYER_NAME.eq('Derrick White')].copy()
        guard['PLAYER_ID']=999999998;guard['PLAYER_NAME']='New small guard'
        result,_=assign_named_roles(guard,reference,features,'def',rules,apply_overrides)
        self.assertNotIn(result.iloc[0].Cluster,[1,3,7])
        rules['DEF_CLUSTER_OVERRIDES']['New small guard']=1
        exception,_=assign_named_roles(guard,reference,features,'def',rules,apply_overrides)
        self.assertEqual(exception.iloc[0].Cluster,1)

    def test_matchup_samples_use_named_reviewed_roles_for_every_player(self):
        roster=pd.read_csv(ROOT/'preseason_archetypes.csv')
        logs=pd.read_csv(ROOT/'data/game_logs_2026.csv.gz')
        assigned=assign_game_archetypes(logs,roster)
        for kind in ['Off','Def']:
            valid=named_roles(load_rules(),kind.lower())
            self.assertTrue(assigned[f'{kind}Cluster'].dropna().isin(valid).all())
        for name,off,defense in [('Draymond Green',6,8),('Mikal Bridges',0,11),('Dyson Daniels',6,11),('Jarrett Allen',1,1)]:
            sample=assigned[assigned.PLAYER_NAME.eq(name)]
            self.assertGreater(len(sample),0)
            self.assertTrue(sample.OffCluster.eq(off).all())
            self.assertTrue(sample.DefCluster.eq(defense).all())
        effects=matchup_effects(assigned)
        for side,cluster,opponent in effects:
            self.assertIn(cluster,named_roles(load_rules(),side.lower()))

    def test_all_33_additional_players_keep_roles_below_training_cutoff(self):
        reviewed=pd.read_csv(ROOT/'archetype_rotation_reviews.csv')
        self.assertEqual(len(reviewed),33)
        roster=pd.read_csv(ROOT/'preseason_archetypes.csv').set_index('PLAYER_ID')
        selected=roster.loc[reviewed.PLAYER_ID]
        self.assertTrue(selected.PROJECTED_MIN.le(10).all())
        self.assertTrue(selected[['OffCluster','DefCluster']].notna().all().all())
        self.assertTrue(roster.loc[roster.PROJECTED_MIN.gt(0),['OffCluster','DefCluster']].notna().all().all())
        kyle=roster[roster.PLAYER_NAME.eq('Kyle Anderson')].iloc[0]
        self.assertEqual(kyle.DefCluster,12)
        self.assertEqual(kyle.DefAssignmentSource,'manual_override')
        from preseason_bootstrap import bootstrap
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            refreshed,_=bootstrap(ROOT/'data/preseason/projected_minutes.csv',folder)
        refreshed=refreshed.set_index('PLAYER_ID').loc[reviewed.PLAYER_ID]
        self.assertEqual(refreshed.OffCluster.tolist(),selected.OffCluster.tolist())
        self.assertEqual(refreshed.DefCluster.tolist(),selected.DefCluster.tolist())
        # A player with no saved features receives labels, not invented observations.
        swain=roster[roster.PLAYER_NAME.eq('Dailyn Swain')].iloc[0]
        self.assertFalse(swain.HasHistoricalProfile)
        self.assertTrue(pd.isna(swain.SourceSeasonID))



if __name__=='__main__':
    unittest.main()
