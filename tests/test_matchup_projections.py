import unittest
import numpy as np
import pandas as pd
from matchup_data import parse_boxscore
from matchup_projections import (matchup_effects, apply_matchup, assign_game_archetypes,
                                 prior_season_profiles, OFF_STATS, DEF_STATS)
from season_pipeline import ROOT


class MatchupTests(unittest.TestCase):
    def setUp(self):
        self.roster = pd.DataFrame([{'PLAYER_ID':1,'PLAYER_NAME':'Example','WorkbookPlayerName':'Example',
                                    'OffCluster':6,'DefCluster':11}])
        self.logs = pd.DataFrame([
            {'ESPN_PLAYER_ID':'100','PLAYER_NAME':'Example','GAME_ID':str(i),'GAME_DATE':f'2025-11-0{i}',
             'Opponent':opponent,'MIN':36.,'OffCluster':6,'DefCluster':11,
             **{s:value for s in OFF_STATS+DEF_STATS},'REB':value*2}
            for i,opponent,value in [(1,'BOS',10.),(2,'NYK',30.)]])
        self.base = pd.DataFrame([{'PLAYER_ID':1,'Player':'Example','Opponent':'BOS',
                                  **{s:10. for s in OFF_STATS+DEF_STATS},'REB':20.,'PRA':40.}])
        self.teams = pd.DataFrame([{'TEAM_ABBREVIATION':'BOS','GAMES_PLAYED':3}])

    def test_opponent_effect_shrinkage_and_consistent_stat_totals(self):
        effects = matchup_effects(self.logs)
        self.assertAlmostEqual(effects['Off',6,'BOS']['PTS'],-12.5)
        self.assertAlmostEqual(effects['Off',6,'NYK']['PTS'],12.5)
        a = apply_matchup(self.base,self.roster,self.teams,effects,{})
        b = apply_matchup(self.base,self.roster,self.teams,effects,{},opponent='NYK')
        self.assertAlmostEqual(a.iloc[0].PTS,8.75)
        self.assertGreater(b.iloc[0].PTS,a.iloc[0].PTS)
        self.assertEqual(a.iloc[0].REB,a.iloc[0].OREB+a.iloc[0].DREB)
        self.assertEqual(a.iloc[0].PRA,a.iloc[0].PTS+a.iloc[0].REB+a.iloc[0].AST)
        self.assertEqual(a.iloc[0]['Off samples'],1)
        self.assertEqual(a.iloc[0]['FG3M %'],a.iloc[0]['FG3A %'])

    def test_opponent_game_five_boundary_and_missing_current_history(self):
        effects = matchup_effects(self.logs)
        early = apply_matchup(self.base,self.roster,self.teams,effects,{})
        self.assertEqual(early.iloc[0]['Matchup data'],'2025–26')
        self.teams.GAMES_PLAYED = 4
        late = apply_matchup(self.base,self.roster,self.teams,effects,{})
        self.assertEqual(late.iloc[0]['Matchup data'],'2026–27')
        self.assertEqual(late.iloc[0].PTS,10.)
        self.assertEqual(late.iloc[0]['Off samples'],0)
        self.assertTrue(pd.isna(late.iloc[0]['PTS %']))
        self.assertEqual(late.iloc[0]['Matchup status'],'Baseline fallback — missing matchup history')
        current = {key:{**value,'PTS':20.} for key,value in effects.items()}
        self.assertEqual(apply_matchup(self.base,self.roster,self.teams,effects,current).iloc[0].PTS,12.)

    def test_missing_defensive_history_preserves_team_coverage_and_real_offense(self):
        from team_projection_view import team_projection_tables
        from market_points import match_points
        prior = matchup_effects(self.logs)
        current = {('Off',6,'BOS'):prior['Off',6,'BOS']}
        teams = self.teams.assign(GAMES_PLAYED=4)
        base = self.base.assign(Team='ATL',**{'Projected minutes':24.})
        original = self.roster.copy(deep=True)
        result = apply_matchup(base,self.roster,teams,prior,current)
        row = result.iloc[0]
        self.assertEqual(row.PTS,8.75)  # Existing current offense still applies.
        self.assertEqual(row.DREB,10.)  # No old-season effect substituted.
        self.assertEqual(row.STL,10.)
        self.assertEqual(row.BLK,10.)
        self.assertEqual(row.REB,18.75)
        self.assertEqual(row.PRA,row.PTS+row.REB+row.AST)
        self.assertEqual(row['Def samples'],0)
        self.assertEqual(row['Matchup fallback stats'],'DREB, STL, BLK, PF')
        self.assertTrue(pd.isna(row['DREB %']))
        adjusted,_ = match_points(result,{'ATL':110.})
        totals,_,issues = team_projection_tables(base,result,adjusted,['ATL'])
        self.assertEqual(totals.iloc[0]['Stat coverage'],'Complete')
        self.assertFalse(issues)
        self.assertEqual(adjusted.iloc[0].PRA,110.+row.REB+row.AST)
        pd.testing.assert_frame_equal(original,self.roster)

    def test_missing_single_effect_keeps_baseline_but_not_missing_player_stats(self):
        effects = matchup_effects(self.logs)
        effects['Def',11,'BOS']['BLK'] = np.nan
        row = apply_matchup(self.base,self.roster,self.teams,effects,{}).iloc[0]
        self.assertEqual(row.BLK,10.)
        self.assertEqual(row.DREB,8.75)
        self.assertEqual(row['Matchup fallback stats'],'BLK')
        self.assertEqual(row['Def samples'],1)
        self.assertEqual(row['Matchup status'],'Baseline fallback — missing matchup history')
        missing = apply_matchup(self.base.assign(BLK=np.nan),self.roster,self.teams,effects,{}).iloc[0]
        self.assertTrue(pd.isna(missing.BLK))

    def test_steals_blocks_shrink_without_changing_other_stat_weights(self):
        # Two differently aged BOS observations test decay as well as the prior.
        last = self.logs.iloc[[1]].copy()
        last['GAME_ID'], last['GAME_DATE'], last['Opponent'] = '3', '2025-11-03', 'BOS'
        for stat in OFF_STATS + DEF_STATS:
            last[stat] = 50.
        logs = pd.concat([self.logs, last], ignore_index=True)
        effects = matchup_effects(logs)
        for kind, stats in [('Off', OFF_STATS), ('Def', DEF_STATS)]:
            for stat in stats:
                with self.subTest(stat=stat):
                    half = 15 if stat == 'AST' else 10 if stat in ['OREB','DREB','STL','BLK','PF'] else 999
                    older_weight = 2 ** (-1/half)
                    numerator = ((10/30-1)*older_weight + (50/30-1))*100
                    prior = 20 if stat in ['STL','BLK'] else 3
                    self.assertAlmostEqual(effects[kind,6 if kind=='Off' else 11,'BOS'][stat],
                                           numerator/(prior+older_weight+1))

    def test_steals_blocks_smoothing_applies_to_both_seasons_and_signs(self):
        prior = matchup_effects(self.logs)
        current_logs = self.logs.copy()
        current_logs[['STL','BLK']] = [[30.,30.],[10.,10.]]
        current = matchup_effects(current_logs)
        for completed, expected in [(3,10*(1-.5/21)),(4,10*(1+.5/21))]:
            with self.subTest(completed=completed):
                teams = self.teams.assign(GAMES_PLAYED=completed)
                row = apply_matchup(self.base,self.roster,teams,prior,current).iloc[0]
                self.assertAlmostEqual(row.STL,expected)
                self.assertAlmostEqual(row.BLK,expected)
                self.assertEqual(row['Def samples'],1)  # The prior is not real coverage.

    def test_missing_block_observations_are_not_neutral_prior_events(self):
        self.logs.loc[0,'BLK'] = np.nan
        effect = matchup_effects(self.logs)['Def',11,'BOS']
        self.assertTrue(np.isnan(effect['BLK']))
        self.assertAlmostEqual(effect['STL'],-50/21)

    def test_unclassified_side_keeps_baseline_and_known_side_keeps_effect(self):
        effects = matchup_effects(self.logs)
        roster = self.roster.assign(OffCluster=np.nan)
        before = roster.copy(deep=True)
        row = apply_matchup(self.base,roster,self.teams,effects,{}).iloc[0]
        self.assertEqual(row.PTS,10.)
        self.assertEqual(row.AST,10.)
        self.assertEqual(row.DREB,8.75)
        self.assertEqual(row.REB,18.75)
        self.assertEqual(row.PRA,38.75)
        self.assertEqual(row['Off samples'],0)
        self.assertTrue(pd.isna(row['PTS %']))
        self.assertEqual(row['Matchup status'],'Baseline fallback — missing archetype')
        pd.testing.assert_frame_equal(roster,before)
        missing = apply_matchup(self.base.assign(PTS=np.nan),roster,self.teams,effects,{}).iloc[0]
        self.assertTrue(pd.isna(missing.PTS))

    def test_baseline_fallback_contributes_to_team_point_allocation(self):
        from market_points import match_points
        roster = self.roster.assign(OffCluster=np.nan,DefCluster=np.nan)
        base = self.base.assign(Team='ATL',**{'Projected minutes':10.})
        fallback = apply_matchup(base,roster,self.teams,{}, {})
        classified = fallback.copy().assign(PLAYER_ID=2,PTS=90.,AST=20.,REB=30.,PRA=140.)
        combined = pd.concat([fallback,classified],ignore_index=True)
        result,_ = match_points(combined,{'ATL':110.})
        self.assertAlmostEqual(result.PTS.sum(),110.)
        self.assertAlmostEqual(result.iloc[0].PTS,11.)
        self.assertAlmostEqual(result.iloc[1].PTS,99.)
        self.assertAlmostEqual(result.iloc[0].PRA,41.)

    def test_fouls_use_defensive_matchup_and_missing_rates_remain_unavailable(self):
        self.logs['PF'] = [1.,3.]
        self.base['PF'] = 2.
        effects = matchup_effects(self.logs)
        self.assertNotIn('PF',effects['Off',6,'BOS'])
        self.assertEqual(effects['Def',11,'BOS']['PF'],-12.5)
        result = apply_matchup(self.base,self.roster,self.teams,effects,{})
        self.assertEqual(result.iloc[0].PF,1.75)
        self.assertEqual(result.iloc[0]['Baseline PF'],2.)
        missing = apply_matchup(self.base.assign(PF=np.nan),self.roster,self.teams,effects,{})
        self.assertTrue(pd.isna(missing.iloc[0].PF))
        self.logs.loc[0,'PF'] = np.nan
        self.assertTrue(pd.isna(matchup_effects(self.logs)['Def',11,'BOS']['PF']))

    def test_minute_cutoff_and_zero_baseline_are_not_fabricated(self):
        self.logs.loc[0,'MIN']=4
        self.logs['STL']=0
        effects = matchup_effects(self.logs)
        self.assertNotIn(('Off',6,'BOS'),effects)
        self.assertTrue(np.isnan(effects['Def',11,'NYK']['STL']))

    def test_current_manual_archetypes_win_over_archived_clusters(self):
        roster = pd.read_csv(ROOT / 'preseason_archetypes.csv')
        names = ['Kingston Flemings','Darryn Peterson','Cameron Carr']
        chosen = roster[roster.PLAYER_NAME.isin(names)]
        self.assertEqual(len(chosen),3)
        logs = pd.DataFrame({'PLAYER_NAME':names})
        assigned = assign_game_archetypes(logs,roster).set_index('PLAYER_NAME')
        self.assertEqual(assigned.loc['Kingston Flemings','OffCluster'],6)
        self.assertEqual(assigned.loc['Darryn Peterson','OffCluster'],6)
        self.assertEqual(assigned.loc['Cameron Carr','OffCluster'],4)

    def test_knicks_corrections_override_bad_model_labels_and_relabel_samples(self):
        from season_pipeline import apply_overrides
        names = ['OG Anunoby','Josh Hart']
        bad_model = pd.DataFrame({'PLAYER_NAME':names,'Cluster':[3,3],
                                  'AssignmentSource':['kmeans_aligned']*2})
        off = apply_overrides(bad_model,'off')
        defense = apply_overrides(bad_model,'def')
        self.assertEqual(off.Cluster.tolist(),[0,6])
        self.assertEqual(defense.Cluster.tolist(),[5,5])
        self.assertTrue(off.AssignmentSource.eq('manual_override').all())
        self.assertTrue(defense.AssignmentSource.eq('manual_override').all())
        roster = pd.read_csv(ROOT / 'preseason_archetypes.csv')
        selected = roster.set_index('PLAYER_NAME').loc[names]
        self.assertEqual(selected.OffCluster.tolist(),[0,6])
        self.assertEqual(selected.DefCluster.tolist(),[5,5])
        self.assertEqual(selected.OffArchetype.tolist(),['Catch & Shoot Wing','Secondary Playmaker'])
        self.assertEqual(selected.DefArchetype.tolist(),['Big Wing Defender']*2)
        logs = self.logs.copy()
        logs['PLAYER_NAME'] = names
        logs = logs.drop(columns=['OffCluster','DefCluster'])
        assigned = assign_game_archetypes(logs,roster)
        effects = matchup_effects(assigned)
        self.assertIn(('Off',0,'BOS'),effects)
        self.assertIn(('Off',6,'NYK'),effects)
        self.assertIn(('Def',5,'BOS'),effects)
        self.assertNotIn(('Off',3,'BOS'),effects)
        self.assertNotIn(('Def',3,'BOS'),effects)
        for side, expected in [('player',[0,6]),('def_player',[5,5])]:
            export = pd.read_csv(ROOT / f'{side}_clusters_detailed.csv').set_index('PLAYER_NAME').loc[names]
            self.assertEqual(export.Cluster.tolist(),expected)
            self.assertTrue(export.AssignmentSource.eq('manual_override').all())

    def test_full_prior_averages_use_observed_minutes(self):
        profiles = prior_season_profiles(self.logs,self.roster)
        self.assertEqual(profiles.iloc[0].MIN,36.)
        self.assertEqual(profiles.iloc[0].PTS,20.)

    def test_boxscore_rejects_wrong_season_and_cup_final(self):
        with self.assertRaises(ValueError):
            parse_boxscore({'header':{'season':{'year':2025,'type':2}}},2026)
        with self.assertRaises(ValueError):
            parse_boxscore({'header':{'season':{'year':2026,'type':1}}},2026)
        self.assertEqual(parse_boxscore({'header':{'season':{'year':2026,'type':2},'gameNote':'NBA Cup Championship'}},2026),[])

    def test_boxscore_reads_fouls_without_dropping_rows_when_pf_is_missing(self):
        values = {'MIN':'20','PTS':'10','AST':'2','REB':'4','OREB':'1','DREB':'3',
                  'STL':'1','BLK':'0','TO':'2','FG':'4-8','3PT':'1-3','FT':'1-2','PF':'3'}
        category = {'labels':list(values),'athletes':[{'athlete':{'id':'100','displayName':'Example'},'stats':list(values.values())}]}
        payload = {'header':{'id':'1','season':{'year':2026,'type':2},'competitions':[{
            'date':'2025-11-01T00:00Z','status':{'type':{'completed':True}},
            'competitors':[{'team':{'id':'1','abbreviation':'ATL'}},{'team':{'id':'2','abbreviation':'BOS'}}]}]},
            'boxscore':{'players':[{'team':{'id':'1'},'statistics':[category]}]}}
        self.assertEqual(parse_boxscore(payload,2026)[0]['PF'],3.)
        category['labels'].pop()
        category['athletes'][0]['stats'].pop()
        rows = parse_boxscore(payload,2026)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['PTS'],10.)
        self.assertTrue(pd.isna(rows[0]['PF']))

    def test_historical_snapshot_is_full_regular_season(self):
        logs = pd.read_csv(ROOT / 'data/game_logs_2026.csv.gz')
        self.assertEqual(logs.GAME_ID.nunique(),1230)
        self.assertTrue(logs.groupby('Team').GAME_ID.nunique().eq(82).all())
        self.assertTrue((logs.REB == logs.OREB+logs.DREB).all())
        self.assertFalse(logs.duplicated(['GAME_ID','ESPN_PLAYER_ID']).any())
        self.assertTrue(logs.PF.notna().all())
        self.assertTrue(logs.PF.ge(0).all())


if __name__ == '__main__':
    unittest.main()
