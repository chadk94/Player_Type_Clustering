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
        self.assertTrue(pd.isna(late.iloc[0].PTS))
        self.assertEqual(late.iloc[0]['Off samples'],0)
        current = {key:{**value,'PTS':20.} for key,value in effects.items()}
        self.assertEqual(apply_matchup(self.base,self.roster,self.teams,effects,current).iloc[0].PTS,12.)

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

    def test_historical_snapshot_is_full_regular_season(self):
        logs = pd.read_csv(ROOT / 'data/game_logs_2026.csv.gz')
        self.assertEqual(logs.GAME_ID.nunique(),1230)
        self.assertTrue(logs.groupby('Team').GAME_ID.nunique().eq(82).all())
        self.assertTrue((logs.REB == logs.OREB+logs.DREB).all())
        self.assertFalse(logs.duplicated(['GAME_ID','ESPN_PLAYER_ID']).any())


if __name__ == '__main__':
    unittest.main()
