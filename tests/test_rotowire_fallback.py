import io
import unittest
import pandas as pd

from import_minutes import TEAM_IDS
from live_projections import CORE_STATS, STAT_NAMES, build_projections
from rotowire_fallback import normalize_rotowire, read_rotowire_csv
from season_pipeline import ROOT


class RotoWireFallbackTests(unittest.TestCase):
    def setUp(self):
        self.roster = pd.DataFrame([{'PLAYER_ID':1,'PLAYER_NAME':'Example Rookie',
            'WorkbookPlayerName':'Example Rookie','TEAM_ID':TEAM_IDS['ATL'],
            'TEAM_ABBREVIATION':'ATL','PROJECTED_MIN':32.,'OffArchetype':'Secondary Playmaker',
            'DefArchetype':'POA Defender'}])
        self.export = pd.DataFrame([{'Player':'Example Rookie','MIN':20.,'PTS':10.,
            'REB':4.,'AST':3.,'STL':1.,'BLK':.5,'3PM':1.,'TO':2.}])
        self.history = pd.DataFrame(columns=['PLAYER_ID','MIN',*STAT_NAMES])

    def test_export_rates_scale_to_user_minutes_without_inventing_extra_stats(self):
        fallback, unmatched = normalize_rotowire(self.export,self.roster)
        self.assertTrue(unmatched.empty)
        teams = pd.DataFrame([{'TEAM_ID':TEAM_IDS['ATL'],'GAMES_PLAYED':2}])
        row = build_projections(self.roster,self.history,teams,pd.DataFrame(),fallback=fallback).iloc[0]
        self.assertEqual(row.PTS,16.)
        self.assertEqual(row.FG3M,1.6)
        self.assertEqual(row['Projected minutes'],32.)
        self.assertEqual(row['Stat baseline'],'RotoWire 2026–27 projection')
        self.assertEqual(row.Status,'Available')
        self.assertTrue(pd.isna(row.FGA))
        self.assertTrue(pd.isna(row.PF))

    def test_optional_fouls_scale_with_minutes_and_can_use_current_rates(self):
        fallback,_ = normalize_rotowire(self.export.assign(FOULS=2.),self.roster)
        teams = pd.DataFrame([{'TEAM_ID':TEAM_IDS['ATL'],'GAMES_PLAYED':0}])
        row = build_projections(self.roster,self.history,teams,pd.DataFrame(),fallback=fallback,
                                minutes_overrides={1:30.}).iloc[0]
        self.assertEqual(row.PF,3.)
        no_fouls,_ = normalize_rotowire(self.export,self.roster)
        current = pd.DataFrame([{'ESPN_PLAYER_ID':'100','PLAYER_NAME':'Example Rookie',
            'GP':4.,'MIN':24.,**{s:2. for s in CORE_STATS},'PF':3.}])
        teams.GAMES_PLAYED = 4
        row = build_projections(self.roster,self.history,teams,current,fallback=no_fouls).iloc[0]
        self.assertEqual(row.PF,3.)
        row = build_projections(self.roster,self.history,teams,current.assign(PF=float('nan')),fallback=fallback).iloc[0]
        self.assertAlmostEqual(row.PF,2.4)

    def test_roto_minutes_do_not_override_season_minutes_after_game_three(self):
        fallback,_ = normalize_rotowire(self.export,self.roster)
        teams = pd.DataFrame([{'TEAM_ID':TEAM_IDS['ATL'],'GAMES_PLAYED':3}])
        row = build_projections(self.roster,self.history,teams,pd.DataFrame(),fallback=fallback).iloc[0]
        self.assertTrue(pd.isna(row['Projected minutes']))
        self.assertTrue(pd.isna(row.PTS))
        observed = pd.DataFrame([{'ESPN_PLAYER_ID':'100','PLAYER_NAME':'Example Rookie',
            'GP':2.,'MIN':24.,**{s:12. for s in CORE_STATS}}])
        row = build_projections(self.roster,self.history,teams,observed,fallback=fallback).iloc[0]
        self.assertEqual(row['Projected minutes'],24.)
        self.assertEqual(row['Minutes source'],'Season average')
        self.assertEqual(row['Stat baseline'],'Current season')
        self.assertEqual(row.PTS,12.)
        teams.GAMES_PLAYED = 4
        row = build_projections(self.roster,self.history,teams,observed,fallback=fallback).iloc[0]
        self.assertEqual(row['Stat baseline'],'Current season')
        self.assertEqual(row.PTS,12.)

    def test_existing_historical_profile_takes_precedence(self):
        fallback,_ = normalize_rotowire(self.export,self.roster)
        history = pd.DataFrame([{'PLAYER_ID':1,'MIN':20.,**{s:2. for s in CORE_STATS}}])
        teams = pd.DataFrame([{'TEAM_ID':TEAM_IDS['ATL'],'GAMES_PLAYED':0}])
        row = build_projections(self.roster,history,teams,pd.DataFrame(),fallback=fallback).iloc[0]
        self.assertEqual(row.PTS,3.2)
        self.assertEqual(row['Stat baseline'],'2025–26 archived snapshot')

    def test_unclassified_players_use_rotowire_at_approved_minutes(self):
        fallback,_ = normalize_rotowire(self.export,self.roster)
        history = pd.DataFrame([{'PLAYER_ID':1,'MIN':20.,**{s:2. for s in CORE_STATS}}])
        teams = pd.DataFrame([{'TEAM_ID':TEAM_IDS['ATL'],'GAMES_PLAYED':0}])
        roster = self.roster.assign(OffArchetype=None)
        before = roster.copy(deep=True)
        row = build_projections(roster,history,teams,pd.DataFrame(),fallback=fallback,
            approved_minutes={1:12.25}).iloc[0]
        self.assertEqual(row.PTS,6.125)
        self.assertEqual(row['Projected minutes'],12.25)
        self.assertEqual(row['Minutes source'],'Approved game minutes')
        self.assertEqual(row['Stat baseline'],'RotoWire 2026–27 projection')
        self.assertTrue(pd.isna(row['Offensive archetype']))
        self.assertEqual(row['Defensive archetype'],'POA Defender')
        pd.testing.assert_frame_equal(roster,before)

    def test_totals_and_per36_have_consistent_denominators(self):
        totals=self.export.copy()
        for col in totals.columns[1:]:
            totals[col] *= 10
        totals['GP']=10
        normalized,_=normalize_rotowire(totals,self.roster,'season_totals')
        self.assertEqual(normalized.iloc[0].MIN,20.)
        self.assertEqual(normalized.iloc[0].PTS,10.)
        per36,_=normalize_rotowire(self.export,self.roster,'per_36')
        self.assertEqual(per36.iloc[0].MIN,36.)

    def test_two_line_export_header_and_unique_suffix_matching(self):
        text = ',,,,Popular Stats\nNAME,G,MIN,PTS,REB,AST,STL,BLK,3PM,TO\nExample Rookie,70,20,10,4,3,1,0.5,1,2\n'
        self.roster['PLAYER_NAME'] = 'Example Rookie Jr.'
        self.roster['WorkbookPlayerName'] = 'Example Rookie Jr.'
        normalized,_ = normalize_rotowire(read_rotowire_csv(io.StringIO(text)),self.roster)
        self.assertEqual(normalized.iloc[0].PLAYER_ID,1)
        ambiguous = pd.concat([self.roster,self.roster.assign(PLAYER_ID=2,
            PLAYER_NAME='Example Rookie Sr.',WorkbookPlayerName='Example Rookie Sr.')])
        with self.assertRaises(ValueError):
            normalize_rotowire(read_rotowire_csv(io.StringIO(text)),ambiguous)

    def test_real_sources_fill_31_missing_baselines_without_fabricating_hezonja(self):
        roster = pd.read_csv(ROOT / 'preseason_archetypes.csv')
        history = pd.concat([pd.read_csv(ROOT / 'historical_player_data.csv'),
                             pd.read_csv(ROOT / 'data/prior_stat_supplements.csv')],ignore_index=True)
        fallback = pd.read_csv(ROOT / 'data/rotowire_fallback.csv')
        result = build_projections(roster,history,pd.read_csv(ROOT / 'team_games.csv'),
                                   pd.DataFrame(),fallback=fallback)
        active = result[result['Projected minutes'].gt(0)]
        self.assertEqual(len(active),318)
        self.assertEqual(active.PTS.notna().sum(),317)
        self.assertEqual(active[active.PTS.isna()].Player.tolist(),['Mario Hezonja'])
        indexed = result.set_index('Player')
        self.assertEqual(indexed.loc['Kingston Flemings','Stat baseline'],'RotoWire 2026–27 projection')
        self.assertEqual(indexed.loc['Pacôme Dadiet','Stat baseline'],'2025–26 ESPN season summary')
        self.assertEqual(indexed.loc['Kingston Flemings','Projected minutes'],16.)

    def test_invalid_minutes_missing_core_and_duplicate_matches_are_rejected(self):
        for frame in [self.export.assign(MIN=0),self.export.drop(columns='STL'),pd.concat([self.export,self.export])]:
            with self.assertRaises(ValueError):
                normalize_rotowire(frame,self.roster)
        with self.assertRaises(ValueError):
            normalize_rotowire(self.export.assign(Player='Unknown'),self.roster)


if __name__=='__main__':
    unittest.main()
