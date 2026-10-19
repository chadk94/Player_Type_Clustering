import unittest

import numpy as np
import pandas as pd

from import_minutes import TEAM_IDS
from live_projections import (STAT_NAMES, build_projections, match_player_stats,
                              parse_player_summary, parse_schedule, parse_standings)


class LiveProjectionTests(unittest.TestCase):
    def setUp(self):
        self.roster = pd.DataFrame([{'PLAYER_ID': 1, 'PLAYER_NAME':'Example Guard',
            'WorkbookPlayerName':'Example Guard','TEAM_ID': TEAM_IDS['ATL'],
            'TEAM_ABBREVIATION':'ATL','PROJECTED_MIN':32.,'OffArchetype':'Secondary Playmaker',
            'DefArchetype':'Pesky Guard'}])
        self.history = pd.DataFrame([{'PLAYER_ID':1,'MIN':20.,**{s:10. for s in STAT_NAMES}}])
        self.current = pd.DataFrame([{'ESPN_PLAYER_ID':'100','PLAYER_NAME':'Example Guard',
            'GP':2.,'MIN':24.,**{s:12. for s in STAT_NAMES}}])

    def project(self, count, current=None, history=None):
        teams = pd.DataFrame([{'TEAM_ID':TEAM_IDS['ATL'],'GAMES_PLAYED':count}])
        return build_projections(self.roster, self.history if history is None else history,
                                 teams, self.current if current is None else current).iloc[0]

    def test_first_three_team_games_then_season_appearance_average(self):
        for completed in [0,1,2]:
            result = self.project(completed)
            self.assertEqual(result['Projected minutes'],32.)
            self.assertEqual(result['Minutes source'],'Your workbook')
        for completed in [3,4,8]:
            result = self.project(completed)
            self.assertEqual(result['Projected minutes'],24.)
            self.assertEqual(result['Minutes source'],'Season average')
        # Team has played eight games, player only two: use 48 / 2, not 48 / 8.
        self.assertEqual(self.project(8)['Projected minutes'],24.)

    def test_zero_minutes_missing_profile_and_post_window_no_appearances(self):
        self.roster.PROJECTED_MIN = 0.
        self.assertEqual(self.project(2)['PTS'],0.)
        no_current = self.current.iloc[:0]
        result = self.project(3,current=no_current)
        self.assertTrue(pd.isna(result['Projected minutes']))
        self.assertTrue(pd.isna(result.PTS))
        no_history = self.history.iloc[:0]
        result = self.project(0,current=no_current,history=no_history)
        self.assertTrue(pd.isna(result.PTS))
        self.assertIn('Needs statistical profile',result.Status)
        self.assertEqual(self.project(2).PTS,0.)  # Workbook zero takes precedence in opening window.
        self.roster.PROJECTED_MIN = 32.
        self.assertTrue(pd.isna(self.project(1,history=no_history).PTS))
        self.assertEqual(self.project(4,history=no_history).PTS,12.)

    def test_rate_blend_does_not_replace_observed_denominator(self):
        self.current.loc[0,'PTS'] = 24.
        result = self.project(4)
        weight = 48 / (48 + 200)
        expected_rate = (1-weight) * (10/20) + weight * (24/24)
        self.assertAlmostEqual(result.PTS, expected_rate * 24)
        self.assertAlmostEqual(result.PRA,result.PTS+result.REB+result.AST)

    def test_stat_rates_wait_until_team_game_five(self):
        self.current.loc[0,'PTS'] = 24.
        self.assertEqual(self.project(2).PTS,16.)
        self.assertEqual(self.project(3).PTS,12.)
        self.assertGreater(self.project(4).PTS,12.)

    def test_custom_minutes_recompute_rates_including_zero_automatic_minutes(self):
        teams = pd.DataFrame([{'TEAM_ID':TEAM_IDS['ATL'],'GAMES_PLAYED':0}])
        self.roster.PROJECTED_MIN = 0.
        result = build_projections(self.roster,self.history,teams,self.current,minutes_overrides={1:30.}).iloc[0]
        self.assertEqual(result.PTS,15.)
        self.assertEqual(result['Projected minutes'],30.)
        self.assertEqual(result['Minutes source'],'Manual adjustment')
        zero = build_projections(self.roster,self.history,teams,self.current,minutes_overrides={1:0.}).iloc[0]
        self.assertEqual(zero.PTS,0.)
        self.assertEqual(self.roster.iloc[0].PROJECTED_MIN,0.)
        for invalid in [-1,49,np.nan]:
            with self.assertRaises(ValueError):
                build_projections(self.roster,self.history,teams,self.current,minutes_overrides={1:invalid})

    def test_ambiguous_names_and_unknown_team_counts_are_unavailable(self):
        duplicate = pd.concat([self.current,self.current.assign(ESPN_PLAYER_ID='200')])
        self.assertFalse(match_player_stats(self.roster,duplicate))
        teams = pd.DataFrame(columns=['TEAM_ID','GAMES_PLAYED'])
        row = build_projections(self.roster,self.history,teams,self.current).iloc[0]
        self.assertTrue(pd.isna(row.PTS))
        self.assertEqual(row.Status,'Team count unavailable')

    def test_reject_silent_prior_season_and_preseason_stats(self):
        for scope in [{'year':2026,'type':2},{'year':2027,'type':1}]:
            with self.assertRaises(ValueError):
                parse_player_summary({'requestedSeason':scope,'results':[]},2027)
        stats = [{'name':'gamesPlayed','value':3},{'name':'avgMinutes','value':22},
                 *[{'name':name,'value':6} for name in STAT_NAMES.values()]]
        payload = {'requestedSeason':{'year':2027,'type':2},'results':[{'leaders':[{
            'athlete':{'id':'200','displayName':'Example Guard'},
            'statistics':[{'stats':stats}]}]}]}
        row = parse_player_summary(payload,2027).iloc[0]
        self.assertEqual(row.GP,3)
        self.assertEqual(row.MIN,22)
        self.assertEqual(row.FG3M,6)

    def test_standings_require_all_30_current_regular_season_records(self):
        entries = [{'team':{'id':str(i),'abbreviation':abbr},
                    'stats':[{'name':'wins','value':2},{'name':'losses','value':1}]}
                   for i,abbr in enumerate(TEAM_IDS)]
        payload = {'children':[{'standings':{'season':2027,'seasonType':2,'entries':entries}}]}
        frame = parse_standings(payload,2027)
        self.assertEqual(len(frame),30)
        self.assertTrue(frame.GAMES_PLAYED.eq(3).all())
        payload['children'][0]['standings']['seasonType'] = 1
        with self.assertRaises(ValueError):
            parse_standings(payload,2027)
        payload['children'][0]['standings']['seasonType'] = 2
        payload['children'][0]['standings']['entries'] = entries[:29]
        with self.assertRaises(ValueError):
            parse_standings(payload,2027)

    def test_schedule_ignores_preseason_and_completed_games(self):
        event = {'season':{'year':2027,'type':2},'date':'2026-10-20T23:00Z',
                 'status':{'type':{'completed':False,'description':'Scheduled'}},
                 'competitions':[{'competitors':[{'team':{'abbreviation':'ATL'},'homeAway':'home'},
                                                 {'team':{'abbreviation':'NY'},'homeAway':'away'}]}]}
        self.assertEqual(len(parse_schedule({'events':[event]},2027)),2)
        event['season']['type'] = 1
        self.assertTrue(parse_schedule({'events':[event]},2027).empty)
        event['season']['type'] = 2
        event['status']['type']['completed'] = True
        self.assertTrue(parse_schedule({'events':[event]},2027).empty)


if __name__ == '__main__':
    unittest.main()
