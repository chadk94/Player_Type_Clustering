from datetime import datetime, timezone
import unittest

import numpy as np
import pandas as pd

from import_minutes import TEAM_IDS
from market_points import apply_market_points
from team_assists import apply_team_assists
from team_threes import apply_team_threes
from team_projection_view import upcoming_team_games, team_projection_tables


class TeamProjectionViewTests(unittest.TestCase):
    def setUp(self):
        self.base = pd.DataFrame([{
            'PLAYER_ID': i, 'Team': team, 'Opponent': opponent,
            'Projected minutes': 40., 'PTS': 20., 'AST': 4., 'REB': 6.,
            'FG3M': 1., 'FG3A': 3., 'STL': 1., 'BLK': .5, 'TOV': 2., 'PRA': 30.,
        } for i, (team, opponent) in enumerate([('BOS', 'DET')]*6 + [('DET', 'BOS')]*6)])
        self.before = self.base.copy()
        self.before[['FG3M', 'FG3A']] *= 1.1
        self.targets = {'BOS': 110., 'DET': 115.}
        self.rates = pd.DataFrame({
            'Assist rate': [.6, .65], 'Allowed assist rate': [.62, .64],
            'Three rate': [.30, .32], 'Allowed three rate': [.34, .36],
            'Points per FGM': [2.5, 2.6], 'Allowed points per FGM': [2.4, 2.5],
        }, index=['BOS', 'DET'])
        final, _, _ = apply_market_points(self.before, self.targets)
        final, _, _ = apply_team_assists(final, self.targets, self.rates, {'BOS': 'DET', 'DET': 'BOS'})
        three_rates = self.rates.drop(columns=['Assist rate', 'Allowed assist rate'])
        self.final, _, issues = apply_team_threes(final, self.targets, three_rates, {'BOS': 'DET', 'DET': 'BOS'})
        self.assertEqual(issues, [])

    def test_totals_reconcile_to_player_outputs_and_target_is_not_summed(self):
        original = self.final.copy(deep=True)
        totals, breakdown, issues = team_projection_tables(self.base, self.before, self.final, ['BOS', 'DET'])
        self.assertEqual(issues, [])
        self.assertEqual(totals.Minutes.tolist(), [240., 240.])
        for _, row in totals.iterrows():
            players = self.final[self.final.Team.eq(row.Team)]
            self.assertAlmostEqual(row.PTS, self.targets[row.Team])
            self.assertAlmostEqual(row['3PM'], players.FG3M.sum())
            self.assertAlmostEqual(row.PRA, row.PTS + row.REB + row.AST)
        threes = breakdown[breakdown.Team.eq('BOS') & breakdown.Stat.eq('3PM')].iloc[0]
        self.assertAlmostEqual(threes.Baseline, 6.)
        self.assertAlmostEqual(threes['After archetypes'], 6.6)
        self.assertAlmostEqual(threes['Team target'], 14.52)
        self.assertAlmostEqual(threes['Final projection'], 14.52)
        self.assertAlmostEqual(threes['Team adjustment %'], 120.)
        pd.testing.assert_frame_equal(self.final, original)

    def test_missing_stat_is_blank_instead_of_a_partial_total(self):
        self.final.loc[0, 'FG3M'] = np.nan
        totals, _, issues = team_projection_tables(self.base, self.before, self.final, ['BOS', 'DET'])
        self.assertTrue(pd.isna(totals.iloc[0]['3PM']))
        self.assertAlmostEqual(totals.iloc[0].PTS, 110.)
        self.assertTrue(np.isfinite(totals.iloc[1]['3PM']))
        self.assertIn('3PM', issues[0])

    def test_unknown_minutes_blank_entire_team_but_zero_minute_missing_profile_is_ignored(self):
        for frame in [self.base, self.before, self.final]:
            frame.loc[len(frame)] = {**frame.iloc[0].to_dict(), 'PLAYER_ID': 100,
                                     'Projected minutes': 0., 'PTS': np.nan, 'FG3M': np.nan}
        totals, _, issues = team_projection_tables(self.base, self.before, self.final, ['BOS', 'DET'])
        self.assertEqual(issues, [])
        self.assertAlmostEqual(totals.iloc[0].PTS, 110.)
        self.final.loc[12, 'Projected minutes'] = np.nan
        totals, _, issues = team_projection_tables(self.base, self.before, self.final, ['BOS', 'DET'])
        self.assertTrue(totals.iloc[0].drop('Team').isna().all())
        self.assertEqual(len(issues), 1)
        self.assertAlmostEqual(totals.iloc[1].PTS, 115.)

    def test_no_market_fallback_keeps_model_and_has_no_target(self):
        totals, breakdown, issues = team_projection_tables(self.base, self.before, self.before, ['BOS', 'DET'])
        self.assertEqual(issues, [])
        self.assertTrue(breakdown['Team target'].isna().all())
        self.assertTrue(breakdown['Team adjustment %'].eq(0).all())
        self.assertTrue(totals.PTS.eq(120.).all())

    def test_rebounds_are_in_adjustment_breakdown_and_pra_reconciles(self):
        self.final['Pre-team REB'] = self.final.REB
        self.final.REB *= .8
        self.final.PRA = self.final.PTS + self.final.REB + self.final.AST
        self.final['Team REB target'] = self.final.groupby('Team').REB.transform('sum')
        totals, breakdown, issues = team_projection_tables(self.base, self.before, self.final, ['BOS', 'DET'])
        self.assertEqual(issues, [])
        rebounds = breakdown[breakdown.Stat.eq('REB')]
        self.assertEqual(len(rebounds), 2)
        np.testing.assert_allclose(rebounds['Team adjustment %'], -20.)
        np.testing.assert_allclose(rebounds['Final projection'], rebounds['Team target'])
        np.testing.assert_allclose(totals.PRA, totals.PTS + totals.REB + totals.AST)

    def test_dropped_or_duplicate_players_cannot_silently_change_team_totals(self):
        for invalid in [self.final.iloc[1:], pd.concat([self.final, self.final.iloc[:1]])]:
            with self.assertRaises(ValueError):
                team_projection_tables(self.base, self.before, invalid, ['BOS', 'DET'])

    def test_game_selection_works_without_books_and_excludes_expired_or_postponed_games(self):
        now = datetime(2026, 10, 20, 12, tzinfo=timezone.utc)
        game = {'TEAM_ID': TEAM_IDS['BOS'], 'Opponent': 'DET', 'Venue': 'home',
                'GameTimeUTC': '2026-10-20T20:00:00Z', 'GameStatus': 'Scheduled'}
        schedule = pd.DataFrame([game, game, {**game, 'Venue': 'away'},
                                 {**game, 'GameTimeUTC': '2026-10-20T11:00:00Z'},
                                 {**game, 'GameStatus': 'Postponed'},
                                 {**game, 'GameTimeUTC': '2026-10-20T21:00:00'}])
        games = upcoming_team_games(schedule, now)
        self.assertEqual(len(games), 1)
        self.assertEqual((games[0]['Home'], games[0]['Away']), ('BOS', 'DET'))
        self.assertEqual(upcoming_team_games(pd.DataFrame(), now), [])

    def test_team_games_remain_for_one_hour_after_tip_including_in_progress(self):
        start = pd.Timestamp('2026-10-20T23:00:00Z')
        schedule = pd.DataFrame([{'TEAM_ID':TEAM_IDS['NYK'],'Opponent':'PHI','Venue':'home',
                                  'GameTimeUTC':start,'GameStatus':'In Progress'}])
        for now in [start-pd.Timedelta(minutes=1),start,start+pd.Timedelta(minutes=59,seconds=59),
                    pd.Timestamp('2026-10-20T17:59:59-06:00')]:
            self.assertEqual(len(upcoming_team_games(schedule,now)),1)
        for now in [start+pd.Timedelta(hours=1),start+pd.Timedelta(hours=2)]:
            self.assertEqual(upcoming_team_games(schedule,now),[])
        for status in ['Postponed','Canceled','Suspended','Delayed','Final']:
            self.assertEqual(upcoming_team_games(schedule.assign(GameStatus=status),start),[])


if __name__ == '__main__':
    unittest.main()
