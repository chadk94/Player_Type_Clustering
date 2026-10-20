import unittest

import numpy as np
import pandas as pd

from team_rebounds import historical_rebound_rates, game_rebound_targets, apply_team_rebounds


class TeamReboundTests(unittest.TestCase):
    def setUp(self):
        self.rates = pd.DataFrame({
            'Points per FGM': [2.5, 2.5], 'FG misses per FGM': [1., 1.],
            'FT misses per FGM': [.1, .1], 'Player credited fraction': [.8, .8],
            'Offensive rebound share': [.25, .25],
        }, index=['BOS', 'DET'])
        self.opponents = {'BOS': 'DET', 'DET': 'BOS'}
        self.targets = {'BOS': 100., 'DET': 110.}
        self.frame = pd.DataFrame({
            'PLAYER_ID': [1, 2, 3], 'Team': ['BOS']*3, 'Opponent': ['DET']*3,
            'Projected minutes': [30., 25., 0.], 'PTS': [20., 30., 0.],
            'AST': [4., 6., 0.], 'OREB': [2., 3., 0.], 'DREB': [4., 5., 0.],
            'REB': [6., 8., 0.], 'PRA': [30., 44., 0.], 'FG3M': [2., 3., 0.],
            'Offensive archetype': ['Wing', 'Big', 'Guard'],
        })

    def apply(self, frame=None, targets=None, opponents=None):
        return apply_team_rebounds(
            self.frame if frame is None else frame, self.targets if targets is None else targets,
            self.rates, self.rates, self.opponents if opponents is None else opponents)

    def test_pools_conserve_opposing_offensive_and_defensive_rebounds(self):
        targets = game_rebound_targets(self.rates, self.rates, self.opponents, self.targets)
        self.assertAlmostEqual(targets.loc['BOS', 'Player rebound pool'], 35.2)
        self.assertAlmostEqual(targets.loc['BOS', 'Team OREB target'], 8.8)
        self.assertAlmostEqual(targets.loc['DET', 'Team DREB target'], 26.4)
        self.assertAlmostEqual(targets.loc['BOS', 'Team DREB target'], 29.04)
        self.assertAlmostEqual(targets['Team REB target'].sum(), targets['Player rebound pool'].sum())
        for bad in [{'BOS': 100.}, {'BOS': np.nan, 'DET': 110.}]:
            with self.assertRaises(ValueError):
                game_rebound_targets(self.rates, self.rates, self.opponents, bad)
        with self.assertRaises(ValueError):
            game_rebound_targets(self.rates, self.rates, {'BOS': 'DET'}, self.targets)

    def test_one_team_preview_uses_both_game_totals(self):
        result, summary, issues = self.apply()
        self.assertEqual(issues, [])
        self.assertAlmostEqual(result.OREB.sum(), 8.8)
        self.assertAlmostEqual(result.DREB.sum(), 29.04)
        self.assertAlmostEqual(result.REB.sum(), summary.iloc[0]['Team REB target'])
        changed, _, _ = self.apply(targets={'BOS': 100., 'DET': 120.})
        np.testing.assert_allclose(result.OREB, changed.OREB)
        self.assertGreater(changed.DREB.sum(), result.DREB.sum())

    def test_separate_normalization_cancels_common_effects_and_preserves_relative_shares(self):
        original = self.frame.copy(deep=True)
        result, _, _ = self.apply()
        self.assertAlmostEqual(result.OREB.iloc[0]/result.OREB.iloc[1], 2/3)
        self.assertAlmostEqual(result.DREB.iloc[0]/result.DREB.iloc[1], 4/5)
        common = self.frame.copy()
        common.OREB *= .72
        common.DREB *= .85
        common.REB = common.OREB + common.DREB
        common.PRA = common.PTS + common.REB + common.AST
        other, _, _ = self.apply(common)
        np.testing.assert_allclose(result[['OREB', 'DREB', 'REB']], other[['OREB', 'DREB', 'REB']])
        np.testing.assert_allclose(result.PRA - self.frame.PRA, result.REB - self.frame.REB)
        np.testing.assert_allclose(result.REB, result.OREB + result.DREB)
        pd.testing.assert_frame_equal(result[['PTS', 'AST', 'FG3M', 'Projected minutes', 'Offensive archetype']],
                                      original[['PTS', 'AST', 'FG3M', 'Projected minutes', 'Offensive archetype']])
        pd.testing.assert_frame_equal(self.frame, original)
        self.assertEqual(result.REB.iloc[2], 0.)

    def test_minutes_reallocation_changes_shares_without_changing_team_targets(self):
        frame = pd.concat([self.frame.iloc[:2]]*3, ignore_index=True)
        frame['PLAYER_ID'] = range(6)
        frame['Projected minutes'] = 40.
        first, _, _ = self.apply(frame)
        for idx, minutes in [(0, 45.), (1, 35.)]:
            frame.loc[idx, ['OREB', 'DREB']] *= minutes/40.
            frame.loc[idx, 'Projected minutes'] = minutes
        frame.REB = frame.OREB + frame.DREB
        frame.PRA = frame.PTS + frame.REB + frame.AST
        second, _, _ = self.apply(frame)
        self.assertEqual(frame['Projected minutes'].sum(), 240.)
        self.assertAlmostEqual(first.REB.sum(), second.REB.sum())
        self.assertGreater(second.REB.iloc[0], first.REB.iloc[0])
        self.assertLess(second.REB.iloc[1], first.REB.iloc[1])

    def test_missing_inputs_retain_whole_team_but_other_team_can_adjust(self):
        for field, value in [('OREB', np.nan), ('DREB', -1.), ('PTS', np.nan),
                             ('AST', np.nan), ('Projected minutes', np.nan),
                             ('Opponent', 'NYK'), ('REB', 123.)]:
            invalid = self.frame.copy()
            invalid.loc[0, field] = value
            result, summary, issues = self.apply(invalid)
            self.assertTrue(summary.empty)
            self.assertEqual(len(issues), 1)
            pd.testing.assert_frame_equal(result[invalid.columns], invalid)
        result, summary, issues = self.apply(targets={'BOS': 100.})
        self.assertTrue(summary.empty)
        pd.testing.assert_frame_equal(result[self.frame.columns], self.frame)
        other = self.frame.copy()
        other['Team'], other['Opponent'] = 'DET', 'BOS'
        combined = pd.concat([self.frame, other], ignore_index=True)
        combined.loc[0, 'OREB'] = np.nan
        result, summary, issues = self.apply(combined)
        self.assertEqual(summary.Team.tolist(), ['DET'])
        self.assertEqual(len(issues), 1)
        pd.testing.assert_frame_equal(result.loc[:2, combined.columns], combined.loc[:2])

    def test_historical_rates_use_all_appearances_and_pooled_ratios(self):
        logs = pd.DataFrame([
            ['g1', 'BOS', 'DET', 20., 40., 8., 10., 5., 12., 17., 50., 30.],
            ['g1', 'BOS', 'DET', 2., 4., 0., 0., 1., 1., 2., 4., 2.],
            ['g1', 'DET', 'BOS', 18., 36., 8., 10., 4., 16., 20., 50., 30.],
        ], columns=['GAME_ID', 'Team', 'Opponent', 'FGM', 'FGA', 'FTM', 'FTA',
                    'OREB', 'DREB', 'REB', 'PTS', 'MIN'])
        own, allowed = historical_rebound_rates(logs, 1)
        self.assertAlmostEqual(own.loc['BOS', 'Offensive rebound share'], 6/22)
        self.assertAlmostEqual(own.loc['BOS', 'Player credited fraction'], 22/24)
        self.assertAlmostEqual(allowed.loc['DET', 'FG misses per FGM'], 1.)
        for bad in [logs.iloc[:-1], logs.assign(REB=999), logs.assign(FGM=100),
                    logs.assign(FTA=np.nan), logs.assign(Opponent='NYK')]:
            with self.assertRaises(ValueError):
                historical_rebound_rates(bad, 1)
        with self.assertRaises(ValueError):
            historical_rebound_rates(logs, 82)


if __name__ == '__main__':
    unittest.main()
