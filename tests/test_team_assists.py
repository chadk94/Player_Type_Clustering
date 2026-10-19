import unittest
import numpy as np
import pandas as pd

from market_points import apply_market_points
from team_assists import historical_assist_rates, assist_target, apply_team_assists


class TeamAssistTests(unittest.TestCase):
    def setUp(self):
        self.rates = pd.DataFrame({
            'Assist rate': [.60, .64], 'Allowed assist rate': [.62, .66],
            'Points per FGM': [2.5, 2.6], 'Allowed points per FGM': [2.4, 2.5],
        }, index=['BOS', 'DET'])
        self.frame = pd.DataFrame({
            'PLAYER_ID': [1, 2, 3], 'Team': ['BOS']*3, 'Opponent': ['DET']*3,
            'Projected minutes': [30., 25., 0.], 'PTS': [20., 10., 0.],
            'REB': [5., 3., 0.], 'AST': [8., 10., 0.], 'PRA': [33., 23., 0.],
            'FG3M': [2., 1., 0.], 'Offensive archetype': ['Primary', 'Secondary', 'Wing'],
        })

    def calibrate(self, frame=None, targets=None):
        return apply_team_assists(self.frame if frame is None else frame,
                                  {'BOS': 100.} if targets is None else targets,
                                  self.rates, {'BOS': 'DET', 'DET': 'BOS'})

    def test_rates_use_pooled_makes_and_all_appearances(self):
        logs = pd.DataFrame([
            ['g1','BOS','DET',10.,20.,50.,30.],
            ['g1','BOS','DET',2.,5.,12.,2.],  # Include the low-minute player.
            ['g1','DET','BOS',14.,30.,75.,40.],
            ['g2','BOS','DET',22.,40.,110.,40.],
            ['g2','DET','BOS',20.,35.,90.,40.],
        ], columns=['GAME_ID','Team','Opponent','AST','FGM','PTS','MIN'])
        rates = historical_assist_rates(logs, expected_games=2)
        self.assertAlmostEqual(rates.loc['BOS','Assist rate'],34/65)
        self.assertAlmostEqual(rates.loc['DET','Allowed assist rate'],34/65)
        self.assertAlmostEqual(rates.loc['BOS','Points per FGM'],172/65)
        with self.assertRaises(ValueError):
            historical_assist_rates(logs, expected_games=82)
        with self.assertRaises(ValueError):
            historical_assist_rates(logs.iloc[:-1])
        logs.loc[0,'AST'] = np.nan
        with self.assertRaises(ValueError):
            historical_assist_rates(logs)

    def test_target_combines_offense_defense_and_market_points(self):
        target, rate, fgm = assist_target(self.rates,'BOS','DET',100.)
        self.assertAlmostEqual(rate,.63)
        self.assertAlmostEqual(fgm,40.)
        self.assertAlmostEqual(target,25.2)
        with self.assertRaises(ValueError):
            assist_target(self.rates,'BOS','BOS',100.)
        with self.assertRaises(ValueError):
            assist_target(self.rates,'BOS','DET',float('nan'))

    def test_common_defensive_penalty_cancels_but_relative_effects_survive(self):
        # Equal 10-assist baselines, modifiers .8 and 1.0: weighted mean .9.
        result,summary,issues = self.calibrate()
        self.assertEqual(issues,[])
        self.assertAlmostEqual(result.AST.iloc[0]/(25.2/2),.8/.9)
        self.assertAlmostEqual(result.AST.iloc[0]/result.AST.iloc[1],.8)
        self.assertAlmostEqual(result.AST.sum(),25.2)
        common = self.frame.copy()
        common['AST'] *= .72
        other,_,_ = self.calibrate(common)
        np.testing.assert_allclose(result.AST,other.AST)

    def test_minutes_and_point_calibration_flow_into_assists_and_pra(self):
        original = self.frame.copy(deep=True)
        points,_,_ = apply_market_points(self.frame,{'BOS':100.})
        result,_,issues = self.calibrate(points)
        self.assertEqual(issues,[])
        np.testing.assert_allclose(result.PRA,result.PTS+result.REB+result.AST)
        pd.testing.assert_frame_equal(result[['PTS','REB','FG3M','Projected minutes','Offensive archetype']],
                                      points[['PTS','REB','FG3M','Projected minutes','Offensive archetype']])
        pd.testing.assert_frame_equal(original,self.frame)
        # A custom minutes increase changes the selected player's production/share.
        custom = self.frame.copy()
        custom.loc[0,['AST','PTS','REB','Projected minutes']] *= 1.5
        changed,_,_ = self.calibrate(custom)
        self.assertAlmostEqual(changed.AST.sum(),25.2)
        self.assertGreater(changed.AST.iloc[0],result.AST.iloc[0])
        self.assertLess(changed.AST.iloc[1],result.AST.iloc[1])
        self.assertEqual(changed.AST.iloc[2],0.)

    def test_incomplete_team_keeps_existing_assists_while_other_team_adjusts(self):
        other = self.frame.copy()
        other['Team'],other['Opponent'] = 'DET','BOS'
        frame = pd.concat([self.frame,other],ignore_index=True)
        frame.loc[0,'AST'] = np.nan
        result,summary,issues = self.calibrate(frame,{'BOS':100.,'DET':110.})
        self.assertEqual(summary.Team.tolist(),['DET'])
        self.assertEqual(len(issues),1)
        pd.testing.assert_series_equal(result.loc[:2,'AST'],frame.loc[:2,'AST'])
        pd.testing.assert_series_equal(result.loc[:2,'PRA'],frame.loc[:2,'PRA'])
        for field,value in [('AST',np.nan),('AST',-1.),('Projected minutes',np.nan),
                            ('Projected minutes',float('inf')),('Opponent','NYK')]:
            invalid = self.frame.copy()
            invalid.loc[0,field] = value
            result,summary,issues = self.calibrate(invalid)
            self.assertTrue(summary.empty)
            self.assertEqual(len(issues),1)
            pd.testing.assert_series_equal(result.AST,invalid.AST)
        result,summary,issues = self.calibrate(targets={})
        self.assertTrue(summary.empty)
        pd.testing.assert_series_equal(result.AST,self.frame.AST)


if __name__ == '__main__':
    unittest.main()
