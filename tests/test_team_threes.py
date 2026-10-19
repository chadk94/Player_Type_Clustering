import unittest
import numpy as np
import pandas as pd

from team_threes import historical_three_rates, three_target, apply_team_threes


class TeamThreeTests(unittest.TestCase):
    def setUp(self):
        self.rates = pd.DataFrame({
            'Three rate': [.30,.32], 'Allowed three rate': [.34,.40],
            'Points per FGM': [2.5,2.6], 'Allowed points per FGM': [2.4,2.5],
        }, index=['BOS','DET'])
        self.frame = pd.DataFrame({
            'PLAYER_ID': [1,2,3], 'Team': ['BOS']*3, 'Opponent': ['DET']*3,
            'Projected minutes': [30.,25.,0.], 'PTS': [50.,40.,0.],
            'REB': [5.,3.,0.], 'AST': [8.,10.,0.], 'PRA': [63.,53.,0.],
            'FG3M': [4.,5.,0.], 'FG3A': [10.,15.,0.],
            'Offensive archetype': ['Primary','Wing','Big'],
        })

    def calibrate(self, frame=None, targets=None):
        return apply_team_threes(self.frame if frame is None else frame,
                                 {'BOS':100.} if targets is None else targets,
                                 self.rates, {'BOS':'DET','DET':'BOS'})

    def test_pooled_historical_three_rates_include_low_minute_appearances(self):
        logs = pd.DataFrame([
            ['g1','BOS','DET',5.,20.,50.,30.],
            ['g1','BOS','DET',1.,5.,12.,2.],
            ['g1','DET','BOS',8.,30.,75.,40.],
            ['g2','BOS','DET',12.,40.,110.,40.],
            ['g2','DET','BOS',10.,35.,90.,40.],
        ], columns=['GAME_ID','Team','Opponent','FG3M','FGM','PTS','MIN'])
        rates = historical_three_rates(logs,2)
        self.assertAlmostEqual(rates.loc['BOS','Three rate'],18/65)
        self.assertAlmostEqual(rates.loc['DET','Allowed three rate'],18/65)
        with self.assertRaises(ValueError): historical_three_rates(logs,82)
        with self.assertRaises(ValueError): historical_three_rates(logs.iloc[:-1])
        logs.loc[0,'FG3M'] = 21.
        with self.assertRaises(ValueError): historical_three_rates(logs)

    def test_team_target_matches_field_goals_times_blended_three_share(self):
        target,rate,fgm = three_target(self.rates,'BOS','DET',100.)
        self.assertAlmostEqual(rate,.35)
        self.assertAlmostEqual(fgm,40.)
        self.assertAlmostEqual(target,14.)
        with self.assertRaises(ValueError): three_target(self.rates,'BOS','BOS',100.)
        with self.assertRaises(ValueError): three_target(self.rates,'BOS','DET',float('nan'))

    def test_normalized_archetypes_preserve_relative_effects_and_cancel_shared_penalty(self):
        result,summary,issues = self.calibrate()
        self.assertEqual(issues,[])
        # Equal baseline shares and modifiers .8/1.0 -> weighted mean .9.
        self.assertAlmostEqual(result.FG3M.iloc[0]/7.,.8/.9)
        self.assertAlmostEqual(result.FG3M.sum(),14.)
        self.assertAlmostEqual(result.FG3M.iloc[0]/result.FG3M.iloc[1],.8)
        common = self.frame.copy()
        common[['FG3M','FG3A']] *= .72
        other,_,_ = self.calibrate(common)
        np.testing.assert_allclose(result.FG3M,other.FG3M)
        np.testing.assert_allclose(result.FG3A,other.FG3A)

    def test_shooting_percentage_points_pra_and_overrides_preserved(self):
        original = self.frame.copy(deep=True)
        result,_,issues = self.calibrate()
        self.assertEqual(issues,[])
        active=self.frame['Projected minutes'].gt(0)
        np.testing.assert_allclose(result.loc[active,'FG3M']/result.loc[active,'FG3A'],
                                   self.frame.loc[active,'FG3M']/self.frame.loc[active,'FG3A'])
        pd.testing.assert_frame_equal(result[self.frame.columns].drop(columns=['FG3M','FG3A']),
                                      self.frame.drop(columns=['FG3M','FG3A']))
        pd.testing.assert_frame_equal(self.frame,original)
        self.assertEqual(result.FG3M.iloc[2],0.)

    def test_240_minute_roster_reallocation_recalculates_player_shares(self):
        frame = pd.DataFrame({'Team':['BOS']*6,'Opponent':['DET']*6,
            'Projected minutes':[40.]*6,'PTS':[20.]*6,'PRA':[30.]*6,
            'FG3M':[3.2,2.8,2.4,2.,1.6,1.2],'FG3A':[8.,7.,6.,5.,4.,3.]})
        first,_,issues = self.calibrate(frame)
        self.assertEqual(issues,[])
        updated = frame.copy()
        for idx,new_minutes in [(0,45.),(1,35.)]:
            updated.loc[idx,['FG3M','FG3A']] *= new_minutes/40.
            updated.loc[idx,'Projected minutes'] = new_minutes
        second,_,issues = self.calibrate(updated)
        self.assertEqual(issues,[])
        self.assertEqual(updated['Projected minutes'].sum(),240.)
        self.assertAlmostEqual(first.FG3M.sum(),second.FG3M.sum())
        self.assertGreater(second.FG3M.iloc[0],first.FG3M.iloc[0])
        self.assertLess(second.FG3M.iloc[1],first.FG3M.iloc[1])
        self.assertNotEqual(first['Team 3PM factor'].iloc[0],second['Team 3PM factor'].iloc[0])

    def test_incomplete_or_impossible_team_retains_original_threes(self):
        for field,value in [('FG3M',np.nan),('FG3A',np.nan),('FG3A',1.),
                            ('Projected minutes',np.nan),('Opponent','NYK'),('PTS',1.)]:
            frame = self.frame.copy()
            frame.loc[0,field] = value
            result,summary,issues = self.calibrate(frame)
            self.assertTrue(summary.empty)
            self.assertEqual(len(issues),1)
            pd.testing.assert_series_equal(result.FG3M,frame.FG3M)
            pd.testing.assert_series_equal(result.FG3A,frame.FG3A)
        result,summary,issues = self.calibrate(targets={})
        self.assertTrue(summary.empty)
        pd.testing.assert_series_equal(result.FG3M,self.frame.FG3M)
        other=self.frame.copy()
        other['Team'],other['Opponent']='DET','BOS'
        frame=pd.concat([self.frame,other],ignore_index=True)
        frame.loc[0,'FG3M']=np.nan
        result,summary,issues=self.calibrate(frame,{'BOS':100.,'DET':110.})
        self.assertEqual(summary.Team.tolist(),['DET'])
        self.assertEqual(len(issues),1)
        pd.testing.assert_series_equal(result.loc[:2,'FG3M'],frame.loc[:2,'FG3M'])


if __name__ == '__main__':
    unittest.main()
