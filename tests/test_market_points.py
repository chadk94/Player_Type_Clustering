import unittest
import pandas as pd
from market_points import match_points,team_targets,market_targets,apply_market_points

class MarketPointsTests(unittest.TestCase):
    def test_home_favorite_and_underdog_targets(self):
        self.assertEqual(team_targets(220,-6,'BOS','PHI'),{'BOS':113,'PHI':107})
        self.assertEqual(team_targets(220,6,'BOS','PHI'),{'BOS':107,'PHI':113})

    def test_full_points_match_preserves_shares_and_every_other_column(self):
        frame=pd.DataFrame({'Team':['BOS','BOS','PHI','PHI'],'PTS':[60.,50.,55.,45.],
            'Projected minutes':[120.,120.,120.,120.],'PRA':[80.,60.,75.,55.],
            'AST':[10.,5.,8.,7.],'REB':[10.,5.,12.,3.]})
        original=frame.copy(deep=True)
        adjusted,summary=match_points(frame,{'BOS':104.,'PHI':112.})
        self.assertAlmostEqual(adjusted[adjusted.Team.eq('BOS')].PTS.sum(),104)
        self.assertAlmostEqual(adjusted[adjusted.Team.eq('PHI')].PTS.sum(),112)
        self.assertAlmostEqual(adjusted.PTS.iloc[0]/adjusted.PTS.iloc[1],60/50)
        self.assertTrue(adjusted.PRA.eq(adjusted.PTS+adjusted.REB+adjusted.AST).all())
        pd.testing.assert_frame_equal(frame,original)
        pd.testing.assert_frame_equal(adjusted.drop(columns=['PTS','PRA']),frame.drop(columns=['PTS','PRA']))

    def test_missing_coverage_rejected(self):
        frame=pd.DataFrame({'Team':['BOS'],'PTS':[float('nan')],'Projected minutes':[30.]})
        with self.assertRaises(ValueError):match_points(frame,{'BOS':110.})

    def test_quote_validation_and_safe_team_fallback(self):
        import copy
        from datetime import datetime,timezone,timedelta
        now=datetime(2026,10,19,21,tzinfo=timezone.utc)
        event={'id':'bos-det','home_team':'Detroit Pistons','away_team':'Boston Celtics',
               'commence_time':(now+timedelta(days=1)).isoformat()}
        payload={**event,'bookmakers':[{'key':'fanduel','markets':[
            {'key':'spreads','last_update':now.isoformat(),'outcomes':[
                {'name':'Detroit Pistons','point':1.5,'price':-110},
                {'name':'Boston Celtics','point':-1.5,'price':-110}]},
            {'key':'totals','last_update':now.isoformat(),'outcomes':[
                {'name':'Over','point':222.5,'price':-104},
                {'name':'Under','point':222.5,'price':-118}]}]}]}
        targets=market_targets(payload,event,'fanduel',now)
        self.assertEqual(targets,{'DET':110.5,'BOS':112.})
        with self.assertRaises(ValueError):market_targets(payload,event,'draftkings',now)
        with self.assertRaises(ValueError):market_targets(payload,event,'fanduel',now+timedelta(minutes=16))
        with self.assertRaises(ValueError):market_targets({**payload,'id':'wrong'},event,'fanduel',now)
        wrong=copy.deepcopy(payload)
        wrong['bookmakers'][0]['markets'][1]['outcomes'][1]['point']=223.5
        with self.assertRaises(ValueError):market_targets(wrong,event,'fanduel',now)
        frame=pd.DataFrame({'Team':['BOS','DET'],'PTS':[100.,float('nan')],
            'PRA':[130.,125.],'REB':[10.,10.],'AST':[20.,20.],'Projected minutes':[240.,240.]})
        adjusted,summary,issues=apply_market_points(frame,targets)
        self.assertAlmostEqual(adjusted.PTS.iloc[0],112.)
        self.assertTrue(pd.isna(adjusted.PTS.iloc[1]))
        self.assertEqual(summary.Team.tolist(),['BOS'])
        self.assertEqual(len(issues),1)
        self.assertAlmostEqual(adjusted.PRA.iloc[0],142.)
        self.assertEqual(adjusted.PRA.iloc[1],125.)
        self.assertEqual(adjusted.REB.tolist(),frame.REB.tolist())
        self.assertEqual(adjusted.AST.tolist(),frame.AST.tolist())
        self.assertEqual(adjusted['Pre-market PTS'].iloc[0],100.)

if __name__=='__main__':unittest.main()
