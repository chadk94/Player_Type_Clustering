import copy
import unittest

import numpy as np
import pandas as pd

from career_shooting import blended_accuracy, apply_accuracy, load_career_profiles, parse_career_totals
from live_projections import build_projections, parse_player_summary, STAT_NAMES
from team_threes import apply_team_threes
from import_minutes import TEAM_IDS


class CareerShootingTests(unittest.TestCase):
    def test_floor_zero_attempts_and_large_samples(self):
        for attempts in [0, 1, 36, 199, 200, 201, 500, 1000, 10000]:
            pct, weight = blended_accuracy(400, 1000, attempts*.3, attempts)
            self.assertGreaterEqual(weight, .5)
            self.assertLessEqual(weight, 1.)
            self.assertAlmostEqual(pct, weight*.4+(1-weight)*.3 if attempts else .4)
            if attempts >= 200:
                self.assertEqual(weight, .5)
        self.assertEqual(blended_accuracy(0, 0, 10, 36), (None, None))
        self.assertLess(blended_accuracy(400, 1000, 0, 36)[0], .4)
        for args in [(-1,100,0,0),(101,100,0,0),(40,100,5,4),(40,100,np.nan,10)]:
            with self.assertRaises(ValueError): blended_accuracy(*args)

    def test_parser_scopes_seasons_and_avoids_trade_double_counting(self):
        def item(year, team, value):
            return dict(season={'year':year}, teamId=team, stats=[value])
        payload = {'filters':[{'name':'league','value':'nba'}, {'name':'seasontype','value':'2'}],
            'categories':[{'name':'totals','names':['threePointFieldGoalsMade-threePointFieldGoalsAttempted'],
                'statistics':[item(2025,'1','2-5'), item(2025,'2','3-5'), item(2025,'0','5-10'),
                              item(2026,'1','20-50'),item(2026,'2','30-50'),item(2027,'1','9-10')]}]}
        result = parse_career_totals(payload,'2026-27')
        self.assertEqual((result['FG3M'],result['FG3A']),(55,110))
        combined = payload['categories'][0]['statistics'][2]
        combined.pop('teamId');combined['displayName']='2024-25 Totals'
        self.assertEqual(parse_career_totals(payload,'2026-27')['FG3A'],110)
        payload['filters'][1]['value']='3'
        with self.assertRaises(ValueError): parse_career_totals(payload,'2026-27')

    def test_no_career_and_missing_current_preserve_available_data(self):
        original={'FG3M':1.,'FG3A':4.,'PTS':12.,'PRA':20.,'Projected minutes':30.}
        for career in [None,{'FG3M':0,'FG3A':0}]:
            row=original.copy();apply_accuracy(row,career,None)
            self.assertEqual(row['FG3M'],1.)
        row=original.copy();apply_accuracy(row,{'FG3M':400,'FG3A':1000},None)
        self.assertEqual(row['FG3M'],1.6)
        self.assertEqual(row['Career 3P weight'],1.)
        row=original.copy();apply_accuracy(row,{'FG3M':400,'FG3A':1000},pd.Series({'GP':1,'FG3M':np.nan,'FG3A':np.nan}))
        self.assertEqual(row['FG3M'],1.6)
        self.assertIn('unavailable',row['3P accuracy source'])

    def test_shared_pipeline_preserves_minutes_volume_and_other_stats(self):
        roster=pd.DataFrame([dict(PLAYER_ID=1,PLAYER_NAME='Example',TEAM_ID=TEAM_IDS['ORL'],
            TEAM_ABBREVIATION='ORL',PROJECTED_MIN=34.,OffArchetype='High Usage Wing',DefArchetype='Wing')])
        history=pd.DataFrame([dict(PLAYER_ID=1,MIN=34.,**{s:2. for s in STAT_NAMES})])
        history.loc[0,['PTS','FG3A']]=[20.,6.]
        current=pd.DataFrame([dict(PLAYER_NAME='Example',GP=4.,MIN=32.25,**{s:2. for s in STAT_NAMES})])
        current.loc[0,['PTS','FG3M','FG3A']]=[14.5,2.5,9.]
        current['Season FG3M total']=10.;current['Season FG3A total']=36.
        career={1:{'FG3M':979,'FG3A':2406}}
        for games in [1,4,82]:
            for minutes in [0.,18.25,34.]:
                kw=dict(roster=roster,history=history,teams=pd.DataFrame([dict(TEAM_ID=TEAM_IDS['ORL'],GAMES_PLAYED=games)]),
                        current=current,approved_minutes={1:minutes})
                before=build_projections(**kw,career_profiles={})
                after=build_projections(**kw,career_profiles=career)
                pd.testing.assert_frame_equal(before[[s for s in STAT_NAMES if s!='FG3M']],
                                              after[[s for s in STAT_NAMES if s!='FG3M']])
                self.assertEqual(after.iloc[0]['Projected minutes'],minutes)
                self.assertEqual(after.iloc[0].PRA,before.iloc[0].PRA)
                self.assertAlmostEqual(after.iloc[0].FG3M,after.iloc[0].FG3A*blended_accuracy(979,2406,10,36)[0])

    def test_normalized_archetype_attempt_shares_and_team_target_unchanged(self):
        # Different archetype effects have already been applied to attempts.
        frame=pd.DataFrame({'Team':['ORL','ORL'],'Opponent':['TOR','TOR'],
            'Projected minutes':[34.,30.],'PTS':[20.,20.],'PRA':[30.,30.],
            'FG3A':[6.*.8,5.*1.1],'FG3M':[6.*.8*.3,5.*1.1*.4]})
        changed=frame.copy();changed.loc[0,'FG3M']=changed.loc[0,'FG3A']*blended_accuracy(979,2406,10,36)[0]
        rates=pd.DataFrame({'Three attempt rate':[.4,.4],'Allowed three attempt rate':[.4,.4],
                            'Points per FGA':[1.1,1.1],'Allowed points per FGA':[1.1,1.1]},index=['ORL','TOR'])
        old,_,issues=apply_team_threes(frame,{'ORL':30.},rates,{'ORL':'TOR'})
        new,summary,new_issues=apply_team_threes(changed,{'ORL':30.},rates,{'ORL':'TOR'})
        self.assertEqual(issues,[]);self.assertEqual(new_issues,[])
        for col in ['FG3A','Three-point share','Team 3PA target','PTS','PRA']:
            pd.testing.assert_series_equal(old[col],new[col])
        self.assertAlmostEqual(new.FG3M.iloc[0]/new.FG3A.iloc[0],changed.FG3M.iloc[0]/changed.FG3A.iloc[0])
        self.assertAlmostEqual(new.FG3M.sum(),summary.iloc[0]['Team 3PM target'])

    def test_published_bane_baseline_matches_approved_preview(self):
        career=load_career_profiles('2026-27')[1630217]
        self.assertEqual((career['FG3M'],career['FG3A']),(979,2406))
        pct,weight=blended_accuracy(979,2406,10,36)
        self.assertAlmostEqual(6.275836218846566*pct,2.430021963233875,places=10)
        self.assertAlmostEqual(weight,200/236)


if __name__=='__main__': unittest.main()
