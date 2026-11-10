from datetime import date
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from requests import Timeout

from assist_opportunities import (DailyPassingFeed, SAMPLE, adjusted_assist_samples, apply_assist_sample,
                                  bundled, verified_days)
from live_projections import STAT_NAMES, build_projections
from team_assists import apply_team_assists


class AssistOpportunityTests(TestCase):
    def test_formula_matches_separate_opportunity_estimates_with_partial_coverage(self):
        prior = pd.DataFrame({'AST':[60,80], 'MIN':[400,800], 'POTENTIAL_AST':[100,200], 'PASSES_MADE':[500,1000]},
                             index=[1,2])
        seed = pd.DataFrame([[5,15,100,4,12,40,220], [3,4,60,3,4,0,70]],
                            columns=SAMPLE, index=[1,3])
        with patch('assist_opportunities.bundled', return_value=(prior,seed)):
            profiles, missing = adjusted_assist_samples('2026-27',pd.DataFrame(),{},Mock())
        self.assertEqual(missing,[])
        # Player 1: two independent posterior yields using last year's opportunity pace.
        potential_prior = (60+25*140/300)/(100+25)
        pass_prior = (60+150*140/1500)/(500+150)
        potential = 40*(12+50*potential_prior)/(40+50)
        passes = 220*(12+250*pass_prior)/(220+250)
        # Three credits from the untracked game remain in the numerator.
        observed = pd.Series({'AST':3.,'GP':5.,'MIN':20.})
        rate, used = apply_assist_sample(profiles[1],observed,3/20)
        self.assertTrue(used)
        self.assertAlmostEqual(rate,(3+.5*potential+.5*passes)/100)
        # A newcomer with no potential chances retains actuals for that component.
        pass_exposure = 200*1500/1200
        rookie = .5*(4+70*(4+pass_exposure*140/1500)/(70+pass_exposure))/60
        self.assertAlmostEqual(apply_assist_sample(profiles[3],pd.Series({'GP':3,'MIN':20,'AST':4/3}),4/60)[0],rookie)
        self.assertEqual(apply_assist_sample(profiles[1],pd.Series({'GP':2,'MIN':20,'AST':3.}),.15),(.15,False))
        self.assertEqual(apply_assist_sample(profiles[1],pd.Series({'GP':5,'MIN':20,'AST':0.}),0.),(0.,False))

    def test_completed_days_only_no_overlap_and_feed_retry(self):
        logs = pd.DataFrame([
            ['guard','2026-11-10T01:00:00Z',30,5],  # November 9 Eastern is already bundled.
            ['guard','2026-11-11T01:00:00Z',30,4],
            ['wing','2026-11-11T01:00:00Z',20,2],
        ], columns=['ESPN_PLAYER_ID','GAME_DATE','MIN','AST'])
        day = date(2026,11,10)
        tracking = pd.DataFrame({'GP':[1,1], 'MIN':[30,20], 'AST':[4,3],
                                 'POTENTIAL_AST':[9,6], 'PASSES_MADE':[50,40]}, index=[1,2])
        fetch = Mock(side_effect=[tracking, Timeout()])
        feed = DailyPassingFeed(fetch)
        self.addCleanup(feed.pool.shutdown)
        args = ('2026-27', logs, {'guard':1,'wing':2})
        windows, missing = verified_days(*args, feed.read)
        self.assertEqual(missing,[])
        self.assertEqual(windows[0].index.tolist(),[1])
        fetch.assert_called_once_with('2026-27',day)
        feed.refresh_at['2026-27',day] = 0
        again,_ = verified_days(*args,feed.read)
        pd.testing.assert_frame_equal(windows[0],again[0])
        self.assertEqual(fetch.call_count,2)
        self.assertEqual(verified_days(*args,lambda *_:{})[1],[day])

    def test_new_base_flows_through_minutes_archetype_normalization_and_pra(self):
        roster = pd.DataFrame([dict(PLAYER_ID=i, PLAYER_NAME=n, TEAM_ID=1, TEAM_ABBREVIATION='DET',
                                    PROJECTED_MIN=30.,OffArchetype='Playmaker', DefArchetype='Wing')
                               for i,n in [(1,'Guard'),(2,'Wing')]])
        teams = pd.DataFrame([{'TEAM_ID':1,'GAMES_PLAYED':10}])
        observed = roster[['PLAYER_NAME']].assign(ESPN_PLAYER_ID=['a','b'], GP=10.,MIN=30.,
                                                  **{c:3. for c in STAT_NAMES})
        history = roster[['PLAYER_ID']].assign(MIN=30.,**{c:3. for c in STAT_NAMES})
        sample = {1:dict(tracked_gp=8,box_gp=8,box_min=240,box_ast=24,tracked_ast=24,delta=6.)}
        for minutes in [{}, {1:37.}, {1:0.}]:
            args = dict(approved_minutes=minutes,career_profiles={},prior_season_minutes={})
            old = build_projections(roster,history,teams,observed,**args)
            new = build_projections(roster,history,teams,observed,assist_samples=sample,**args)
            for stat in set(STAT_NAMES)-{'AST'}:
                np.testing.assert_equal(old[stat].values,new[stat].values)
            w = 200*(20-10)/12
            self.assertAlmostEqual((new.AST-old.AST)[0],6/(300+w)*new['Projected minutes'][0])
            factors = np.array([.8,1.1])
            rates = pd.DataFrame({'Assist rate':[.6,.5], 'Allowed assist rate':[.55,.62],
                                  'Points per FGM':[2.5,2.4], 'Allowed points per FGM':[2.5,2.4]}, index=['DET','BOS'])
            for rows in (old,new):
                before = rows.assign(AST=rows.AST*factors)
                final,_,issues = apply_team_assists(before,{'DET':105.},rates,{'DET':'BOS'})
                self.assertFalse(issues)
                self.assertAlmostEqual(final.AST.sum(),105/(2.45)*.61)
                np.testing.assert_allclose(final.AST/final.AST.sum(), before.AST/before.AST.sum())
                np.testing.assert_allclose(final.PRA, final.PTS+final.AST+final.REB)

    def test_launch_sample_totals(self):
        prior, rows = bundled('2026-27')
        self.assertEqual(len(prior),582)
        self.assertEqual(rows.box_gp.sum(),3479)
        self.assertEqual(rows.tracked_gp.sum(),3450)
