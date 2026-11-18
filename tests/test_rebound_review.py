import unittest

import numpy as np
import pandas as pd

from edge_review import assess_edges, role_evidence, unavailable_review
from edge_rating_table import role_popup
from test_edge_review import NOW, GAME, logs, raw_team, finalized, edge, reference, healthy


class ReboundReviewTests(unittest.TestCase):
    def setUp(self):
        self.team = finalized(raw_team())
        self.observed = pd.Series(dict(ESPN_PLAYER_ID='11', GP=10))

    def sample(self, before=10., recent=8.):
        return logs().assign(MIN=30., REB=[before]*5+[recent]*5)

    def review(self, before=10., recent=8., side='Over', implied=6.):
        original = edge(float(self.team.REB.iloc[0]), implied, side, 'REB')
        saved = original.copy(deep=True)
        team_before = self.team.copy(deep=True)
        result = assess_edges(original, self.team, GAME, self.sample(before,recent), {1:self.observed},
                              reference(), healthy(), NOW, NOW)
        # Review must not change means, probabilities, prices, EV, ranks or inputs.
        pd.testing.assert_frame_equal(result[original.columns], saved)
        pd.testing.assert_frame_equal(original, saved)
        pd.testing.assert_frame_equal(self.team, team_before)
        return result.iloc[0]

    def test_rebounds_use_total_counts_over_minutes_without_shooting_tracking(self):
        sample = self.sample().drop(columns=['FGA','FGM','FTA','FTM','FG3A','FG3M'])
        sample.loc[:4,'MIN']=[10.,20.,30.,40.,50.]
        sample.loc[:4,'REB']=[1.,8.,3.,16.,5.]
        signal=role_evidence(self.team.iloc[0],self.observed,sample,NOW)['rebound']
        self.assertAlmostEqual(signal['previous_rate'],33/150)
        self.assertAlmostEqual(signal['rate'],40/150)
        self.assertAlmostEqual(signal['change'],40/33-1)
        longer=pd.concat([self.sample(),self.sample().iloc[:5]],ignore_index=True)
        longer['GAME_ID']=[str(i) for i in range(1,16)]
        longer['GAME_DATE']=[f'2026-11-{i:02d}T12:00:00Z' for i in range(1,16)]
        longer['REB']=[10.]*10+[6.]*5
        result=role_evidence(self.team.iloc[0],self.observed,longer,NOW)
        self.assertNotIn('rebound',result)  # Ten-game summary cannot validate fifteen boxes.
        observed=pd.Series(dict(ESPN_PLAYER_ID='11',GP=15))
        result=role_evidence(self.team.iloc[0],observed,longer,NOW)
        self.assertIn('5 recent vs 10 preceding',result['rebound_detail'])
        self.assertAlmostEqual(result['rebound']['change'],-.4)

    def test_twenty_percent_boundary_and_both_bet_directions(self):
        for side, recent, expected in [('Over',8.01,'Clear'),('Over',8.,'ROLE'),('Over',7.99,'ROLE'),
                                        ('Under',11.99,'Clear'),('Under',12.,'ROLE'),('Under',12.01,'ROLE'),
                                        ('Over',12.,'Clear'),('Under',8.,'Clear')]:
            with self.subTest(side=side,recent=recent):
                row=self.review(recent=recent,side=side,implied=15. if side=='Under' else 6.)
                self.assertEqual(row['Role flag'],expected)
                self.assertAlmostEqual(row['Recent test projection'],recent)
                expected_edge=(recent/(15. if side=='Under' else 6.)-1)*100*(1 if side=='Over' else -1)
                self.assertAlmostEqual(row['Recent test edge %'],expected_edge)
                self.assertAlmostEqual(row['Combined test projection'],recent*(1.1 if side=='Under' else .9))
                if expected=='ROLE': self.assertFalse(row['Robust play'])

    def test_popup_reports_recent_rate_and_signed_remaining_edge(self):
        row=self.review(recent=0.)
        self.assertEqual(row['Recent test projection'],0.)
        self.assertEqual(row['Recent test edge %'],-100.)
        popup=role_popup(row)
        self.assertIn('down 100.0% per minute',popup['detail'])
        self.assertIn('0.0 rebounds',popup['detail'])
        self.assertIn('-100.0% Over edge no longer clears',popup['detail'])
        self.assertIn('5 recent vs 5 preceding',popup['technical'])
        self.assertNotIn('Passing-opportunity',popup['technical'])
        under=role_popup(self.review(recent=12.,side='Under',implied=15.))
        self.assertIn('up 20.0% per minute',under['detail'])
        self.assertIn('20.0% Under edge still clears',under['detail'])

    def test_rebounds_can_pass_robust_filter_when_all_checks_pass(self):
        value=float(self.team.REB.iloc[0])
        row=self.review(before=value,recent=value,implied=value/2)
        self.assertEqual(row['Role flag'],'Clear')
        self.assertTrue(row['Robust play'])
        # A sub-20% trend is unflagged, but losing the 5% edge still fails robust.
        row=self.review(before=10.,recent=9.,implied=9.)
        self.assertEqual(row['Role flag'],'Clear')
        self.assertEqual(row['Recent test edge %'],0.)
        self.assertFalse(row['Robust play'])

    def test_missing_history_zero_denominator_and_corrupt_records_stay_unchecked(self):
        cases=[self.sample().iloc[:-1],self.sample(before=0.),self.sample().drop(columns='REB')]
        for column,value in [('REB',np.nan),('REB',-1.),('MIN',np.inf)]:
            bad=self.sample();bad.loc[7,column]=value;cases.append(bad)
        bad=self.sample();bad.loc[0,'Team']='LAL';cases.append(bad)
        cases.append(pd.concat([self.sample(),self.sample().iloc[:1]]))
        short=self.sample();short.loc[5:,'MIN']=8.;cases.append(short)
        for sample in cases:
            with self.subTest(case=len(sample)):
                result=assess_edges(edge(5.,3.,'Over','REB'),self.team,GAME,sample,{1:self.observed},
                                    reference(),healthy(),NOW,NOW).iloc[0]
                self.assertEqual(result['Role flag'],'Unchecked')
                self.assertTrue(pd.isna(result['Recent test projection']))
                self.assertFalse(result['Robust play'])
                self.assertNotIn('aren’t available yet',role_popup(result)['detail'])

    def test_other_stats_keep_the_same_scenarios_and_future_games_are_excluded(self):
        before=logs();after=self.sample()
        future=after.iloc[-1:].assign(GAME_ID='future',GAME_DATE='2026-11-18T12:00:00Z',REB=99.)
        signal=role_evidence(self.team.iloc[0],self.observed,pd.concat([after,future]),NOW)
        self.assertAlmostEqual(signal['rebound']['rate'],8/30)
        for stat in ['PTS','3PM','AST','PRA','STL','BLK']:
            e=edge(stat=stat)
            a=assess_edges(e,self.team,GAME,before,{1:self.observed},reference(),healthy(),NOW,NOW)
            b=assess_edges(e,self.team,GAME,after,{1:self.observed},reference(),healthy(),NOW,NOW)
            pd.testing.assert_frame_equal(a.drop(columns='Review captured UTC'),b.drop(columns='Review captured UTC'))
        self.assertIn('Rebound trend %',unavailable_review(edge(stat='REB')))


if __name__=='__main__':unittest.main()
