from datetime import datetime, timedelta, timezone
import unittest

import numpy as np
import pandas as pd

from points_probability import PROBABILITY_COLUMNS
from points_tracking import evaluation_rows, grade_records, history_table, snapshot
from prop_distributions import COUNT_STATS, count_profile, distribution_cdf, prop_quote


class CountPropTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(PLAYER_ID=203507, Player='Example', Team='MIL', Opponent='NYK')
        self.row['Projected minutes'] = 28.
        self.now = datetime.now(timezone.utc)
        self.event = {'id':'game','commence_time':(self.now + timedelta(hours=3)).isoformat()}

    def quote(self, stat, mu=2.3, line=2.):
        return {'Stat':stat, 'Projected':mu, 'Book line':line, 'Market status':'Current',
                'Over odds':-110., 'Under odds':-110., 'Updated UTC':self.now.isoformat()}

    def test_discrete_probabilities_match_the_original_mean_and_integer_push(self):
        for stat in COUNT_STATS:
            for mean in [0, .18, 2.8, 5.9] if stat=='PF' else [0, .18, 2.8, 28.9]:
                with self.subTest(stat=stat, mean=mean):
                    half = prop_quote(self.row, self.quote(stat,mean,2.5))
                    whole = prop_quote(self.row, self.quote(stat,mean,3))
                    self.assertEqual(half['Exp. P(push)'],0.)
                    self.assertAlmostEqual(sum(whole[c] for c in PROBABILITY_COLUMNS),1.)
                    self.assertAlmostEqual(half['Exp. P(over)'],whole['Exp. P(over)']+whole['Exp. P(push)'])
                    cdf = distribution_cdf(np.arange(1500), mean, whole['Dispersion'], whole['Distribution'], stat)
                    self.assertTrue((np.diff(cdf) >= -1e-12).all())
                    self.assertAlmostEqual(sum(1-cdf), mean, places=7)

    def test_quote_gates_and_new_player_fallback(self):
        q = self.quote('REB')
        for row, quote in [(self.row, {**q,'Market status':'Stale quote'}), (self.row, {**q,'Book line':1.2}),
                           (self.row, {**q,'Projected':np.nan}), ({**self.row,'Projected minutes':0},q),
                           ({**self.row,'Estimated stat profile':True},q)]:
            self.assertTrue(all(np.isnan(prop_quote(row,quote)[c]) for c in PROBABILITY_COLUMNS))
        self.assertEqual(prop_quote({**self.row,'PLAYER_ID':99999999}, self.quote('PRA'))['Variability source'], 'Stat pool')
        self.assertNotEqual(prop_quote(self.row, self.quote('PRA'))['Dispersion'], count_profile()['stats']['PRA']['pool'])
        manual = {**self.quote('PF',2.1,6),'Market status':'Manual line'}
        self.assertEqual(prop_quote(self.row,manual)['Exp. P(over)'],0.)
        self.assertGreater(prop_quote(self.row,manual)['Exp. P(push)'],0.)
        self.assertTrue(np.isnan(prop_quote(self.row,{**manual,'Projected':6.1})['Exp. P(over)']))

    def test_tracking_grades_each_stat_and_keeps_distinct_snapshots(self):
        records = []
        for stat in COUNT_STATS:
            if stat == 'PF':
                continue  # Manual lines are not sportsbook comparisons.
            quote = self.quote(stat)
            p = prop_quote(self.row,quote)
            record = snapshot({**self.row, **quote, **p},self.event,'fanduel',
                              {203507:{'ESPN_PLAYER_ID':'99'}},'test',now=self.now)
            self.assertIsNotNone(record)
            records.append(record)
        self.assertEqual(len({r['id'] for r in records}),len(records))
        logs = pd.DataFrame([dict(ESPN_PLAYER_ID='99',GAME_ID='123',GAME_DATE=self.event['commence_time'],
                            Team='MIL',Opponent='NYK',MIN=27,PTS=12,REB=3,AST=2,FG3M=1,TOV=0,BLK=1,STL=2)])
        table = evaluation_rows(history_table(records,grade_records(records,logs))).set_index('stat')
        self.assertEqual(table.actual.to_dict(), dict(REB=3,AST=2,PRA=17,STL=2,BLK=1,TOV=0,**{'3PM':1}))
        self.assertEqual(table.loc['STL','outcome'],'Push')
        self.assertTrue(table.actual_points.isna().all())
        self.assertFalse(grade_records(records,logs.assign(MIN=0)))
        missing = grade_records(records,logs.drop(columns='REB'))
        self.assertEqual(len(missing),len(records)-2)
