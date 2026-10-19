from datetime import datetime,timezone,timedelta
import unittest
from unittest.mock import patch
import requests
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sportsbook_lines import (american_probability,fair_over_probability,implied_total,
    next_matchup,player_quotes,compare_with_book,comparison_style,read_feed,OddsFeedError)


class SportsbookTests(unittest.TestCase):
    def setUp(self):
        self.now=datetime(2026,10,19,19,tzinfo=timezone.utc)
        self.market={'key':'player_threes','last_update':self.now.isoformat(),
                     'outcomes':[{'description':'Jalen Brunson','name':'Over','point':2.5,'price':-150},
                                 {'description':'Jalen Brunson','name':'Under','point':2.5,'price':120}]}
        self.payload={'bookmakers':[{'key':'fanduel','markets':[self.market]}]}

    def test_juice_adjusted_mean_uses_both_sides_without_margin(self):
        self.assertAlmostEqual(american_probability(-150),.6)
        self.assertAlmostEqual(american_probability(120),100/220)
        fair=fair_over_probability(-150,120)
        mean=implied_total(2.5,-150,120)
        self.assertGreater(mean,2.5)
        self.assertAlmostEqual(poisson.sf(2,mean),fair)
        self.assertNotAlmostEqual(fair,.6)
        self.assertLess(implied_total(2.5,150,-180),mean)
        for invalid in [0,50,-50,np.nan]:
            with self.assertRaises(ValueError):american_probability(invalid)

    def test_integer_lines_condition_on_non_push_outcomes(self):
        mean=implied_total(3.,-150,120)
        conditional=poisson.sf(3,mean)/(poisson.sf(3,mean)+poisson.cdf(2,mean))
        self.assertAlmostEqual(conditional,fair_over_probability(-150,120))
        for line in [-1,0,2.2,np.nan]:
            with self.assertRaises(ValueError):implied_total(line,-150,120)

    def test_current_quote_and_missing_categories(self):
        q=player_quotes(self.payload,'Jalen Brunson','fanduel',now=self.now).set_index('Stat')
        self.assertEqual(len(q),8)
        self.assertGreater(q.loc['3PM','Book implied total'],2.5)
        self.assertEqual(q.loc['3PM','Market status'],'Current')
        self.assertTrue(pd.isna(q.loc['TOV','Book implied total']))
        self.assertTrue(player_quotes(self.payload,'Other Player','fanduel',now=self.now)['Book implied total'].isna().all())
        self.assertTrue(player_quotes(self.payload,'Jalen Brunson','draftkings',now=self.now)['Book implied total'].isna().all())

    def test_stale_one_sided_mismatched_and_duplicate_prices_not_compared(self):
        self.market['last_update']=(self.now-timedelta(minutes=16)).isoformat()
        q=player_quotes(self.payload,'Jalen Brunson','fanduel',now=self.now)
        self.assertTrue(q['Book implied total'].isna().all())
        self.market['last_update']=self.now.isoformat()
        self.market['outcomes'][1]['point']=3.5
        self.assertTrue(player_quotes(self.payload,'Jalen Brunson','fanduel',now=self.now)['Book implied total'].isna().all())
        self.market['outcomes'][1]['point']=2.5
        self.market['outcomes'].append(self.market['outcomes'][0].copy())
        self.assertTrue(player_quotes(self.payload,'Jalen Brunson','fanduel',now=self.now)['Book implied total'].isna().all())

    def test_feed_errors_never_expose_credentials(self):
        with patch('sportsbook_lines.requests.get',side_effect=requests.RequestException('failed ?apiKey=private-test-key')):
            with self.assertRaises(OddsFeedError) as caught:
                read_feed('/events','private-test-key')
        self.assertNotIn('private-test-key',str(caught.exception))

    def test_only_next_matching_future_event(self):
        event={'id':'next','home_team':'New York Knicks','away_team':'Philadelphia 76ers','commence_time':'2026-10-20T23:00:00Z'}
        events=[{**event,'id':'wrong','away_team':'Boston Celtics'},
                {**event,'id':'old','commence_time':'2026-10-18T23:00:00Z'},event,
                {**event,'id':'later','commence_time':'2026-10-25T23:00:00Z'}]
        self.assertEqual(next_matchup(events,'NYK','PHI',self.now)['id'],'next')
        self.assertIsNone(next_matchup(events,'NYK','ATL',self.now))

    def test_colors_compare_projection_to_implied_not_posted_line(self):
        quotes=player_quotes(self.payload,'Jalen Brunson','fanduel',now=self.now)
        preview=pd.DataFrame({'Stat':['3PM','TOV'],'Projected':[2.7,1.]})
        result=compare_with_book(preview,quotes)
        # 2.7 exceeds posted 2.5, but is below the juice-adjusted implied mean.
        self.assertLess(result.iloc[0]['vs implied'],0)
        self.assertIn('#f6d2d6',comparison_style(result.iloc[0])[1])
        self.assertTrue(pd.isna(result.iloc[1]['vs implied']))
        self.assertTrue(all(style=='' for style in comparison_style(result.iloc[1])))


if __name__=='__main__':unittest.main()
