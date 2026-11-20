from datetime import datetime,timezone,timedelta
import unittest
from unittest.mock import patch
import requests
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sportsbook_lines import (american_probability,fair_over_probability,implied_total,
    next_matchup,player_quotes,game_quotes,compare_with_book,comparison_style,read_feed,OddsFeedError)


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

    def test_herb_jones_quote_matches_herbert_profile_without_changing_identity(self):
        for book_name,model_name in [('Herb Jones','Herbert Jones'),('Herbert Jones','Herb Jones')]:
            with self.subTest(book_name=book_name):
                self.market['outcomes']=[dict(description=book_name,name=side,point=1.5,price=price)
                    for side,price in [('Over',144),('Under',-192)]]
                quotes=game_quotes(self.payload,[model_name,'Other Jones'],'fanduel',now=self.now)
                self.assertEqual(set(quotes),{model_name.lower(),'other jones'})
                q=quotes[model_name.lower()]
                row=q.set_index('Stat').loc['3PM']
                self.assertEqual(row['Market status'],'Current')
                self.assertEqual(row['Over odds'],144)
                result=compare_with_book(pd.DataFrame(dict(Stat=['3PM'],Projected=[1.8])),q).iloc[0]
                self.assertEqual(result.Play,'Over')
                self.assertGreater(result['Edge %'],30)
                self.assertTrue(quotes['other jones']['Book implied total'].isna().all())

    def test_alias_does_not_merge_ambiguous_profiles_or_suffixes(self):
        for outcome in self.market['outcomes']:outcome['description']='Herb Jones'
        ambiguous=game_quotes(self.payload,['Herb Jones','Herbert Jones'],'fanduel',now=self.now)
        self.assertTrue(all(q['Book implied total'].isna().all() for q in ambiguous.values()))
        suffixed=player_quotes(self.payload,'Herbert Jones Jr.','fanduel',now=self.now)
        self.assertTrue(suffixed['Book implied total'].isna().all())

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

    def test_scheduled_opponent_never_uses_prices_from_later_rematch(self):
        tip = datetime(2026,10,20,23,tzinfo=timezone.utc)
        event = {'id':'today','home_team':'New York Knicks','away_team':'Philadelphia 76ers',
                 'commence_time':(tip+timedelta(minutes=9)).isoformat()}
        later = {**event,'id':'rematch','commence_time':'2026-10-30T23:40:00Z'}
        events = [later,event]
        self.assertEqual(next_matchup(events,'PHI','NYK',tip-timedelta(minutes=1),scheduled_start=tip)['id'],'today')
        self.assertIsNone(next_matchup(events,'PHI','NYK',tip+timedelta(minutes=20),scheduled_start=tip))
        self.assertIsNone(next_matchup([later],'PHI','NYK',tip-timedelta(minutes=1),scheduled_start=tip))
        # A manually chosen opponent without a schedule anchor can still preview
        # its next upcoming fixture, with that game's date shown in the app.
        self.assertEqual(next_matchup([later],'PHI','NYK',tip)['id'],'rematch')

    def test_schedule_match_shares_thirty_minute_window(self):
        tip = pd.Timestamp('2026-11-19T02:30:00Z')
        event = dict(id='bos-den', home_team='Denver Nuggets', away_team='Boston Celtics')
        for offset in [-30, -20, 20, 30]:
            quote = {**event, 'commence_time':(tip+pd.Timedelta(minutes=offset)).isoformat()}
            self.assertEqual(next_matchup([quote], 'BOS', 'DEN', self.now, scheduled_start=tip)['id'], 'bos-den')
        for offset in [-31, 31, 1440]:
            quote = {**event, 'commence_time':(tip+pd.Timedelta(minutes=offset)).isoformat()}
            self.assertIsNone(next_matchup([quote], 'BOS', 'DEN', self.now, scheduled_start=tip))

    def test_colors_compare_projection_to_implied_not_posted_line(self):
        quotes=player_quotes(self.payload,'Jalen Brunson','fanduel',now=self.now)
        preview=pd.DataFrame({'Stat':['3PM','TOV'],'Projected':[2.7,1.]})
        result=compare_with_book(preview,quotes)
        # 2.7 exceeds posted 2.5, but is below the juice-adjusted implied mean.
        self.assertGreater(result.iloc[0]['Edge %'],0)
        self.assertLess(result.iloc[0]['Signed edge %'],0)
        self.assertEqual(result.iloc[0]['Play'],'Under')
        self.assertTrue(any('background-color' in style for style in comparison_style(result.iloc[0])))
        self.assertEqual(comparison_style(result.iloc[0])[1],'')
        self.assertTrue(pd.isna(result.iloc[1]['Edge %']))
        self.assertTrue(all(style=='' for style in comparison_style(result.iloc[1])))
    def test_percent_edge_and_strict_five_percent_play_buffer(self):
        preview=pd.DataFrame({'Stat':['PTS','REB','AST','3PM','STL','BLK','TOV','PRA'],
                              'Projected':[105.,105.01,95.,94.99,100.,125.,75.,np.nan]})
        quotes=pd.DataFrame({'Stat':preview.Stat,'Book implied total':[100.]*8})
        result=compare_with_book(preview,quotes)
        self.assertEqual(result['Edge %'].iloc[:7].tolist(),[5.,5.01,5.,5.01,0.,25.,25.])
        self.assertEqual(result['Signed edge %'].iloc[:7].tolist(),[5.,5.01,-5.,-5.01,0.,25.,-25.])
        self.assertEqual(result.Play.tolist(),['','Over','','Under','','Over','Under',''])
        for index in [0,2,4,7]:
            self.assertTrue(all(style=='' for style in comparison_style(result.iloc[index])))
        weak=comparison_style(result.iloc[1])
        strong=comparison_style(result.iloc[5])
        self.assertNotEqual(weak,strong)
        self.assertTrue(all(not style for col,style in zip(result.columns,strong) if col not in ['Edge %','Play']))
        quotes['Book implied total']=0.
        unavailable=compare_with_book(preview,quotes)
        self.assertTrue(unavailable['Edge %'].isna().all())
        self.assertTrue(unavailable.Play.eq('').all())

    def test_steal_and_block_unders_use_the_under_price_inclusively(self):
        preview = pd.DataFrame({'Stat': ['STL', 'BLK', 'REB'], 'Projected': [1., 1., 1.]})
        for price in [-250, -200, -161, -160, -150, 110]:
            quotes = pd.DataFrame({'Stat': preview.Stat, 'Book implied total': 1.5,
                                   'Under odds': price, 'Over odds': -200})
            compared = compare_with_book(preview, quotes)
            for _, row in compared.iloc[:2].iterrows():
                self.assertEqual(row.Play, 'Under' if price >= -160 else '')
                self.assertEqual(pd.notna(row['Edge %']), price >= -160)
                self.assertEqual(any(comparison_style(row)), price >= -160)
                self.assertEqual(row.Projected, 1.)
                self.assertEqual(row['Under odds'], price)
            self.assertEqual(compared.iloc[2].Play, 'Under')
            overs = compare_with_book(preview.assign(Projected=2.), quotes)
            self.assertTrue(overs.Play.eq('Over').all())


if __name__=='__main__':unittest.main()
