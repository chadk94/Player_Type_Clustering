"""Regression: Sam Hauser's FD threes existed only as X+ offers on Nov 21."""
from datetime import datetime, timedelta, timezone
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from scipy.stats import poisson

from sportsbook_lines import (MARKETS, FANDUEL_THREES_OVER_ONLY, american_probability,
    player_quotes, compare_with_book, projection_markets, implied_total)
from sportsbook_edges import game_edges
from saved_sportsbook_lines import SOURCE, SAVED
from todays_plays import edge_pick
from jae_card import card_quotes, DENVER
from edge_rating_table import table_payload


class FanDuelMilestoneTests(TestCase):
    def setUp(self):
        self.now = datetime(2026, 11, 22, 0, 30, 29, tzinfo=timezone.utc)
        self.event = dict(id='b754a1bf84af81d45d4a3a16398daa44', Home='SAC', Away='BOS',
            home_team='Sacramento Kings', away_team='Boston Celtics',
            commence_time='2026-11-22T03:10:00Z')
        self.market = dict(key='player_threes_alternate', last_update=self.now.isoformat(), outcomes=[
            dict(description='Sam Hauser', name='Over', point=line, price=price)
            for line, price in [(.5,-3500),(1.5,-400),(2.5,-128),(3.5,196),(4.5,460),
                                (5.5,900),(6.5,1500),(7.5,3500)]])
        self.payload = dict(self.event, bookmakers=[dict(key='fanduel', markets=[self.market])])
        self.players = pd.DataFrame([dict(PLAYER_ID=1630573, Player='Sam Hauser', Team='BOS',
            Opponent='SAC', FG3M=3.200649, Status='Available', **{'Matchup status':'Available',
            'Projected minutes':28.})])

    def quote(self, **kwargs):
        return player_quotes(self.payload, 'Sam Hauser', 'fanduel', now=kwargs.pop('now', self.now), **kwargs)

    def test_actual_offer_reaches_edges_saves_and_requotes_for_jae(self):
        quote = self.quote().set_index('Stat').loc['3PM']
        self.assertEqual(quote['Quote basis'], FANDUEL_THREES_OVER_ONLY)
        self.assertEqual((quote['Book line'],quote['Over odds']), (2.5,-128))
        self.assertTrue(pd.isna(quote['Under odds']))
        self.assertTrue(pd.isna(quote['Fair over probability']))
        self.assertAlmostEqual(poisson.sf(2,quote['Book implied total']),american_probability(-128))
        self.assertGreater(quote['Book implied total'], implied_total(2.5,-128,100))
        edges, count = game_edges(self.players,self.payload,self.event,'fanduel',now=self.now)
        self.assertEqual(count,1)
        row = edges.iloc[0]
        self.assertEqual(row.Projected,3.200649)
        self.assertEqual(row.Play,'Over')
        self.assertGreater(row['Edge %'],5)
        day = self.now.astimezone(DENVER).date()
        pick = edge_pick(row,self.event,'FanDuel',day)
        self.assertEqual((pick['line'],pick['odds'],pick['side']),(2.5,-128,'Over'))
        card, skipped = card_quotes([dict(pick,status='Confirmed')],Mock(return_value=self.payload),lambda:self.now)
        self.assertEqual(skipped,[])
        self.assertEqual((card[0]['line'],card[0]['odds'],card[0]['worst']),(2.5,-128,-133))
        self.assertEqual(card[0]['alt'],dict(line=3.5,odds=196))
        edges['Book line (Over)'] = '2.5 -128'
        view = table_payload(edges,['Rank','Player','Stat','Play','Edge %','Book line (Over)','Under odds'],
            self.players,{},'BOS',self.event,'FanDuel',day)['rows'][0]
        self.assertEqual(view['labels'][-2:],["2.5 -128 · 3+ threes",'—'])
        self.assertIn('No Under price',view['quoteDetail'])

    def test_never_makes_an_under_or_chooses_a_line_based_on_projection(self):
        for projection in [.2,1.,2.,3.2,8.]:
            q = self.quote()
            compared = compare_with_book(pd.DataFrame(dict(Stat=['3PM'],Projected=[projection])),q).iloc[0]
            self.assertEqual(compared['Book line'],2.5)
            self.assertNotEqual(compared.Play,'Under')
            if projection < compared['Book implied total']:
                self.assertTrue(pd.isna(compared['Edge %']))
        edges,count = game_edges(self.players.assign(FG3M=.5),self.payload,self.event,'fanduel',now=self.now)
        self.assertEqual(count,0)

    def test_main_line_wins_regardless_of_market_order_and_even_when_unusable(self):
        main = dict(key='player_threes',last_update=self.now.isoformat(),outcomes=[
            dict(description='Sam Hauser',name=side,point=3.5,price=price)
            for side,price in [('Over',196),('Under',-250)]])
        for markets in [[main,self.market],[self.market,main]]:
            self.payload['bookmakers'][0]['markets']=markets
            q=self.quote().set_index('Stat').loc['3PM']
            self.assertEqual(q['Book line'],3.5)
            self.assertEqual(q['Book implied total'],implied_total(3.5,196,-250))
        main['outcomes'].pop()
        self.assertTrue(self.quote()['Book implied total'].isna().all())
        self.assertEqual(self.quote().set_index('Stat').loc['3PM','Market status'],'Needs matching over/under prices')

    def test_ambiguous_price_or_even_money_tie_does_not_pick_arbitrarily(self):
        self.market['outcomes'].append(dict(self.market['outcomes'][2],price=-130))
        self.assertTrue(self.quote()['Book implied total'].isna().all())
        self.market['outcomes']=[dict(description='Sam Hauser',name='Over',point=line,price=price)
                                 for line,price in [(2.5,-150),(3.5,150)]]
        self.assertTrue(self.quote()['Book implied total'].isna().all())

    def test_stale_future_wrong_book_and_explicitly_saved_quotes(self):
        for delta in [timedelta(minutes=16),timedelta(minutes=-2)]:
            self.assertTrue(self.quote(now=self.now+delta)['Book implied total'].isna().all())
        self.assertTrue(player_quotes(self.payload,'Sam Hauser','draftkings',now=self.now)['Book implied total'].isna().all())
        self.payload[SOURCE]=dict(saved_at=self.now.isoformat(),fallback=False,reused=True)
        q=self.quote(now=self.now+timedelta(minutes=20)).set_index('Stat').loc['3PM']
        self.assertEqual(q['Market status'],SAVED)
        self.assertEqual(q['Updated UTC'],self.now.isoformat())
        self.assertTrue(np.isfinite(q['Book implied total']))

    def test_regular_quotes_and_other_books_keep_existing_markets(self):
        self.assertEqual(projection_markets('draftkings'),','.join(MARKETS))
        self.assertEqual(projection_markets('betmgm'),','.join(MARKETS))
        self.assertEqual(projection_markets('fanduel'),','.join([*MARKETS,'player_threes_alternate']))
        # Exercise the request path, not just the selector helper.
        import projection_dashboard as dashboard
        with patch.object(dashboard,'read_saved_feed',return_value=self.payload) as fetch:
            dashboard.sportsbook_props(self.event['id'],'fanduel','test',force=True)
        self.assertEqual(fetch.call_args.kwargs['markets'],projection_markets('fanduel'))
        self.assertTrue(fetch.call_args.kwargs['force'])
