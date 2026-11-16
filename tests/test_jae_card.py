from datetime import datetime, timedelta, timezone
from unittest import TestCase
from unittest.mock import Mock

from jae_card import card_quotes, send_card, worst_price


class JaeCardTests(TestCase):
    def test_points_x_plus_wins_over_the_same_line_in_a_player_ladder(self):
        from jae_card import alternate_prices, next_over, shopping_prices

        def offer(player, line, odds, market, selection, side='Over'):
            return dict(name=side, description=player, point=line, price=odds,
                        link=f'https://sportsbook.fanduel.com/addToBetslip?marketId={market}&selectionId={selection}')

        player = self.pick['player']
        pick = dict(self.pick, side='Over', line=6.5)
        native_ten = offer(player, 9.5, 265, '42.10', '1001')
        ladder = [
            offer(player, 9.5, 300, '42.100', '2010'), offer(player, 10.5, 400, '42.100', '2011'),
            native_ten, offer('Other Player', 9.5, 125, '42.10', '1002'),
            offer(player, 14.5, 970, '42.15', '1001'),  # Only this player at 15+.
            offer(player, 14.5, 920, '42.100', '2015'),
            offer(player, 9.5, -325, '42.200', '3010', 'Under'),
        ]
        market = dict(key='player_points_alternate', last_update=self.now.isoformat(), outcomes=ladder)
        self.event['bookmakers'][0]['markets'].append(market)
        prices = alternate_prices(self.event, self.now)
        self.assertEqual(next_over(pick, prices), {'line':9.5, 'odds':265})
        self.assertEqual(shopping_prices(pick, prices)['offers'], [
            {'line':9.5,'odds':265}, {'line':10.5,'odds':400}, {'line':14.5,'odds':970}])
        self.assertEqual(prices['sample player','PTS','Under'][9.5], {-325})
        ladder.append(dict(native_ten, price=270))
        self.assertNotIn(9.5, [o['line'] for o in shopping_prices(pick, alternate_prices(self.event,self.now))['offers']])

    def test_request_source_market_links_for_points_alternates(self):
        from unittest.mock import patch
        from jae_card import current_event
        with patch('FanduelScrape.get_odds_api_key', return_value='test'), patch('jae_card.read_feed') as fetch:
            current_event('a'*32, {'PTS'}, {'PTS'})
        self.assertEqual(fetch.call_args.kwargs['includeLinks'], 'true')
        self.assertIn('player_points_alternate', fetch.call_args.kwargs['markets'].split(','))

    def setUp(self):
        self.now = datetime.now(timezone.utc)
        self.start = self.now + timedelta(days=1)
        from jae_card import DENVER
        self.day = self.start.astimezone(DENVER).date().isoformat()
        self.pick = dict(id='a'*64, event='a'*32, day=self.day, book='FanDuel', stat='PTS', side='Under',
                         game='OKC @ SAC', player_id=1, player='Sample Player', line=14.5, status='Confirmed',odds=-110)
        self.market = dict(key='player_points',last_update=self.now.isoformat(),
                           outcomes=[dict(name=s,description=self.pick['player'],point=14.5,price=o)
                                     for s,o in [('Under',102),('Over',-115)]])
        self.event = dict(id='a'*32, home_team='Sacramento Kings', away_team='Oklahoma City Thunder',
                          commence_time=self.start.isoformat(),bookmakers=[dict(key='fanduel',markets=[self.market])])
        self.fetch = Mock(return_value=self.event)

    def test_five_cent_ticks_and_current_price(self):
        for quote, worst in [(-110,-115),(130,125),(105,100),(102,-103),(100,-105),(-100,-105)]:
            self.assertEqual(worst_price(quote),worst)
        rows, skipped = card_quotes([self.pick],self.fetch,lambda:self.now)
        self.assertFalse(skipped)
        row, = rows
        self.assertEqual((row['odds'],row['worst'],row['line'],row['start'],row['sent']),
                         (102,-103,14.5,self.start.isoformat(),self.now.isoformat()))
        self.assertAlmostEqual(row['breakeven'],103/203)

    def test_new_only_and_retry(self):
        calls=[]
        written={}
        def request(connection,action,day,ids=(),rows=()):
            calls.append(action)
            if action=='check': return dict(existing=[pid for pid in ids if pid in written])
            for r in rows: written.setdefault(r['id'],r)
            return dict(added=len(rows))
        picks=[self.pick,dict(self.pick,id='b'*64,status='Review')]
        self.assertIn('1 added',send_card(self.day,picks,'test',request,self.fetch)[0])
        self.assertIn('1 already sent',send_card(self.day,picks,'test',request,self.fetch)[0])
        self.assertEqual(calls,['check','append','check'])
        self.fetch.assert_called_once()
        self.assertEqual(list(written),[self.pick['id']])

    def test_confidence_sets_play_to_from_the_fresh_side_price(self):
        for cents, limit in [(5,-115),(10,-120),(15,-125)]:
            self.assertEqual(worst_price(-110,cents),limit)
            row, = card_quotes([dict(self.pick,confidence_cents=cents)],self.fetch,lambda:self.now)[0]
            self.assertEqual(row['odds'],102)
            self.assertEqual(row['worst'],-(98+cents))
            self.assertAlmostEqual(row['breakeven'],(98+cents)/(198+cents))
        with self.assertRaises(ValueError):
            worst_price(-110,7)

    def test_stale_moved_or_started_held_for_review(self):
        for pick in [dict(self.pick,line=15.5),dict(self.pick,day='2020-01-01')]:
            rows, skipped=card_quotes([pick],self.fetch,lambda:self.now)
            self.assertEqual((len(rows),len(skipped)),(0,1))
        self.market['last_update']=(self.now-timedelta(minutes=16)).isoformat()
        self.assertFalse(card_quotes([self.pick],self.fetch,lambda:self.now)[0])
        self.event['commence_time']=self.now.isoformat()
        self.assertFalse(card_quotes([self.pick],self.fetch,lambda:self.now)[0])

    def test_exact_threes_alt_uses_one_sided_fresh_fanduel_market(self):
        pick = dict(self.pick,stat='3PM',side='Over',line=1.5,book='DraftKings')
        self.market['key'] = 'player_threes'
        for outcome in self.market['outcomes']:
            outcome['point'] = 1.5
        alt = dict(key='player_threes_alternate', last_update=self.now.isoformat(),
                   outcomes=[dict(name='Over',description=pick['player'],point=line,price=odds)
                             for line,odds in [(1.5,-115),(2.5,400),(3.5,1000)]])
        self.event['bookmakers'][0]['markets'].append(alt)
        self.event['bookmakers'].append(dict(key='draftkings',markets=[dict(alt,outcomes=[
            dict(name='Over',description=pick['player'],point=2.5,price=550)])]))
        def row():
            rows, skipped=card_quotes([pick],self.fetch,lambda:self.now)
            self.assertFalse(skipped)
            return rows[0]
        self.assertEqual((row()['book'],row()['odds'],row()['alt']),('FanDuel',-115,{'line':2.5,'odds':400}))
        alt['outcomes'].append(dict(alt['outcomes'][1],price=420))
        self.assertIsNone(row()['alt']['odds'])
        alt['outcomes'] = [alt['outcomes'][2]]
        self.assertEqual(row()['alt'],{'line':2.5,'odds':None})
        alt['last_update']=(self.now-timedelta(minutes=16)).isoformat()
        self.assertIsNone(row()['alt']['odds'])
        from jae_card import next_over
        self.assertEqual(next_over(dict(pick,stat='PTS',line=14.5),
                         {('sample player','PTS','Over'):{19.5:{180},24.5:{400}}}),{'line':19.5,'odds':180})

    def test_cross_book_saves_use_one_fanduel_id_and_recognize_legacy_rows(self):
        from play_identity import prop_id
        original=dict(self.pick,book='DraftKings')
        original['id']=prop_id(original)
        fd=dict(original,book='FanDuel')
        fd['id']=prop_id(fd)
        req=Mock(side_effect=[dict(existing=[]),dict(added=1)])
        sent,_,_=send_card(self.day,[dict(original,confidence_cents=10),dict(fd,confidence_cents=15)],'test',req,self.fetch)
        self.assertIn('1 added',sent)
        frozen,=req.call_args.kwargs['rows']
        self.assertEqual((frozen['id'],frozen['book']),(fd['id'],'FanDuel'))
        self.assertEqual(frozen['worst'],-108)
        self.fetch.reset_mock()
        req=Mock(return_value=dict(existing=[original['id']]))
        self.assertIn('1 already sent',send_card(self.day,[fd],'test',req,self.fetch)[0])
        self.fetch.assert_not_called()

    def test_resending_an_edit_targets_original_row_without_requoting(self):
        req=Mock(side_effect=[dict(existing=[self.pick['id']],confidenceEdits=True),dict(added=0,updated=1)])
        p=dict(self.pick,confidence_cents=15,confidence_edited=True)
        message,skipped,synced=send_card(self.day,[p],'test',req,self.fetch)
        self.assertEqual(req.call_args.kwargs,dict(rows=[],edits=[dict(id=p['id'],cents=15)]))
        self.assertEqual((skipped,synced),([],{p['id']:15}))
        self.assertIn('1 updated',message)
        self.fetch.assert_not_called()

    def test_common_points_and_rebound_alt_milestones(self):
        from jae_card import next_over
        for stat, line, ladder, expected in [
            ('PTS', 6.5, {7.5:{120},9.5:{180}}, 9.5),
            ('REB', 3.5, {4.5:{120},5.5:{180}}, 5.5),
            ('REB', 3.5, {4.5:{120},5.5:{150,180}}, 4.5),
        ]:
            result = next_over(dict(self.pick,stat=stat,side='Over',line=line),{('sample player',stat,'Over'):ladder})
            self.assertEqual(result,{'line':expected,'odds':next(iter(ladder[expected]))})

    def test_unders_only_use_fresh_same_side_fd_alternates(self):
        from jae_card import alternate_prices, shopping_prices
        p = dict(self.pick, line=10.5)
        market = dict(key='player_points_alternate', last_update=self.now.isoformat(), outcomes=[
            dict(name=s, description=p['player'], point=9.5, price=o) for s,o in [('Over',-150),('Under',125)]])
        self.event['bookmakers'][0]['markets'].append(market)
        get = lambda: shopping_prices(p,alternate_prices(self.event,self.now))
        self.assertEqual(get()['offers'],[{'line':9.5, 'odds':125}])
        market['last_update'] = (self.now - timedelta(minutes=16)).isoformat()
        self.assertEqual(get()['offers'],[])
