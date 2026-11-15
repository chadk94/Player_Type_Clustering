from datetime import datetime, timedelta, timezone
from unittest import TestCase
from unittest.mock import Mock

from jae_card import card_quotes, send_card, worst_price


class JaeCardTests(TestCase):
    def setUp(self):
        self.now = datetime.now(timezone.utc)
        self.start = self.now + timedelta(days=1)
        from jae_card import DENVER
        self.day = self.start.astimezone(DENVER).date().isoformat()
        self.pick = dict(id='a'*64, event='a'*32, day=self.day, book='FanDuel', stat='PTS', side='Under',
                         game='OKC @ SAC', player='Sample Player', line=14.5, status='Confirmed',odds=-110)
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

    def test_stale_moved_or_started_held_for_review(self):
        for pick in [dict(self.pick,line=15.5),dict(self.pick,day='2020-01-01')]:
            rows, skipped=card_quotes([pick],self.fetch,lambda:self.now)
            self.assertEqual((len(rows),len(skipped)),(0,1))
        self.market['last_update']=(self.now-timedelta(minutes=16)).isoformat()
        self.assertFalse(card_quotes([self.pick],self.fetch,lambda:self.now)[0])
        self.event['commence_time']=self.now.isoformat()
        self.assertFalse(card_quotes([self.pick],self.fetch,lambda:self.now)[0])
