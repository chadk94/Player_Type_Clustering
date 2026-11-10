from concurrent.futures import ThreadPoolExecutor
from datetime import date
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from todays_plays import comment, edge_pick, public_token, publish, read_board


class TodaysPlaysTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        env = patch.dict(os.environ, {'TODAYS_PLAYS_DB_PATH':str(Path(temp.name)/'reviews.sqlite')})
        env.start()
        self.addCleanup(env.stop)
        self.day = '2026-11-10'
        self.owner = 'a'*64
        self.row = pd.Series({'PLAYER_ID':33, 'Player':'Example Guard', 'Stat':'AST', 'Book line':4.5,
                              'Over odds':-120,'Under odds':107,'Signed edge %':-8.,'Edge %':8.,'Projected':4.1})
        self.game = {'id':'abc','Home':'NYK','Away':'MIL'}

    def pick(self,row=None,book='FanDuel'):
        return edge_pick(self.row if row is None else row,self.game,book,date.fromisoformat(self.day))

    def test_identity_keeps_direction_book_and_line_while_quote_is_frozen(self):
        p=self.pick()
        self.assertEqual((p['side'],p['odds']),('Under',107))
        changed=self.row.copy()
        changed['Under odds']=111
        self.assertEqual(self.pick(changed)['id'],p['id'])
        changed['Signed edge %']=8.
        self.assertNotEqual(self.pick(changed)['id'],p['id'])
        self.assertEqual(self.pick(changed)['odds'],-120)
        self.assertNotEqual(self.pick(book='DraftKings')['id'],p['id'])
        changed=self.row.copy()
        changed['Book line']=5.5
        self.assertNotEqual(self.pick(changed)['id'],p['id'])
        changed['Stat']='STL'; changed['Under odds']=-200
        self.assertIsNone(self.pick(changed))

    def test_private_scopes_updates_feedback_and_expiry(self):
        p=self.pick()
        token=publish(self.day,self.owner,[p])
        self.assertNotEqual(token,public_token('b'*64,self.day))
        self.assertNotEqual(token,public_token(token,self.day))
        self.assertNotEqual(token,public_token(self.owner,'2026-11-11'))
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda n:comment(token,p['id'],'Partner','Check minutes',n),['a','b','a']))
        self.assertEqual(len(read_board(token)['notes']),2)
        self.assertEqual(publish(self.day,self.owner,[dict(p,status='Confirmed')]),token)
        self.assertEqual(read_board(token)['picks'][0]['status'],'Confirmed')
        self.assertEqual(len(read_board(token)['notes']),2)
        with self.assertRaises(ValueError):
            publish(self.day,self.owner,[dict(p,line=9)])
        self.assertEqual(len(read_board(token)['picks']),1)
        publish(self.day,self.owner,[])
        with self.assertRaises(ValueError):
            comment(token,p['id'],'Partner','Removed','c')
        self.assertEqual(read_board(token)['picks'],[])
        with patch('todays_plays.time.time',return_value=5e9):
            self.assertIsNone(read_board(token))

    def test_partner_form_posts_to_owner_without_any_edit_controls(self):
        p=self.pick()
        token=publish(self.day,self.owner,[p])
        app=AppTest.from_string(f"import streamlit as st\nfrom todays_plays import render_partner_review\nrender_partner_review(st, '{token}')").run()
        self.assertEqual([b.label for b in app.button],['Refresh shared list','Post feedback'])
        app.text_input[0].set_value('Partner').run()
        app.text_area[0].set_value('Minutes look right').run()
        app.button[1].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(read_board(token)['notes'][0]['note'],'Minutes look right')
        self.assertEqual(app.text_area[0].value,'')
        self.assertIn('Feedback posted.',[x.value for x in app.success])
