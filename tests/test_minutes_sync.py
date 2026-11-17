from copy import deepcopy
import unittest
from unittest.mock import patch, Mock
import hashlib
import json

from test_approved_minutes import example, reseal
from minutes_sync import validate_feed, read_feed
from approved_minutes import apply_bundle_change, for_matchup


class MinutesSyncTests(unittest.TestCase):
    def test_approved_dates_revisions_and_zero_minutes_reach_projections(self):
        first, roster = example()
        second = deepcopy(first)
        day = first['games'][0]['game_date']
        tomorrow = day[:-2] + str(int(day[-2:])+1)
        for row in second['games'] + second['approvals']:
            row['game_date'] = tomorrow
            row['game_key'] = row['game_key'].replace(day,tomorrow)
        reseal(second)
        linked = validate_feed(dict(bundles=[first,second]), roster, '2026-27')
        self.assertEqual(len(linked['games']),2)
        self.assertEqual(sum(for_matchup(linked,roster,'BOS','DET',day+'T23:00:00Z').values()),480)
        self.assertEqual(sum(for_matchup(linked,roster,'BOS','DET',tomorrow+'T23:00:00Z').values()),480)
        self.assertEqual(for_matchup(linked,roster,'BOS','DET','2026-12-20T23:00:00Z'),{})
        state=dict(approved_minutes_bundle='old',team_projection_snapshots={'old':1},custom_minutes_2=40,other=3)
        apply_bundle_change(state,linked)
        self.assertEqual(state,dict(approved_minutes_bundle=linked['bundle_id'],other=3))
        state['custom_minutes_2']=40
        apply_bundle_change(state,linked)
        self.assertEqual(state['custom_minutes_2'],40)
        invalid=deepcopy(first)
        invalid['approvals'][0]['players'][0]['minutes']=0
        with self.assertRaises(ValueError):validate_feed(dict(bundles=[invalid]),roster,'2026-27')
        with self.assertRaises(ValueError):validate_feed(dict(bundles=[first,first]),roster,'2026-27')

    def test_private_connections_cannot_request_untrusted_addresses(self):
        for url in ['http://127.0.0.1/', 'https://example.com/exec', 'https://script.google.com.evil/exec']:
            with self.assertRaises(ValueError):read_feed(url+'#'+'a'*64)

    def test_authenticated_transfer_preserves_float_checksums_and_handles_failures(self):
        bundle, roster = example()
        raw = json.dumps(dict(schema='nba_minutes_feed_v1', bundles=[bundle]))
        response = Mock(status_code=200)
        response.json.return_value = dict(ok=True,snapshot=raw,digest=hashlib.sha256(raw.encode()).hexdigest())
        connection = 'https://script.google.com/macros/s/example/exec#'+'a'*64
        with patch('minutes_sync.requests.post',return_value=response) as post:
            self.assertEqual(len(validate_feed(read_feed(connection),roster,'2026-27')['games']),1)
            self.assertFalse(post.call_args.kwargs['allow_redirects'])
            response.json.return_value['digest']='bad'
            with self.assertRaisesRegex(ValueError,'transfer check'):read_feed(connection)
            response.json.return_value=dict(ok=False)
            with self.assertRaisesRegex(ValueError,'rejected'):read_feed(connection)

    def test_first_browser_restore_resumes_the_page_without_a_connection(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_string('''
import streamlit as st
from unittest.mock import patch
from types import SimpleNamespace
from minutes_sync import render_sync
with patch('minutes_sync.connection_component',return_value=lambda **kw: SimpleNamespace(connection={'ready':True,'connection':None})):
    render_sync(None,'2026-27')
st.success('Projection page ready')
''').run(timeout=10)
        self.assertFalse(app.exception)
        self.assertEqual(app.success[0].value,'Projection page ready')
