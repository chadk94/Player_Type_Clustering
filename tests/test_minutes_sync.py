from copy import deepcopy
import unittest
from unittest.mock import patch, Mock
import hashlib
import json
import requests

from test_approved_minutes import example, reseal
from minutes_sync import validate_feed, read_feed
from background_feed import BackgroundFeed
from test_background_feed import ImmediateExecutor
from approved_minutes import apply_bundle_change, for_matchup


class MinutesSyncTests(unittest.TestCase):
    connection = 'https://script.google.com/macros/s/example/exec#'+'a'*64

    def setUp(self):
        patcher = patch('minutes_sync.minutes_feed_worker', return_value=BackgroundFeed(ImmediateExecutor()))
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def response(status=200, snapshot=None):
        response = Mock(status_code=status)
        if status >= 400:
            response.raise_for_status.side_effect = requests.HTTPError('private-url', response=response)
        raw = json.dumps(snapshot or dict(schema='nba_minutes_feed_v1', bundles=[]))
        response.json.return_value = dict(ok=True, snapshot=raw, digest=hashlib.sha256(raw.encode()).hexdigest())
        return response

    def test_transient_errors_retry_once_and_recover(self):
        for failure in [requests.Timeout('private-url'), requests.ConnectionError('private-url'), self.response(503)]:
            with self.subTest(failure=type(failure).__name__), patch('minutes_sync.sleep') as sleep, patch(
                'minutes_sync.requests.post', side_effect=[failure, self.response()]
            ) as post:
                self.assertEqual(read_feed(self.connection)['bundles'], [])
                self.assertEqual(post.call_count, 2)
                sleep.assert_called_once()

    def test_redirect_retry_does_not_forward_credentials(self):
        redirect = Mock(status_code=302, headers={'Location':'https://script.googleusercontent.com/macros/echo?private=example'})
        with patch('minutes_sync.sleep'), patch('minutes_sync.requests.post', return_value=redirect) as post, patch(
            'minutes_sync.requests.get', side_effect=[requests.Timeout('private-url'), self.response()]
        ) as get:
            self.assertEqual(read_feed(self.connection)['bundles'], [])
            self.assertEqual(post.call_count, 2)
            self.assertEqual(get.call_count, 2)
            for call in get.call_args_list:
                self.assertEqual(set(call.kwargs), {'timeout', 'allow_redirects'})
                self.assertFalse(call.kwargs['allow_redirects'])

    def test_retries_are_bounded_and_errors_do_not_expose_private_urls(self):
        cases = [(requests.Timeout('private-url'), 'timed out', 2),
                 (requests.ConnectionError('private-url'), 'connection', 2),
                 (self.response(503), 'HTTP 503', 2),
                 (self.response(401), 'denied access', 1),
                 (self.response(403), 'denied access', 1)]
        for failure, message, attempts in cases:
            kwargs = dict(side_effect=failure) if isinstance(failure, Exception) else dict(return_value=failure)
            with self.subTest(message=message), patch('minutes_sync.sleep'), patch('minutes_sync.requests.post', **kwargs) as post:
                with self.assertRaisesRegex(ValueError, message) as error:
                    read_feed(self.connection)
                self.assertNotIn('private-url', str(error.exception))
                self.assertEqual(post.call_count, attempts)

    def test_untrusted_redirect_and_invalid_payloads_are_not_retried(self):
        redirect = Mock(status_code=302, headers={'Location':'https://example.com/private'})
        unreadable = self.response()
        unreadable.json.side_effect = ValueError('private-url')
        non_object = self.response()
        non_object.json.return_value = []
        bad_checksum = self.response()
        bad_checksum.json.return_value['digest'] = 'bad'
        invalid_snapshot = self.response(snapshot=['invalid'])
        for response, message in [(redirect,'owner'), (unreadable,'unreadable'), (non_object,'unsupported'),
                                  (bad_checksum,'transfer check'), (invalid_snapshot,'unsupported')]:
            with self.subTest(message=message), patch('minutes_sync.requests.post', return_value=response) as post, patch('minutes_sync.requests.get') as get:
                with self.assertRaisesRegex(ValueError, message):
                    read_feed(self.connection)
                post.assert_called_once()
                get.assert_not_called()

    def test_failed_poll_preserves_verified_minutes_and_retry_recovers(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_string('''
import streamlit as st
from unittest.mock import patch
from types import SimpleNamespace
from minutes_sync import render_sync, validate_feed
from approved_minutes import apply_bundle_change
from test_approved_minutes import example
bundle, roster = example()
feed = {'bundles':[bundle]}
linked = validate_feed(feed, roster, '2026-27')
connection = 'https://script.google.com/macros/s/example/exec#'+'a'*64
if 'nba_minutes_link' not in st.session_state:
    st.session_state['nba_minutes_link'] = dict(ready=True, connection=connection, bundle=linked, success_at='Nov 20, 08:00 PM UTC')
    st.session_state['approved_minutes_bundle'] = linked['bundle_id']
    st.session_state['custom_minutes_2'] = 37
with patch('minutes_sync.connection_component',return_value=lambda **kw: SimpleNamespace(connection={'ready':True,'connection':connection})):
    with patch('minutes_sync.read_feed', side_effect=ValueError('Google’s minutes feed timed out after retrying.') if not st.session_state.get('recovered') else None, return_value=feed):
        render_sync(roster,'2026-27')
apply_bundle_change(st.session_state,st.session_state['nba_minutes_link']['bundle'])
st.success('Verified minutes still available')
''').run(timeout=10)
        self.assertFalse(app.exception)
        self.assertIn('Using the last verified approvals', app.warning[0].value)
        self.assertIn('Nov 20, 08:00 PM UTC', app.warning[0].value)
        self.assertEqual(app.session_state['custom_minutes_2'], 37)
        original = app.session_state['nba_minutes_link']['bundle']
        app.session_state['recovered'] = True
        app.button(key='retry_nba_minutes_sync').click().run(timeout=10)
        self.assertFalse(app.exception)
        self.assertFalse(app.warning)
        self.assertEqual(app.session_state['nba_minutes_link']['bundle'], original)
        self.assertEqual(app.session_state['custom_minutes_2'], 37)

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

    def test_pending_refresh_keeps_controls_and_verified_minutes_available(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event
        from time import perf_counter
        from types import SimpleNamespace
        from streamlit.testing.v1 import AppTest
        raw, roster = example()
        feed = dict(bundles=[raw])
        linked = validate_feed(feed, roster, '2026-27')
        release = Event()
        def slow_read(_):
            release.wait(5)
            return feed
        code = '''
import streamlit as st
from minutes_sync import render_sync
from test_approved_minutes import example
_, roster = example()
render_sync(roster, '2026-27')
st.slider('Still responsive', 0, 48, key='test_minutes_control')
'''
        with ThreadPoolExecutor(max_workers=1) as pool:
            worker = BackgroundFeed(pool)
            component = lambda **kw: SimpleNamespace(connection={'ready':True,'connection':self.connection})
            app = AppTest.from_string(code)
            app.session_state['nba_minutes_link'] = dict(connection=self.connection, ready=True, bundle=linked)
            with patch('minutes_sync.connection_component',return_value=component), \
                 patch('minutes_sync.minutes_feed_worker',return_value=worker), \
                 patch('minutes_sync.read_feed',side_effect=slow_read) as read:
                try:
                    start = perf_counter()
                    app.run(timeout=2)
                    self.assertLess(perf_counter()-start, 1)
                    self.assertFalse(app.exception)
                    self.assertTrue(app.session_state['nba_minutes_link']['pending'])
                    app.slider[0].set_value(37).run(timeout=2)
                    self.assertFalse(app.exception)
                    self.assertEqual(app.slider[0].value, 37)
                    self.assertEqual(app.session_state['nba_minutes_link']['bundle'], linked)
                    read.assert_called_once()
                finally:
                    release.set()
                worker.future.result(timeout=2)
                app.run(timeout=2)
                self.assertFalse(app.exception)
                self.assertFalse(app.session_state['nba_minutes_link']['pending'])
                self.assertEqual(app.session_state['nba_minutes_link']['bundle'], linked)
                self.assertEqual(app.slider[0].value, 37)
