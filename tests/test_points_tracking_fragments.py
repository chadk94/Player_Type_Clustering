"""Reproduce browser acknowledgements inside nested Streamlit fragments."""
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.local_script_runner import LocalScriptRunner

import points_tracking as tracking
from tests import test_points_probability as fixtures


class TrackingFragmentTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.PointsProbabilityTests()
        fixture.setUp()
        self.record = fixture.record()
        self.logs = pd.DataFrame([dict(ESPN_PLAYER_ID=self.record['espn_id'],
            GAME_ID='123', GAME_DATE=self.record['start'], Team='NYK',
            Opponent='PHI', PTS=6., MIN=23.)])
        self.fragment_id = None
        self.storage = None
        self.reply_immediately = False
        self.sent = []
        owner = self

        class FragmentRunner(LocalScriptRunner):
            def __init__(runner, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if owner.storage is None:
                    owner.storage = runner._fragment_storage
                else:
                    runner._fragment_storage = owner.storage

            def request_rerun(runner, data):
                return super().request_rerun(replace(data, fragment_id=owner.fragment_id))

        def component(*, data, key, on_reply_change, **kwargs):
            self.sent.append(data)
            response = dict(request=data['request'], status='ok',
                records=[json.dumps(self.record)], results=data['results'])
            def deliver():
                st.session_state[key] = {'reply': response}
                on_reply_change()
            st.button('Browser acknowledgement', key='browser_ack', on_click=deliver)
            reply = response if self.reply_immediately else st.session_state.get(key, {}).get('reply')
            return SimpleNamespace(reply=reply)

        for patcher in [
            patch('streamlit.testing.v1.app_test.LocalScriptRunner', FragmentRunner),
            patch.object(tracking, 'tracking_component', return_value=component),
            patch.object(tracking, '_test_logs', self.logs, create=True),
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.app = AppTest.from_string("""
import streamlit as st
import points_tracking as tracking
@st.fragment
def preview():
    st.session_state['preview_runs'] = st.session_state.get('preview_runs', 0) + 1
    st.button('Refresh preview', key='parent')
    tracking.render_probability_tracking(st, [], tracking._test_logs, 'test')
preview()
""").run()
        self.assertFalse(self.app.exception)

    def run_button(self, key):
        button = self.app.button(key=key)
        self.fragment_id = self.app.session_state._state._get_widget_metadata(button.id).fragment_id
        button.click().run()
        self.assertFalse(self.app.exception, [e.message for e in self.app.exception])

    def test_acknowledgement_during_parent_and_full_rerun_never_requests_nested_rerun(self):
        self.reply_immediately = True
        self.run_button('parent')
        self.assertEqual(self.app.session_state['points_tracking_test']['records'], [self.record])
        self.assertTrue(any('1 snapshots saved' in c.value for c in self.app.caption))
        self.fragment_id = None
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.sent[-1]['results'][0]['outcome'], 'Under')

    def test_callback_restores_then_grades_without_rebuilding_parent(self):
        count = self.app.session_state['preview_runs']
        self.run_button('browser_ack')
        self.assertEqual(self.app.session_state['preview_runs'], count)
        self.assertEqual(self.app.session_state['points_tracking_test']['records'], [self.record])
        self.assertEqual(self.sent[-1]['results'][0]['id'], self.record['id'])
        self.assertEqual(self.sent[-1]['results'][0]['outcome'], 'Under')
        self.run_button('browser_ack')
        self.assertEqual(self.app.session_state['preview_runs'], count)
        self.assertEqual(self.app.session_state['points_tracking_test']['results'][0]['outcome'], 'Under')

    def test_stale_and_invalid_history_does_not_replace_valid_forecasts(self):
        key = 'points_tracking_test'
        state = {key: dict(records=[self.record], results=[], status='ok')}
        tracking.accept_tracking_reply(state, key, 'current',
            dict(request='old', status='ok', records=[], results=[]))
        self.assertEqual(state[key]['records'], [self.record])
        tracking.accept_tracking_reply(state, key, 'current',
            dict(request='current', status='ok', records=[json.dumps(self.record), '{broken'], results=[]))
        self.assertEqual(state[key]['records'], [self.record])
        self.assertEqual(state[key]['invalid_records'], 1)
        tracking.accept_tracking_reply(state, key, 'current',
            dict(request='current', status='unavailable'))
        self.assertEqual(state[key]['status'], 'unavailable')
        self.assertEqual(state[key]['records'], [self.record])


    def test_incremental_reply_preserves_history_and_rejects_stale_parent(self):
        key = 'history'
        state = {key: dict(records=[], results=[], status='loading')}
        first = json.dumps(self.record)
        second = dict(self.record, mean=self.record['mean'], mode='Manual minutes')
        second['id'] = tracking.record_id(second)
        tracking.accept_tracking_reply(state, key, 'restore', dict(request='restore', full=True, status='ok', records=[first]))
        delta = dict(request='next', base='wrong', full=False, status='ok', records=[json.dumps(second)])
        tracking.accept_tracking_reply(state, key, 'next', delta)
        self.assertEqual(state[key]['records'], [self.record])
        tracking.accept_tracking_reply(state, key, 'next', dict(delta, base='restore', records=[first,json.dumps(second)]))
        self.assertEqual(state[key]['records'], [self.record, second])
        tracking.accept_tracking_reply(state, key, 'reset', dict(request='reset', full=True, status='ok', records=[]))
        self.assertFalse(state[key]['records'])
