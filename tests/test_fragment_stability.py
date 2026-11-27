"""Reproduce overlapping dashboard/child refreshes with Streamlit's real runner."""
from dataclasses import replace
from unittest import TestCase
from unittest.mock import patch
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.local_script_runner import LocalScriptRunner


APP = '''
import streamlit as st
def count(name):
    st.session_state[name] = st.session_state.get(name, 0) + 1
@st.fragment
def dashboard():
    count('dashboard_runs')
    st.button('Minutes update', key='minutes_update')
    player, edges = st.tabs(['Player', 'Edges'])
    @st.fragment
    def player_page():
        count('player_runs')
        st.selectbox('Player', ['A', 'B'], key='player')
        @st.fragment
        def preview():
            count('preview_runs')
            minutes = st.slider('Minutes', 0., 48., 30., key='minutes')
            st.write(f'Projection: {minutes * .5}')
        preview()
    @st.fragment
    def edges_page():
        count('edges_runs')
        st.date_input('Game date', key='edges_date')
        st.selectbox('Game', ['HOU @ IND', 'BOS @ DEN'], key='game')
    with player: player_page()
    with edges: edges_page()
dashboard()
'''


class FragmentStabilityTests(TestCase):
    def setUp(self):
        owner = self
        self.storage = None
        self.queue = []

        class QueuedRunner(LocalScriptRunner):
            def __init__(runner, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if owner.storage is None:
                    owner.storage = runner._fragment_storage
                else:
                    runner._fragment_storage = owner.storage
                # LocalScriptRunner starts with a full-app request. Consume it
                # before supplying the browser's fragment-only request; 1.67
                # correctly gives an already-pending full-app run precedence.
                runner._requests.on_scriptrunner_ready()

            def request_rerun(runner, data):
                return super().request_rerun(replace(data, fragment_id_queue=owner.queue))

        runner = patch('streamlit.testing.v1.app_test.LocalScriptRunner', QueuedRunner)
        runner.start()
        self.addCleanup(runner.stop)
        self.app = AppTest.from_string(APP).run()

    def fragment(self, widget):
        return self.app._session_state._state._get_widget_metadata(widget.id).fragment_id

    def check_queue(self, queue, counters):
        before = {key: self.app.session_state[key] for key in counters}
        self.queue = queue
        self.app.run()
        self.assertFalse(self.app.exception, [e.message for e in self.app.exception])
        for key in counters:
            self.assertEqual(self.app.session_state[key], before[key] + 1, key)

    def test_minutes_refresh_and_edges_change_render_once_in_both_orders(self):
        parent = self.fragment(self.app.button(key='minutes_update'))
        edges = self.fragment(self.app.date_input(key='edges_date'))
        for queue in ([parent, edges], [edges, parent]):
            with self.subTest(queue=queue):
                self.check_queue(queue, ['dashboard_runs', 'player_runs', 'preview_runs', 'edges_runs'])
                self.assertEqual(len(self.app.date_input), 1)
                self.assertEqual(len(self.app.slider), 1)

    def test_background_approval_refresh_updates_all_pages_without_user_interaction(self):
        from copy import deepcopy
        from background_feed import BackgroundFeed
        from test_background_feed import ImmediateExecutor
        from test_approved_minutes import example, reseal
        import minutes_sync
        import streamlit as st
        first, _ = example()
        def rotation(minutes, version):
            raw=deepcopy(first)
            record=raw['approvals'][0]
            record['approval_version']=version
            record['approved_at']=f'2026-10-21T12:0{version}:00+00:00'
            for i, player in enumerate(record['players'][:6]):
                player['minutes']=minutes if i==0 else (240-minutes)/5
            return {'bundles':[reseal(raw)]}
        feed=rotation(24.,1)
        connection='https://script.google.com/macros/s/example/exec#'+'a'*64
        def component(**kwargs):
            # The real browser schedules this fragment's timer independently.
            st.button('Sync panel',key='sync_panel')
            return SimpleNamespace(connection={'ready':True,'connection':connection})
        worker=BackgroundFeed(ImmediateExecutor())
        self.storage=None
        self.queue=[]
        code='''
import streamlit as st
from approved_minutes import render_import, for_matchup
from test_approved_minutes import example
_, roster = example()
@st.fragment(key='projection_dashboard')
def dashboard():
    st.session_state['model_runs']=st.session_state.get('model_runs',0)+1
    bundle=render_import(st,roster,'2026-27')
    minutes=for_matchup(bundle,roster,'BOS','DET','2026-10-21T23:00:00Z')[1]
    st.selectbox('Selected game',['BOS-DET','POR-GSW'],key='selected_game')
    for label in ['Player', 'Matchups', 'Sportsbook Edges', 'Team Projections']:
        st.metric(label,minutes)
    st.slider('Preview minutes',0.,48.,minutes,key='custom_minutes_1')
    st.session_state['team_projection_snapshots']={'minutes':minutes}
dashboard()
'''
        with patch.object(minutes_sync,'connection_component',return_value=component), \
             patch.object(minutes_sync,'minutes_feed_worker',return_value=worker), \
             patch.object(minutes_sync,'read_feed',return_value=feed) as read:
            self.app=AppTest.from_string(code).run()
            self.assertFalse(self.app.exception)
            self.assertTrue(all(float(m.value)==24 for m in self.app.metric))
            self.app.session_state['custom_minutes_1']=11.
            self.app.session_state['selected_game']='POR-GSW'
            runs=self.app.session_state['model_runs']
            sync_id=self.fragment(self.app.button(key='sync_panel'))
            self.queue=[sync_id]
            read.return_value=rotation(30.,2)
            worker.invalidate()
            self.app.run()  # Timer only: no game, slider or refresh button click.
            self.assertFalse(self.app.exception,[e.message for e in self.app.exception])
            self.assertEqual(self.app.session_state['model_runs'],runs+1)
            self.assertEqual(len(self.app.metric),4)
            self.assertTrue(all(float(m.value)==30 for m in self.app.metric))
            self.assertEqual(self.app.slider[0].value,30.)
            self.assertEqual(self.app.selectbox[0].value,'POR-GSW')
            self.assertEqual(self.app.session_state['team_projection_snapshots'],{'minutes':30.})
            self.assertTrue(self.app.get('toast'))
            # An unchanged poll must not rebuild the model or clear a new preview.
            runs=self.app.session_state['model_runs']
            self.app.session_state['custom_minutes_1']=35.
            self.queue=[self.fragment(self.app.button(key='sync_panel'))]
            worker.invalidate()
            self.app.run()
            self.assertFalse(self.app.exception)
            self.assertEqual(self.app.session_state['model_runs'],runs)
            self.assertEqual(self.app.session_state['custom_minutes_1'],35.)
            self.assertFalse(self.app.metric)  # Timer emitted only its own panel.

    def test_player_and_slider_overlap_preserves_latest_values(self):
        player = self.fragment(self.app.selectbox(key='player'))
        preview = self.fragment(self.app.slider(key='minutes'))
        for queue in ([preview, player], [player, preview]):
            with self.subTest(queue=queue):
                self.app.selectbox(key='player').select('B')
                self.app.slider(key='minutes').set_value(35.)
                self.check_queue(queue, ['player_runs', 'preview_runs'])
                self.assertEqual(self.app.selectbox(key='player').value, 'B')
                self.assertEqual(self.app.slider(key='minutes').value, 35.)
                self.assertEqual(self.app.session_state['dashboard_runs'], 1)
                self.assertEqual(self.app.session_state['edges_runs'], 1)

    def test_tabs_timestamp_callback_is_local_but_real_changes_refresh_parent(self):
        import streamlit as st
        import tabs_minutes
        from tests.test_tabs_minutes import browser, ROSTER, GAME, NOW
        pending={'ok':True,'snapshot':browser()}
        real_parse=tabs_minutes.parse_tabs_browser_snapshot

        def component(*, key, on_result_change, **kwargs):
            def deliver():
                st.session_state[key]={'result':pending}
                on_result_change()
            st.button('Source update',key='source_update',on_click=deliver)
            return SimpleNamespace(result=st.session_state.get(key,{}).get('result'))

        self.storage=None
        self.queue=[]
        with patch.object(tabs_minutes,'_browser_component',return_value=component), \
             patch.object(tabs_minutes,'parse_tabs_browser_snapshot',side_effect=lambda s,r,g,a=None,**kw:real_parse(s,r,g,NOW,**kw)):
            self.app=AppTest.from_string('''
import streamlit as st
from tabs_minutes import render_tabs_comparison
from tests.test_tabs_minutes import ROSTER, GAME
@st.fragment(key='projection_dashboard')
def dashboard():
    st.session_state['model_runs']=st.session_state.get('model_runs',0)+1
    st.button('Refresh projections',key='parent')
    st.session_state['tabs_minutes_reference']=render_tabs_comparison(st,ROSTER,[GAME])
dashboard()
''').run()
            def deliver():
                self.queue=[self.fragment(self.app.button(key='source_update'))]
                self.app.button(key='source_update').click().run()
                self.assertFalse(self.app.exception,[e.message for e in self.app.exception])
            deliver()
            self.assertEqual(self.app.session_state['model_runs'],2)
            pending={'ok':True,'snapshot':browser()|{'source_updated_at':NOW.isoformat()}}
            deliver()
            self.assertEqual(self.app.session_state['model_runs'],2)
            self.assertEqual(self.app.session_state['tabs_minutes_reference'].source_updated_at.iloc[0],NOW.isoformat())
            pending['snapshot']['rows'][0]['minutes']=21
            deliver()
            self.assertEqual(self.app.session_state['model_runs'],3)
            self.assertEqual(self.app.session_state['tabs_minutes_reference'].minutes.iloc[0],21)
            pending={'ok':False,'error':'Disconnected'}
            deliver()
            self.assertEqual(self.app.session_state['model_runs'],4)
            retained=self.app.session_state['tabs_minutes_reference']
            self.assertEqual(retained.minutes.iloc[0],21)
            self.assertTrue(retained.source_saved.all())
            self.assertEqual(retained.source_updated_at.iloc[0],NOW.isoformat())
            deliver()
            self.assertEqual(self.app.session_state['model_runs'],4)
            pending={'ok':True,'snapshot':browser()|{'source_updated_at':NOW.isoformat()}}
            deliver()
            self.assertEqual(self.app.session_state['model_runs'],5)
            self.assertFalse(self.app.session_state['tabs_minutes_reference'].source_saved.any())
