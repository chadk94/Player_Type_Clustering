import unittest

from streamlit.testing.v1 import AppTest

from opponent_selection import sync_opponent


class OpponentSelectionTests(unittest.TestCase):
    def test_automatic_choice_follows_schedule_and_recovers_from_missing_data(self):
        state = {}
        sync_opponent(state, 'BOS', ['DET', 'NYK'], {'Opponent': 'DET'})
        self.assertEqual(state['projection_opponent'], 'DET')
        sync_opponent(state, 'BOS', ['DET', 'NYK'], {'Opponent': 'NYK'})
        self.assertEqual(state['projection_opponent'], 'NYK')
        sync_opponent(state, 'BOS', ['DET', 'NYK'], None)
        self.assertIsNone(state['projection_opponent'])
        sync_opponent(state, 'BOS', ['DET', 'NYK'], {'Opponent': 'NYK'})
        self.assertEqual(state['projection_opponent'], 'NYK')

    def test_manual_choice_survives_refresh_but_team_switch_resets_it(self):
        state = {'projection_opponent_team': 'BOS', 'projection_opponent': 'DET',
                 'projection_opponent_manual': True}
        for upcoming in [{'Opponent': 'NYK'}, None]:
            sync_opponent(state, 'BOS', ['DET', 'NYK'], upcoming)
            self.assertEqual(state['projection_opponent'], 'DET')
            self.assertTrue(state['projection_opponent_manual'])
        sync_opponent(state, 'DET', ['BOS', 'MIA'], {'Opponent': 'MIA'})
        self.assertEqual(state['projection_opponent'], 'MIA')
        self.assertFalse(state['projection_opponent_manual'])

    def test_missing_or_invalid_opponent_never_defaults_to_own_team(self):
        for upcoming in [None, {'Opponent': 'BOS'}, {'Opponent': 'UNKNOWN'}]:
            state = {}
            sync_opponent(state, 'BOS', ['NYK'], upcoming)
            self.assertIsNone(state['projection_opponent'])

    def test_actual_widgets_preserve_manual_selection_and_restore_automatic(self):
        app = AppTest.from_string('''
import streamlit as st
from opponent_selection import render_opponent_selector
team = st.selectbox('Team', ['BOS', 'PHI'])
opponents = [t for t in ['BOS', 'DET', 'NYK', 'PHI'] if t != team]
next_team = st.session_state.get('next_team', 'DET' if team == 'BOS' else 'NYK')
render_opponent_selector(st, team, opponents, {'Opponent': next_team})
st.slider('Minutes', 0, 48, 30)
''').run()
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key='projection_opponent').value, 'DET')
        app.session_state['next_team'] = 'NYK'
        app.run()
        self.assertEqual(app.selectbox(key='projection_opponent').value, 'NYK')
        app.selectbox(key='projection_opponent').select('DET').run()
        app.slider[0].set_value(25).run()
        self.assertEqual(app.selectbox(key='projection_opponent').value, 'DET')
        app.button(key='reset_projection_opponent').click().run()
        self.assertEqual(app.selectbox(key='projection_opponent').value, 'NYK')
        app.selectbox(key='projection_opponent').select('DET').run()
        app.selectbox[0].select('PHI').run()
        self.assertEqual(app.selectbox(key='projection_opponent').value, 'NYK')
        app.selectbox(key='projection_opponent').select(None).run()
        self.assertEqual(app.selectbox(key='projection_opponent').value, 'NYK')
        self.assertFalse(app.exception)


if __name__ == '__main__':
    unittest.main()
