import unittest

import pandas as pd

from player_navigation import selected_player_request, apply_pending_navigation, selected_game_context
from opponent_selection import sync_opponent


class PlayerNavigationTests(unittest.TestCase):
    def setUp(self):
        self.rows = pd.DataFrame([
            dict(PLAYER_ID=5, Player='Home player', Team='HOU'),
            dict(PLAYER_ID=6, Player='Away player', Team='MIL'),
        ], index=[45, 91])
        self.roster = self.rows.rename(columns={'Player': 'PLAYER_NAME', 'Team': 'TEAM_ABBREVIATION'})
        self.game = dict(Home='HOU', Away='MIL', GameTimeUTC=pd.Timestamp('2026-10-29T00:00:00Z'))

    def test_cells_and_rows_open_either_side_by_position_not_dataframe_index(self):
        for selection, position, opponent in [({'cells': [(0, 'Player')]}, 0, 'MIL'),
                                               ({'rows': [1]}, 1, 'HOU')]:
            request = selected_player_request(selection, self.rows, self.game, 'FanDuel')
            self.assertEqual(request['player_id'], self.rows.iloc[position].PLAYER_ID)
            self.assertEqual(request['opponent'], opponent)
            self.assertEqual(request['start'], self.game['GameTimeUTC'].isoformat())

    def test_navigation_is_consumed_once_preserves_minutes_and_beats_next_opponent(self):
        state = {'pending_player_navigation': selected_player_request({'rows': [1]}, self.rows, self.game, 'FanDuel'),
                 'custom_minutes_2026-27_6': 35.5, 'approved_bundle': 'saved',
                 'projection_opponent_manual': False, 'projection_opponent_team': 'ATL'}
        apply_pending_navigation(state, self.roster)
        sync_opponent(state, 'MIL', ['HOU', 'ATL'], {'Opponent': 'ATL'})
        self.assertEqual(state['projection_player'], 'Away player')
        self.assertEqual(state['projection_opponent'], 'HOU')
        self.assertEqual(state['projection_team'], 'MIL')
        self.assertEqual(state['custom_minutes_2026-27_6'], 35.5)
        self.assertEqual(state['approved_bundle'], 'saved')
        self.assertEqual(state['player_navigation_revision'], 1)
        apply_pending_navigation(state, self.roster)
        self.assertEqual(state['player_navigation_revision'], 1)
        self.assertEqual(selected_game_context(state, 'MIL', 'HOU')['GameTimeUTC'], self.game['GameTimeUTC'])
        self.assertIsNone(selected_game_context(state, 'MIL', 'ATL'))
        self.assertNotIn('projection_game_context', state)

    def test_empty_invalid_and_stale_selections_do_not_navigate(self):
        for selection in ({}, {'rows': [2]}, {'cells': [(-1, 'Player')]}, {'cells': [(0, 'Projected minutes')]}):
            self.assertIsNone(selected_player_request(selection, self.rows, self.game, 'FanDuel'))
        request = selected_player_request({'rows': [0]}, self.rows, self.game, 'FanDuel')
        request['player_id'] = 999
        state = {'pending_player_navigation': request}
        apply_pending_navigation(state, self.roster)
        self.assertEqual(state, {})
