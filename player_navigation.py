"""Open a contribution row in the player preview, retaining its exact game."""
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st


def selected_player_request(selection, rows, game, book):
    """Streamlit selection positions refer to input rows, even after UI sorting."""
    selected = selection.get('rows', [])
    cells = selection.get('cells', [])
    if cells and cells[0][1] != 'Player':
        return None
    position = cells[0][0] if cells else selected[0] if selected else None
    if not isinstance(position, int) or not 0 <= position < len(rows):
        return None
    row = rows.iloc[position]
    team = row['Team']
    if team not in (game['Home'], game['Away']):
        return None
    return dict(player_id=int(row['PLAYER_ID']), player_name=row['Player'], team=team,
                opponent=game['Away'] if team == game['Home'] else game['Home'],
                start=game['GameTimeUTC'].isoformat(), book=book)


def apply_pending_navigation(state, roster):
    """Apply once, before destination widgets are constructed on the app rerun."""
    request = state.pop('pending_player_navigation', None)
    if not request:
        return
    player = roster.loc[roster.PLAYER_ID.eq(request['player_id']) &
                        roster.TEAM_ABBREVIATION.eq(request['team'])]
    if len(player) != 1 or request['opponent'] == request['team']:
        return
    name = player.iloc[0].PLAYER_NAME
    state['projection_team'] = request['team']
    state['projection_player'] = name
    state['projection_opponent_team'] = request['team']
    state['projection_opponent'] = request['opponent']
    state['projection_opponent_manual'] = True
    state['projection_sportsbook'] = request['book']
    state['projection_game_context'] = request
    revision = state.get('player_navigation_revision', 0) + 1
    state['player_navigation_revision'] = revision
    state['player_navigation_applied'] = request.get('request_id')


def selected_game_context(state, team, opponent):
    """Manual team/opponent changes return to the ordinary schedule preview."""
    context = state.get('projection_game_context')
    if context and (context['team'], context['opponent']) != (team, opponent):
        state.pop('projection_game_context', None)
        return None
    if context:
        return dict(Opponent=opponent, GameTimeUTC=pd.Timestamp(context['start']))
    return None


def queue_player_selection(key, rows, game, book):
    """Selection callback clears the table before its fragment reruns."""
    selection = st.session_state[key]['selection']
    request = selected_player_request(selection, rows, game, book)
    if request:
        request['request_id'] = uuid4().hex
        st.session_state['player_navigation_signal'] = request
        st.session_state['contribution_selection_revision'] = st.session_state.get('contribution_selection_revision', 0) + 1


_component = None


def navigation_component():
    global _component
    if _component is None:
        from streamlit.components.v2 import component
        _component = component('player_navigation', js=Path(__file__).with_suffix('.mjs').read_text())
    return _component


def render_player_navigation():
    """Must run inside the player fragment, before its selectboxes.

    A browser event transfers the selection between the two fragments. Only
    the player fragment reruns; no full-dashboard or league-feed refresh.
    """
    reply = navigation_component()(data={'mode': 'receive'}, key='player_navigation_receiver', height=0,
        default={'requested': None}, on_requested_change=lambda: None).requested
    if reply and reply['request_id'] != st.session_state.get('player_navigation_applied'):
        st.session_state['pending_player_navigation'] = reply


def send_player_navigation():
    request = st.session_state.get('player_navigation_signal')
    if request and request['request_id'] == st.session_state.get('player_navigation_applied'):
        request = None
    navigation_component()(data={'mode': 'send', 'request': request},
                           key='player_navigation_sender', height=0)
