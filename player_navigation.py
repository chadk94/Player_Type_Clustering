"""Open a contribution row in the player preview, retaining its exact game."""
from pathlib import Path

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
    return dict(player_id=int(row['PLAYER_ID']), team=team,
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
    state['player_navigation_request'] = dict(id=revision, heading=f'{name} vs {request["opponent"]}')


def selected_game_context(state, team, opponent):
    """Manual team/opponent changes return to the ordinary schedule preview."""
    context = state.get('projection_game_context')
    if context and (context['team'], context['opponent']) != (team, opponent):
        state.pop('projection_game_context', None)
        return None
    if context:
        return dict(Opponent=opponent, GameTimeUTC=pd.Timestamp(context['start']))
    return None


_component = None


@st.fragment
def render_player_navigation():
    # Streamlit 1.53 tabs are client-side. A one-shot component activates the
    # existing tab; its acknowledgement reruns only this tiny fragment.
    global _component
    if _component is None:
        from streamlit.components.v2 import component
        _component = component('player_navigation', js=Path(__file__).with_suffix('.mjs').read_text())
    request = st.session_state.get('player_navigation_request')
    reply = _component(data=request, key='player_navigation', height=0,
                       default={'opened': None}, on_opened_change=lambda: None).opened
    if request and reply == request['id']:
        st.session_state.pop('player_navigation_request', None)
