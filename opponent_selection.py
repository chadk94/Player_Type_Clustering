"""Opponent defaults that follow the schedule without overwriting manual choices."""
import streamlit as st


def sync_opponent(state, team, opponents, upcoming):
    """Run before the widget, keeping automatic and manual selection separate."""
    if state.get('projection_opponent_team') != team:
        state['projection_opponent_manual'] = False
    state['projection_opponent_team'] = team
    current = state.get('projection_opponent')
    if state.get('projection_opponent_manual', False) and current in opponents:
        return
    state['projection_opponent_manual'] = False
    default = upcoming.get('Opponent') if upcoming else None
    state['projection_opponent'] = default if default in opponents else None


def mark_manual_opponent():
    # Clearing the dropdown restores automatic selection on the next render.
    st.session_state['projection_opponent_manual'] = st.session_state.get('projection_opponent') is not None


def reset_opponent():
    st.session_state['projection_opponent_manual'] = False


def render_opponent_selector(ui, team, opponents, upcoming):
    sync_opponent(st.session_state, team, opponents, upcoming)
    opponent = ui.selectbox('Opponent', opponents, index=None,
                            key='projection_opponent', placeholder='Choose opponent',
                            on_change=mark_manual_opponent)
    ui.button('Use next opponent', key='reset_projection_opponent',
              on_click=reset_opponent, disabled=not upcoming,
              help='Restore the next scheduled opponent for this team.')
    return opponent
