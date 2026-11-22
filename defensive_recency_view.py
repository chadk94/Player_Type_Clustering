"""On-demand comparison and private, prospective browser history."""
import base64
from hashlib import sha256
import json
from pathlib import Path
import zlib

import pandas as pd
import streamlit as st

from defensive_recency import (VERSION, compare_game, snapshot, valid_record,
                               evaluate, error_summary)


@st.cache_resource
def history_component():
    from streamlit.components.v2 import component
    # Reuse the tested, append-only transport with an entirely separate store.
    source = Path(__file__).with_name('points_tracking.mjs').read_text()
    old_name = 'nba-props-points-probability-history-v1'
    assert source.count(old_name) == 1
    return component('defensive_recency_trial_v1', js=source.replace(
        old_name, 'nba-props-defensive-recency-trial-v1'))


def accept_reply(state, request, reply):
    if not isinstance(reply, dict) or reply.get('request') != request:
        return
    if state.get('ack') == request and state.get('status') == reply.get('status'):
        return
    try:
        if reply.get('encoding'):
            if reply['encoding'] != 'gzip-base64' or len(reply['payload']) > 8000000:
                raise ValueError('Invalid history encoding')
            decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
            raw = decoder.decompress(base64.b64decode(reply['payload'], validate=True), 128000001)
            if len(raw) > 128000000 or not decoder.eof or decoder.unused_data:
                raise ValueError('Invalid history transfer')
            reply = json.loads(raw)
        if reply.get('status') != 'ok' or reply.get('request') != request:
            raise ValueError('History unavailable')
        if not reply.get('full', True) and reply.get('base') != state.get('ack'):
            return
        records = {} if reply.get('full', True) else dict(state.get('records', {}))
        invalid = 0 if reply.get('full', True) else state.get('invalid', 0)
        for body in reply['records']:
            try:
                record = json.loads(body) if isinstance(body, str) and len(body) <= 50000 else None
                if valid_record(record):
                    records.setdefault(record['id'], record)
                else:
                    invalid += 1
            except (ValueError, TypeError):
                invalid += 1
        state.update(records=records, status='ok', ack=request, invalid=invalid)
    except (ValueError, KeyError, TypeError, zlib.error):
        state['status'] = 'unavailable'


@st.fragment
def render_history(ui, observation, logs, book):
    key = 'defensive_recency_history'
    state = st.session_state.setdefault(key, {'records': {}, 'status': 'loading'})
    revision = st.session_state.get(key + '_revision', 0)
    request = sha256(json.dumps([VERSION, observation and observation['id'], revision]).encode()).hexdigest()
    restore = not state.get('ack') or state.get('status') != 'ok' or revision != state.get('revision')

    def receive():
        reply = st.session_state.get(key + '_sync', {}).get('reply')
        accept_reply(state, request, reply)
        if state.get('ack') == request:
            state['revision'] = revision

    reply = history_component()(data={'request': request, 'base': state.get('ack'), 'restore': restore,
        'season': VERSION, 'records': [] if observation is None else [
            {'id': observation['id'], 'season': VERSION, 'body': json.dumps(observation, allow_nan=False)}],
        'results': []}, key=key + '_sync', height=0, on_reply_change=receive).reply
    accept_reply(state, request, reply)
    if state.get('ack') == request:
        state['revision'] = revision
    if state['status'] == 'unavailable':
        ui.warning('This comparison could not be saved to this browser. Your main projections are unaffected.')
    elif state['status'] == 'loading':
        ui.caption('Opening this browser’s defensive-recency trial history…')
    else:
        ui.caption(f"{len(state['records'])} comparison snapshots saved in this browser.")
        if observation is not None and observation['id'] not in state['records']:
            ui.caption('Saving this pregame comparison…')
    if state.get('invalid'):
        ui.warning(f"{state['invalid']} invalid history records were excluded.")
    ui.caption('Only comparisons you open before tip-off are recorded. Results update from completed box scores on a new visit or Refresh live data. Nothing runs while the app is closed. Clearing site data removes this history; download a backup below.')
    ui.button('Reload trial history', key=key + '_reload', on_click=lambda: st.session_state.update(
        {key + '_revision': st.session_state.get(key + '_revision', 0) + 1}))
    if not state['records']:
        return
    record_ids = tuple(state['records'])
    cached = st.session_state.get(key + '_evaluation')
    if cached is None or cached[0] is not logs or cached[1] != record_ids:
        cached = (logs, record_ids, evaluate(list(state['records'].values()), logs))
        st.session_state[key + '_evaluation'] = cached
    history = cached[2].loc[lambda f: f.Book.eq(book)]
    ui.caption('Evaluation uses one latest saved pregame comparison per game and sportsbook. '
               'Repeated views are not independent trials. The results below use ' + book + '.')
    documented = ui.toggle('Evaluate documented opponent rotation changes only', key=key + '_documented')
    summary = error_summary(history, documented)
    if summary.empty:
        ui.info('No completed, verified results in this sample yet.')
    else:
        ui.dataframe(summary.style.format(precision=3), hide_index=True, width='stretch')
        ui.caption('Lower RMSE and MAE mean better forecasts. Small early samples are descriptive; this is not a betting-return test.')
    ui.download_button('Download trial predictions and results', history.to_csv(index=False),
                       'defensive_recency_results.csv', 'text/csv', key=key + '_csv')
    ui.download_button('Back up frozen trial snapshots', json.dumps(list(state['records'].values()), indent=2),
                       'defensive_recency_snapshots.json', 'application/json', key=key + '_json')


def render_trial(ui, final, game, book, load_rates, logs):
    with ui.expander('Defensive recency trial — comparison only'):
        ui.caption('A 10-game half-life for defensive assist, three-attempt and offensive-rebound rates. '
                   'Current projections, rankings, probabilities and archetype effects stay unchanged.')
        if not ui.toggle('Show defensive-recency comparison', key='show_defensive_recency_trial'):
            return
        ui.caption('The comparison starts after eight verified team games. It keeps the existing season blend, '
                   'minutes, points totals and shooting accuracy. Rebounds and PRA update within the comparison.')
        observation = None
        opponents = {game['Home']: game['Away'], game['Away']: game['Home']}
        try:
            totals, players = compare_game(final, load_rates(), opponents)
        except (ValueError, KeyError, TypeError) as exc:
            ui.info('Comparison unavailable: ' + str(exc))
        else:
            ui.dataframe(totals.style.format({'Current': '{:.2f}', 'Defensive recency': '{:.2f}', 'Change': '{:+.2f}'}),
                         hide_index=True, width='stretch', row_height=36)
            if ui.toggle('Show player impact', key='defensive_recency_players'):
                search = ui.text_input('Find a player in the comparison', key='defensive_recency_search')
                display = players[players.Player.str.contains(search, case=False, regex=False)] if search else players
                ui.dataframe(display.style.format({'Current': '{:.2f}', 'Defensive recency': '{:.2f}', 'Change': '{:+.2f}'}),
                             hide_index=True, width='stretch', row_height=36, height=400)
            now = pd.Timestamp.now(tz='UTC')
            note_key = 'defensive_recency_notes_' + sha256(game['key'].encode()).hexdigest()[:12]
            notes = st.session_state.setdefault(note_key, {})
            history = st.session_state.get('defensive_recency_history', {}).get('records', {})
            for record in sorted(history.values(), key=lambda r: r['captured_at'], reverse=True):
                if (record['home'] == game['Home'] and record['away'] == game['Away'] and
                        pd.Timestamp(record['start']) == pd.Timestamp(game['GameTimeUTC'])):
                    for team, note in record['notes'].items():
                        notes.setdefault(team, dict(note))
            if now < pd.Timestamp(game['GameTimeUTC']):
                with ui.form(note_key + '_form'):
                    ui.caption('Optional: document a defensive starter/rotation change before the game. '
                               'These games will be evaluated separately. Include the effective date in the note.')
                    team = ui.selectbox('Team with the change', list(opponents), key=note_key + '_team')
                    note = ui.text_input('Change and effective date', max_chars=500, key=note_key + '_note')
                    source = ui.text_input('Source link for the change', max_chars=1000, key=note_key + '_source')
                    if ui.form_submit_button('Save documented change'):
                        if note.strip() and source.startswith(('https://', 'http://')):
                            notes[team] = {'change': note.strip(), 'source': source.strip()}
                        else:
                            ui.warning('Add the change, its date and a source link before saving the tag.')
                for team, note in notes.items():
                    ui.caption(f"{team}: {note['change']}")
                observation = snapshot(totals, game, book, logs, final, notes, now=now)
                if observation is None:
                    ui.caption('This preview cannot be saved as a verified pregame comparison.')
            else:
                ui.caption('Tip-off has passed. No new trial snapshot will be recorded for this game.')
        render_history(ui, observation, logs, book)
