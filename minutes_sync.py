"""Read the owner's private NBA-model approvals without changing baseline minutes."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from time import sleep

import requests
import streamlit as st
from background_feed import BackgroundFeed


FEED_TIMEOUT = (4, 12)
FEED_ATTEMPTS = 2
RETRY_DELAY_SECONDS = .5
RETRYABLE_HTTP = {408, 429, 500, 502, 503, 504}
FEED_REFRESH_SECONDS = 10


def _read_response(url, key):
    """Retry one transient read failure; never forward the key on redirects."""
    for attempt in range(FEED_ATTEMPTS):
        try:
            response = requests.post(url, json={'action':'minutes_read','key':key},
                                     timeout=FEED_TIMEOUT, allow_redirects=False)
            if response.status_code in (302,303):
                location = response.headers.get('Location','')
                if not location.startswith('https://script.googleusercontent.com/'):
                    raise ValueError('Google needs the owner to finish connecting the minutes feed.')
                response = requests.get(location, timeout=FEED_TIMEOUT, allow_redirects=False)
            response.raise_for_status()
            try:
                return response.json()
            except ValueError as exc:
                raise ValueError('Google returned an unreadable minutes response. No new minutes were imported.') from exc
        except requests.RequestException as exc:
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            transient = isinstance(exc, (requests.Timeout, requests.ConnectionError)) or status in RETRYABLE_HTTP
            if transient and attempt + 1 < FEED_ATTEMPTS:
                sleep(RETRY_DELAY_SECONDS)
                continue
            if isinstance(exc, requests.Timeout):
                message = 'Google’s minutes feed timed out after retrying.'
            elif status in (401,403):
                message = 'Google denied access to the minutes feed. Check the private connection and deployment access.'
            elif status in RETRYABLE_HTTP:
                message = f'Google’s minutes feed is temporarily unavailable (HTTP {status}).'
            else:
                message = 'The connection to Google’s minutes feed failed.'
            # Request exceptions may contain private redirect URLs. Never display them.
            raise ValueError(message) from exc


def read_feed(connection):
    url, _, key = connection.partition('#')
    if not re.fullmatch(r'https://script.google.com/macros/s/[A-Za-z0-9_-]+/exec', url) or not re.fullmatch(r'[a-f0-9]{64}', key):
        raise ValueError('Use the private minutes connection from the NBA model.')
    answer = _read_response(url, key)
    if not isinstance(answer, dict):
        raise ValueError('The minutes feed returned an unsupported format.')
    if not answer.get('ok'):
        raise ValueError('The private minutes connection was rejected. Reconnect the NBA model.')
    snapshot = answer.get('snapshot')
    if snapshot is None:
        return {'schema':'nba_minutes_feed_v1','bundles':[]}
    if not isinstance(snapshot, str) or sha256(snapshot.encode()).hexdigest() != answer.get('digest'):
        raise ValueError('Minutes failed the transfer check. Waiting for a complete update.')
    try:
        data = json.loads(snapshot)
    except ValueError as exc:
        raise ValueError('The minutes feed returned an unsupported format.') from exc
    if not isinstance(data, dict) or data.get('schema') != 'nba_minutes_feed_v1' or not isinstance(data.get('bundles'), list):
        raise ValueError('The minutes feed returned an unsupported format.')
    return data


def validate_feed(feed, roster, season):
    from approved_minutes import digest, validate_bundle, RosterProfileError
    games, approvals, blocked, dates = [], [], [], set()
    for raw in feed['bundles']:
        # Verify the complete original checksum and rotation constraints before
        # isolating profile problems. Never split an unverified payload.
        bundle = validate_bundle(raw, roster, season, check_profiles=False)
        day = bundle['games'][0]['game_date']
        if day in dates:
            raise ValueError('The minutes feed returned duplicate dates.')
        dates.add(day)
        for game in bundle['games']:
            records = [a for a in bundle['approvals'] if a['game_key'] == game['game_key']]
            candidate = dict(schema='nba_approved_slate_v1', games=[game], approvals=records)
            candidate['bundle_id'] = digest(candidate)
            try:
                validate_bundle(candidate, roster, season, allow_roster_moves=True)
            except RosterProfileError as exc:
                blocked.append({**game, 'reason': str(exc)})
                continue
            games.append(game)
            approvals.extend(records)
    if not games and not blocked:
        return None
    result = dict(schema='nba_linked_minutes_v1', games=games, approvals=approvals, blocked_games=blocked)
    result['bundle_id'] = digest(result)
    return result


@st.cache_resource
def connection_component():
    from streamlit.components.v2 import component
    return component('nba_minutes_connection', js=Path(__file__).with_suffix('.mjs').read_text())


def _request_sync_retry():
    st.session_state.setdefault('nba_minutes_link', {}).pop('checked', None)


@st.cache_resource(max_entries=8, show_spinner=False)
def minutes_feed_worker(connection_digest):
    # Cache keys contain only a hash, never the private connection URL/key.
    return BackgroundFeed()


@st.fragment(run_every='5s')
def render_sync(roster, season):
    state = st.session_state.setdefault('nba_minutes_link', {})
    reply = connection_component()(data={'connected':bool(state.get('connection')),
                                        'refresh_seconds':FEED_REFRESH_SECONDS},
        key='nba_minutes_connection', default={'connection':None}, height='content',
        on_connection_change=lambda: None).connection
    if not reply or not reply.get('ready'):
        st.stop()
    connection = reply.get('connection')
    first_restore = not state.get('ready')
    changed = state.get('connection') != connection
    previous_id = (state.get('bundle') or {}).get('bundle_id')
    waiting_for_first_result = state.get('pending') and not state.get('bundle')
    now = datetime.now(timezone.utc)
    if changed:
        state.clear()
        state['connection'] = connection
    state['ready'] = True
    if reply.get('error'):
        st.caption(reply['error'])
    if connection:
        worker = minutes_feed_worker(sha256(connection.encode()).hexdigest())
        snapshot = worker.read(lambda: read_feed(connection), refresh_after=FEED_REFRESH_SECONDS,
                               force=changed or not state.get('checked'))
        state['pending'] = snapshot.pending
        state['checked'] = now.timestamp()
    else:
        snapshot = None
    if snapshot and snapshot.revision and snapshot.revision != state.get('feed_revision'):
        try:
            if snapshot.error is not None:
                message = (str(snapshot.error) if isinstance(snapshot.error, ValueError)
                           else 'The minutes feed could not be read. Retrying shortly.')
                raise ValueError(message)
            data = snapshot.value
            state['bundle'] = validate_feed(data, roster, season)
            state['published_at'] = data.get('published_at')
            state['error'] = None
            state['success_at'] = snapshot.checked_at.strftime('%b %d, %I:%M %p UTC')
        except (ValueError, KeyError, TypeError) as exc:
            state['error'] = str(exc)
        state['feed_revision'] = snapshot.revision
    if state.get('pending'):
        st.caption('Checking NBA model minutes in the background…' +
                   (' Your last verified rotations remain in use.' if state.get('bundle') else ''))
    if state.get('error'):
        note = (' Using the last verified approvals (last successful sync: '+state.get('success_at','unknown')+').'
                if state.get('bundle') else ' No verified minutes are available in this session yet.')
        st.warning('Automatic minutes sync needs attention. '+state['error']+note+' Retrying automatically.')
        st.button('Retry minutes sync now', key='retry_nba_minutes_sync', on_click=_request_sync_retry)
    blocked = (state.get('bundle') or {}).get('blocked_games', [])
    if blocked:
        st.warning(f'{len(blocked)} game(s) need a roster profile check. Only those games are paused; verified games remain available.')
        with st.expander('Minutes sync details'):
            for game in blocked:
                st.write(f"{game['game_date']}: {game['reason']}")
    if connection and state.get('bundle'):
        n = len(state['bundle']['approvals'])
        st.caption(f"NBA model linked · {n} approved team rotations · Auto-refresh every {FEED_REFRESH_SECONDS}s · Last checked {state.get('success_at')}.")
    elif connection and not state.get('error') and not state.get('pending'):
        st.info('NBA model linked. Approve both teams in a game to sync its minutes.')
    current_id = (state.get('bundle') or {}).get('bundle_id')
    if first_restore or changed or previous_id != current_id or (waiting_for_first_result and not state.get('pending')):
        # Only an actual rotation change refreshes projections. Timer checks stay in this panel.
        from approved_minutes import apply_bundle_change
        apply_bundle_change(st.session_state, state.get('bundle'))
        if previous_id and current_id and previous_id != current_id and not changed:
            st.toast('New approved minutes received. Refreshing projections.', icon='✅')
        st.rerun()
