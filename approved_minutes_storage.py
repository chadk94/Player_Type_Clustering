"""Browser-local persistence; never publish uploaded rotations to shared storage."""
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4

from approved_minutes import validate_bundle


def new_state(season):
    state = dict(bundle=None, seen_upload=None, error=None, storage_error=None)
    command(state, season, 'restore')
    return state


def command(state, season, action, bundle=None):
    # Keep the JSON as text through JavaScript: parsing/stringifying there turns
    # 18.0 into 18 and would invalidate the export's canonical Python checksum.
    state['request'] = dict(request=str(uuid4()), action=action,
                            storage_key=f'nba-props.approved-minutes.v1.{season}',
                            bundle=json.dumps(bundle, allow_nan=False) if bundle is not None else None)
    state['pending'] = True
    state['storage_error'] = None


def update_upload(state, content, roster, season, clear=False):
    """Uploads win over late restore replies; removal clears persisted data too."""
    fingerprint = sha256(content).hexdigest() if content is not None else None
    if clear or (content is None and state['seen_upload'] is not None):
        state.update(bundle=None, seen_upload=None, error=None)
        command(state, season, 'clear')
    elif content is not None and fingerprint != state['seen_upload']:
        state.update(bundle=None, seen_upload=fingerprint, error=None)
        try:
            state['bundle'] = validate_bundle(content, roster, season)
            command(state, season, 'save', state['bundle'])
        except ValueError as exc:
            # A rejected replacement must not resurrect the older saved rotation.
            state['error'] = str(exc)
            command(state, season, 'clear')


def accept_reply(state, reply, roster, season):
    if not isinstance(reply, dict) or reply.get('request') != state['request']['request']:
        return
    if not state['pending']:
        return
    state['pending'] = False
    if reply.get('status') != 'ok':
        state['storage_error'] = reply.get('status', 'unavailable')
        if state['request']['action'] == 'restore' and reply.get('status') == 'invalid':
            state['error'] = 'Saved minutes could not be read. Clear them or import the approved file again.'
        return
    if state['request']['action'] == 'restore' and reply.get('bundle') is not None:
        try:
            # Browser storage is untrusted input, just like a fresh upload.
            state['bundle'] = validate_bundle(reply['bundle'], roster, season)
        except ValueError as exc:
            state['error'] = f'Saved minutes failed validation: {exc}'


_component = None


def storage_component():
    global _component
    if _component is None:
        from streamlit.components.v2 import component
        _component = component('approved_minutes_storage',
                               js=Path(__file__).with_suffix('.mjs').read_text())
    return _component
