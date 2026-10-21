"""Private import of explicitly approved, game-specific NBA minutes."""
from datetime import date
from hashlib import sha256
import json
import math
import pandas as pd
from import_minutes import TEAM_IDS


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_bundle(content, roster, season):
    """Reject incomplete/tampered exports and unmatched positive-minute players."""
    try:
        bundle = json.loads(content) if isinstance(content, (str, bytes)) else content
        if bundle['schema'] != 'nba_approved_slate_v1': raise ValueError('Unsupported minutes file format')
        payload = {k:v for k,v in bundle.items() if k != 'bundle_id'}
        if digest(payload) != bundle['bundle_id']: raise ValueError('Minutes file checksum does not match; download it again')
        games = bundle['games']; records = bundle['approvals']
        if not games or not records: raise ValueError('The file contains no approved games')
        required = {}; dates = set()
        for game in games:
            day = date.fromisoformat(game['game_date']); dates.add(day)
            if day.year - (day.month < 7) != int(season[:4]): raise ValueError('Minutes file belongs to a different season')
            away,home = game['away_team_id'],game['home_team_id']
            key = f'{day}_{away}_{home}'
            if away == home or not {away,home} <= set(TEAM_IDS.values()) or game['game_key'] != key:
                raise ValueError('Invalid game identifiers')
            for team, opponent in ((away,home),(home,away)):
                if (key,team) in required: raise ValueError('Duplicate game/team')
                required[key,team] = opponent
        if len(dates) != 1: raise ValueError('Import one approved game day at a time')
        known = {int(r.PLAYER_ID):int(r.TEAM_ID) for r in roster.itertuples()}
        seen = set(); all_players = set()
        for record in records:
            key = (record['game_key'], record['team_id']); team = record['team_id']
            if key in seen or key not in required or record['opponent_id'] != required[key]:
                raise ValueError('Duplicate or incorrect team approval')
            seen.add(key)
            if record['game_date'] != str(next(iter(dates))) or TEAM_IDS.get(record['team']) != team:
                raise ValueError('Approval date/team mismatch')
            if record['schema'] != 'nba_approved_minutes_v1' or not record['approved_at'] or not record['approved_by'] or record['approval_version'] < 1:
                raise ValueError('Missing explicit team approval')
            ids = set(); total = 0.; starters = 0
            for player in record['players']:
                pid, minutes = player['player_id'], player['minutes']
                if not isinstance(pid,int) or isinstance(pid,bool) or pid in ids or pid in all_players:
                    raise ValueError('Duplicate or invalid player ID')
                ids.add(pid); all_players.add(pid)
                if isinstance(minutes,bool) or not isinstance(minutes,(int,float)) or not math.isfinite(minutes) or not 0 <= minutes <= 48:
                    raise ValueError('Minutes must be finite numbers from 0 to 48')
                if player['injury_status'] in ('Out','Doubtful') and minutes != 0:
                    raise ValueError('Out/doubtful players must have zero minutes')
                if type(player['starter']) is not bool: raise ValueError('Invalid starter flag')
                if player['starter']:
                    starters += 1
                    if minutes < 8: raise ValueError('Approved starters need at least eight minutes')
                if minutes > 0 and known.get(pid) != team:
                    raise ValueError(f"{record['team']}: {player['player_name']} ({pid}) needs a matching prop-model roster profile before import")
                total += minutes
            if abs(total-240) > 1e-6 or starters != 5:
                raise ValueError('Every approved team needs 240 minutes and five starters')
        if seen != set(required): raise ValueError('Both teams must be approved for every game')
        return bundle
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('Invalid approved-minutes file; use the NBA model’s Download Approved Minutes button') from exc


def for_matchup(bundle, roster, team, opponent, start):
    """Only the exact dated pairing applies; never carry minutes into a rematch."""
    if not bundle or start is None or team not in TEAM_IDS or opponent not in TEAM_IDS: return {}
    timestamp = pd.Timestamp(start)
    if timestamp.tzinfo is None: return {}
    day = str(timestamp.tz_convert('America/New_York').date())
    ids = {TEAM_IDS[team],TEAM_IDS[opponent]}
    matches = [g for g in bundle['games'] if g['game_date'] == day and {g['away_team_id'],g['home_team_id']} == ids]
    if len(matches) != 1: return {}
    key = matches[0]['game_key']
    # A prop-only roster player is explicitly inactive in the approved rotation.
    result = {int(p.PLAYER_ID):0. for p in roster[roster.TEAM_ID.isin(ids)].itertuples()}
    for record in bundle['approvals']:
        if record['game_key'] == key:
            result.update({p['player_id']:float(p['minutes']) for p in record['players'] if p['player_id'] in result})
    return result


def projection_gaps(bundle, projections):
    """Report newly active players without changing their saved profiles or roles."""
    columns = ['Player', 'Team', 'Approved minutes', 'Missing data']
    if not bundle: return pd.DataFrame(columns=columns)
    profiles = projections.set_index('PLAYER_ID')
    rows = []
    for approval in bundle['approvals']:
        for player in approval['players']:
            if player['minutes'] <= 0: continue
            profile = profiles.loc[player['player_id']]
            missing = []
            if str(profile['Stat baseline']).startswith('Unavailable'):
                missing.append('Needs statistical profile')
            elif profile['Status'] == 'Some stat rates unavailable':
                missing.append(profile['Status'])
            for side in ('Offensive', 'Defensive'):
                role = profile[f'{side} archetype']
                if pd.isna(role) or str(role).strip().lower() in ('', 'unclassified', 'uncategorized'):
                    missing.append(f'{side.lower()} archetype')
            if missing:
                rows.append([profile['Player'], profile['Team'], player['minutes'], '; '.join(missing)])
    return pd.DataFrame(rows, columns=columns).sort_values(['Team', 'Player']).reset_index(drop=True)


def game_minutes_caption(projections, teams):
    summaries = []
    for team in teams:
        rows = projections[projections.Team.eq(team)]
        sources = ', '.join(sorted(rows['Minutes source'].dropna().unique())) or 'Unavailable'
        total = rows['Projected minutes'].sum(min_count=1)
        allocation = f'{total:.1f} total' if pd.notna(total) else 'total unavailable'
        summaries.append(f'{team}: {sources} ({allocation})')
    return 'Minutes for this game — ' + ' · '.join(summaries)


def render_import(ui, roster, season):
    import streamlit as st
    from approved_minutes_storage import new_state, update_upload, accept_reply, storage_component
    state = st.session_state.setdefault('minutes_import_state', new_state(season))
    def clear_saved():
        st.session_state['minutes_clear_requested'] = True
        st.session_state['minutes_upload_version'] = st.session_state.get('minutes_upload_version', 0) + 1
    with ui.expander('Import Approved Minutes'):
        ui.write('Upload the JSON downloaded after approving every team in the NBA model. Matching games use these minutes across player, matchup, sportsbook and team projections.')
        upload = ui.file_uploader('Approved minutes from NBA model', type=['json'],
                                  key=f"approved_game_minutes_{st.session_state.get('minutes_upload_version', 0)}")
        ui.button('Clear saved minutes', on_click=clear_saved, key='clear_saved_minutes')
        ui.caption('Saved privately in this browser and restored after refresh or reopening the app. A different browser needs its own import. Nothing is published or written to the baseline workbook. Import a new file after changing approvals.')
    update_upload(state, None if upload is None else upload.getvalue(), roster, season,
                  clear=st.session_state.pop('minutes_clear_requested', False))
    reply = storage_component()(data=state['request'], key='approved_minutes_browser_storage',
                                default={'reply': None}, height=1, on_reply_change=lambda: None).reply
    accept_reply(state, reply, roster, season)
    if state['pending']:
        ui.info('Restoring saved minutes…' if state['request']['action'] == 'restore' else 'Saving your minutes selection…')
        st.stop()  # Never briefly rank props using workbook minutes during restore.
    if state['storage_error']:
        ui.warning('Browser storage is unavailable. Minutes can only stay in this session; allow site storage to retain them after refresh.')
    if state['error']:
        ui.error(state['error'])
        st.stop()  # An invalid replacement must not silently use the old workbook.
    bundle = state['bundle']
    if bundle:
        day = bundle['games'][0]['game_date']
        suffix = 'Active for this session only.' if state['storage_error'] else 'Saved in this browser.'
        ui.success(f"Minutes source: approved file · {day} · {len(bundle['approvals'])} teams. {suffix}")
        warnings = [f"{a['team']}: {'; '.join(a['warnings'])}" for a in bundle['approvals'] if a['warnings']]
        if warnings: ui.warning('Approved with injury-report limitations: '+' · '.join(warnings))
    else:
        ui.warning('No approved minutes file is loaded. Projections use workbook/season-average minutes. Open Import Approved Minutes to use your game rotations.')
    identifier = (bundle or {}).get('bundle_id')
    if st.session_state.get('approved_minutes_bundle') != identifier:
        st.session_state['approved_minutes_bundle'] = identifier
        st.session_state.pop('team_projection_snapshots', None)
        # A new approval supersedes the previous preview slider edits.
        for key in list(st.session_state):
            if key.startswith('custom_minutes_'): st.session_state.pop(key, None)
    return bundle
