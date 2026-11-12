"""Prospective, browser-persisted experimental probabilities and result grading."""
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import streamlit as st

from points_probability import MODEL_VERSION, PROBABILITY_COLUMNS, model_profile
from prop_distributions import COUNT_STATS, COUNT_VERSION


def record_id(record):
    fields = {k: v for k, v in record.items() if k not in ('id', 'captured_at', 'page')}
    return sha256(json.dumps(fields, sort_keys=True, allow_nan=False).encode()).hexdigest()


def snapshot(row, event, bookmaker, context, page, mode='Standard', now=None):
    """Freeze the exact line/mean once; all later changes get a different ID."""
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    if row.get('Probability status') != 'Experimental':
        return None
    try:
        start = pd.Timestamp(event['commence_time'])
        updated = pd.Timestamp(row['Updated UTC'])
        if not now < start or not -60 <= (now-updated).total_seconds() <= 900:
            return None
        pid = int(row['PLAYER_ID'])
        past = model_profile()['scoring'].get(str(pid), {})
        espn = str(context.get(pid, {}).get('ESPN_PLAYER_ID', past.get('espn_id', '')))
        if not espn.isdigit():
            return None
        stat = row.get('Stat', 'PTS')
        record = dict(schema=1 if stat == 'PTS' else 2, season=model_profile()['season'], model=row['Probability model'],
            profile=row['Probability profile'], event=str(event['id']), start=start.isoformat(),
            player_id=pid, espn_id=espn, player=str(row['Player']), team=str(row['Team']),
            opponent=str(row['Opponent']), book=bookmaker, line=float(row['Book line']),
            over_odds=float(row['Over odds']), under_odds=float(row['Under odds']),
            quote_at=updated.isoformat(), captured_at=now.isoformat(), page=page, mode=mode,
            mean=float(row['Projected']), minutes=float(row['Projected minutes']),
            minutes_source=str(row.get('Minutes source', 'Unavailable')),
            stat_baseline=str(row.get('Stat baseline', 'Unavailable')),
            variability_source=str(row['Variability source']),
            over=float(row[PROBABILITY_COLUMNS[0]]), under=float(row[PROBABILITY_COLUMNS[1]]),
            push=float(row[PROBABILITY_COLUMNS[2]]))
        if stat == 'PTS':
            record.update(phi=float(row['Variability phi']), ftm=float(row['Expected FTM']),
                          twos=float(row['Expected 2PM']), threes=float(row['Expected 3PM']))
        else:
            record.update(stat=stat, distribution=row['Distribution'], dispersion=float(row['Dispersion']))
        record['id'] = record_id(record)
        return record if valid_record(record) else None
    except (ValueError, KeyError, TypeError, OverflowError):
        return None


def valid_record(r):
    try:
        if not isinstance(r, dict) or r['id'] != record_id(r):
            return False
        if r['season'] != model_profile()['season'] or len(json.dumps(r)) > 6000:
            return False
        if not str(r['espn_id']).isdigit() or not 0 < int(r['player_id']):
            return False
        numeric = ['mean', 'minutes', 'line', 'over', 'under', 'push']
        if r['schema'] == 1 and r['model'] == MODEL_VERSION and r.get('stat', 'PTS') == 'PTS':
            numeric.extend(['ftm', 'twos', 'threes', 'phi'])
            if abs(r['ftm'] + 2*r['twos'] + 3*r['threes'] - r['mean']) > 1e-7:
                return False
        elif r['schema'] == 2 and r['model'] == COUNT_VERSION and r['stat'] in COUNT_STATS:
            if not math.isfinite(r['dispersion']) or r['distribution'] not in (
                    'poisson', 'pooled', 'player', 'pooled_mixture', 'player_mixture', 'binomial6'):
                return False
        else:
            return False
        for k in numeric:
            if not isinstance(r[k], (float, int)) or not math.isfinite(r[k]) or r[k] < 0:
                return False
        if r['minutes'] > 48 or max(r['over'], r['under'], r['push']) > 1 or abs(r['over']+r['under']+r['push']-1) > 1e-8:
            return False
        if r['line']*2 != round(r['line']*2):
            return False
        for k in ['over_odds', 'under_odds']:
            if not np.isfinite(r[k]) or abs(r[k]) < 100:
                return False
        captured, start, quote = [pd.Timestamp(r[k]) for k in ['captured_at', 'start', 'quote_at']]
        return (captured.tzinfo is not None and start.tzinfo is not None and quote.tzinfo is not None
                and captured < start and -60 <= (captured-quote).total_seconds() <= 900
                and r['mode'] in ['Standard', 'Manual minutes', 'Opponent adjustment off', 'Manual minutes + opponent adjustment off'])
    except (ValueError, KeyError, TypeError, OverflowError):
        return False


def decode_records(values):
    """Preserve Python's exact numeric JSON through the browser round trip."""
    records = []
    if not isinstance(values, list):
        return records
    for value in values:
        try:
            if not isinstance(value, str) or len(value) > 6000:
                continue
            record = json.loads(value)
            if valid_record(record):
                records.append(record)
        except (ValueError, TypeError):
            continue
    return records


def grade_records(records, logs):
    """Only unique, completed ESPN game/player matches. Missing/DNP is never zero."""
    if not records or logs.empty:
        return []
    rows = logs.copy()
    rows['_start'] = pd.to_datetime(rows.GAME_DATE, utc=True)
    rows['ESPN_PLAYER_ID'] = rows.ESPN_PLAYER_ID.astype(str)
    results = []
    grouped = {key: g for key, g in rows.groupby(['Team', 'Opponent'])}
    @lru_cache(maxsize=256)
    def game_players(team, opponent, start, season):
        pool = grouped.get((team, opponent))
        if pool is None:
            return {}
        year = int(season[:4])
        match = pool.loc[(pool['_start']-pd.Timestamp(start)).abs() <= pd.Timedelta(minutes=15)]
        if (match.GAME_ID.nunique() != 1 or (match['_start'] < pd.Timestamp(f'{year}-10-01',tz='UTC')).any()
                or (match['_start'] >= pd.Timestamp(f'{year+1}-07-01',tz='UTC')).any()):
            return {}
        unique = match.loc[~match.ESPN_PLAYER_ID.duplicated(keep=False)]
        return unique.set_index('ESPN_PLAYER_ID').to_dict('index')

    for r in records:
        if not valid_record(r):
            continue
        player = game_players(r['team'],r['opponent'],r['start'],r['season']).get(r['espn_id'])
        if player is None:
            continue
        stat = r.get('stat','PTS')
        parts = ('PTS','REB','AST') if stat == 'PRA' else ('FG3M' if stat == '3PM' else stat,)
        try:
            values = [float(player[key]) for key in parts]
            minutes = float(player['MIN'])
        except (ValueError, KeyError, TypeError):
            continue
        if not math.isfinite(minutes) or minutes <= 0 or any(not v.is_integer() or v < 0 for v in values):
            continue
        actual = sum(values)
        outcome = 'Over' if actual > r['line'] else 'Under' if actual < r['line'] else 'Push'
        result = dict(id=r['id'], game=str(player['GAME_ID']), actual_minutes=minutes,
                      outcome=outcome, source='ESPN completed regular-season box score')
        result['points' if stat == 'PTS' else 'actual'] = actual
        results.append(result)
    return results


def history_table(records, results):
    graded = {}
    for result in results if isinstance(results, list) else []:
        try:
            actual, minutes = float(result.get('actual',result.get('points'))), float(result['actual_minutes'])
            if (np.isfinite([actual, minutes]).all() and actual >= 0 and actual.is_integer()
                    and minutes > 0 and str(result['game']).isdigit()):
                graded[result['id']] = {**result, 'actual': actual, 'actual_minutes': minutes}
        except (ValueError, KeyError, TypeError, OverflowError):
            continue
    rows = []
    for r in records:
        if not valid_record(r):
            continue
        result = graded.get(r['id'], {})
        actual = result.get('actual', np.nan)
        stat = r.get('stat', 'PTS')
        outcome = ('Over' if actual > r['line'] else 'Under' if actual < r['line'] else 'Push') if np.isfinite(actual) else 'Pending / no verified appearance'
        rows.append({**r, 'stat':stat, 'actual':actual, 'actual_points':actual if stat == 'PTS' else np.nan, 'actual_minutes': result.get('actual_minutes', np.nan),
                     'outcome': outcome, 'result_game': result.get('game', '')})
    return pd.DataFrame(rows)


def evaluation_rows(frame):
    """One latest standard snapshot per game/player/stat/book, selected without outcomes."""
    if frame.empty:
        return frame
    standard = frame.loc[frame['mode'].eq('Standard')].copy()
    standard['_captured'] = pd.to_datetime(standard.captured_at, utc=True)
    return standard.sort_values(['_captured', 'id']).drop_duplicates(
        ['season', 'model', 'profile', 'event', 'player_id', 'book', 'stat'], keep='last').drop(columns='_captured')


_component = None


def tracking_component():
    global _component
    if _component is None:
        from streamlit.components.v2 import component
        _component = component('points_probability_tracking', js=Path(__file__).with_suffix('.mjs').read_text())
    return _component


def accept_tracking_reply(state, key, request, reply):
    """Validate the browser acknowledgement before the normal fragment rerun."""
    if not isinstance(reply, dict) or reply.get('request') != request:
        return
    if state[key].get('acknowledged') == request and state[key]['status'] == reply.get('status'):
        return
    if reply.get('status') == 'ok' and isinstance(reply.get('records'), list):
        records = decode_records(reply['records'])
        state[key] = {'records': records, 'results': reply.get('results', []),
                      'status': 'ok', 'acknowledged': request, 'invalid_records': len(reply['records']) - len(records)}
    elif reply.get('status') == 'unavailable':
        state[key]['status'] = 'unavailable'


@st.fragment
def render_probability_tracking(ui, observations, logs, page):
    """Storage acknowledgements only rerun this small panel, never refresh feeds."""
    key = f'points_tracking_{page}'
    saved = st.session_state.setdefault(key, {'records': [], 'results': [], 'status': 'loading'})
    records = saved.get('records', [])
    # A fragment reuses the same verified log snapshot until the dashboard refreshes.
    # History is private to this browser session. An arriving box or forecast regrades it.
    record_ids = tuple(r['id'] for r in records)
    graded = st.session_state.get(key+'_graded')
    if graded is None or graded[0] is not logs or graded[1] != record_ids:
        graded = (logs, record_ids, grade_records(records, logs))
        st.session_state[key+'_graded'] = graded
    results = graded[2]
    request = sha256(json.dumps({'ids': [r['id'] for r in observations], 'results': results,
                                'revision': st.session_state.get(key+'_revision', 0)}, sort_keys=True).encode()).hexdigest()
    def on_reply_change():
        # Component callbacks run before the automatically scheduled fragment.
        # Restore first so that this rerun can grade and persist restored history.
        # An explicit nested st.rerun(scope='fragment') fails on parent/app reruns.
        reply = st.session_state.get(key+'_component', {}).get('reply')
        accept_tracking_reply(st.session_state, key, request, reply)

    reply = tracking_component()(data={'request': request, 'season': model_profile()['season'],
        'records': [{'id': r['id'], 'season': r['season'], 'body': json.dumps(r, allow_nan=False)}
                    for r in observations], 'results': results}, key=key+'_component', height=0,
        default={'reply': None}, on_reply_change=on_reply_change).reply
    accept_tracking_reply(st.session_state, key, request, reply)
    saved = st.session_state[key]
    if saved['status'] == 'unavailable':
        ui.warning('Prop probability tracking could not save to this browser. Open the tracking panel to retry; the projection itself is unaffected.')
    with ui.expander('Experimental prop probability tracking'):
        ui.caption('Experimental player-prop probabilities. Extreme estimates and low-minute players are less reliable. '
                   'Over + Under + Push = 100%; integer lines include a separate push probability. '
                   'Projected averages, edge percentages and the 5% play rule are unchanged.')
        if saved['status'] == 'unavailable':
            ui.warning('Browser storage is unavailable. Predictions are not confirmed saved. Enable site storage and retry.')
        elif saved['status'] == 'loading':
            ui.caption('Opening this browser’s prediction history…')
        else:
            ui.caption(f"{len(saved['records'])} snapshots saved in this browser. Loading a verified pregame prop comparison records it automatically, including lines below the 5% edge threshold. "
                       'This covers loaded game comparisons, not every NBA game. Clearing site data removes it; download a backup.')
            if saved.get('invalid_records'):
                ui.warning(f"{saved['invalid_records']} saved records failed validation and were excluded. Download a backup of the valid history.")
        ui.caption('Results update from completed box scores on Refresh live data or a new visit. Nothing runs while the app is closed. '
                   'No verified appearance stays ungraded. Manual-minute and opponent-off previews are saved separately from the standard evaluation.')
        ui.button('Reload saved history', key=key+'_reload', on_click=lambda: st.session_state.update(
            {key+'_revision': st.session_state.get(key+'_revision', 0)+1}))
        if not saved.get('records') or not ui.toggle('Show history and downloads', key=key+'_show_history'):
            return
        frame = history_table(saved['records'], saved.get('results', []))
        if not frame.empty:
            selected = evaluation_rows(frame)
            stat_filter = ui.selectbox('Evaluate stat', ['All', *sorted(selected.stat.unique())], key=key+'_stat')
            sample = selected if stat_filter == 'All' else selected.loc[selected.stat.eq(stat_filter)]
            settled = sample.loc[sample.actual.notna() & sample.outcome.ne('Push')]
            ui.caption(f'{len(sample)} standard player/stat/game/book predictions · {len(settled)} graded non-push results. '
                       'Evaluation uses the latest saved pregame standard snapshot for each player/stat/game/book. Repeated views do not count as independent trials.')
            if not settled.empty:
                p = settled.over / (settled.over + settled.under)
                y = settled.outcome.eq('Over').astype(float)
                ui.caption(f'Over probability (excluding pushes): {p.mean():.1%} · Observed overs: {y.mean():.1%} · '
                           f'Brier error: {((p-y)**2).mean():.4f} (lower is better). Early samples are descriptive only.')
            display = frame[['player', 'team', 'opponent', 'start', 'book', 'stat', 'line', 'mean', 'minutes',
                             'over', 'under', 'push', 'mode', 'outcome', 'actual', 'captured_at']].rename(columns={
                'player':'Player','team':'Team','opponent':'Opponent','start':'Game time (UTC)',
                'book':'Sportsbook','stat':'Stat','line':'Line','mean':'Projected','minutes':'Projected minutes',
                'over':'P(over)','under':'P(under)','push':'P(push)','mode':'Preview type',
                'outcome':'Result','actual':'Actual','captured_at':'Saved at (UTC)'})
            ui.dataframe(display.tail(100).iloc[::-1].style.format(
                {**{key:'{:.1%}' for key in ['P(over)','P(under)','P(push)']},'Projected':'{:.2f}',
                 'Projected minutes':'{:.1f}'}, na_rep='—'), hide_index=True, width='stretch')
            ui.download_button('Download all probability snapshots and results', frame.to_csv(index=False),
                               'experimental_prop_history.csv', 'text/csv', key=key+'_download')
            ui.download_button('Download evaluation sample', selected.to_csv(index=False),
                               'experimental_prop_evaluation.csv', 'text/csv', key=key+'_evaluation')
