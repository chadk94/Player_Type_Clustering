"""Personal prop shortlists and opt-in, comment-only partner reviews."""
from contextlib import closing
from base64 import b64encode
from datetime import date, datetime, timezone
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import time
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from sportsbook_lines import BOOKS

ROOT = Path(__file__).parent
STATS = {'PTS', 'REB', 'AST', '3PM', 'STL', 'BLK', 'PRA'}
WEEK = 7 * 24 * 3600
TOKEN = re.compile(r'[0-9a-f]{64}\Z')
MAX_PLAYS = 400


def prop_id(play):
    parts = [play[k] for k in ('day', 'event', 'book', 'player_id', 'stat', 'side')]
    parts.append(format(play['line'], '.12g'))
    return sha256(json.dumps(parts, separators=(',', ':')).encode()).hexdigest()


def edge_pick(row, event, book, day):
    gap = row['Signed edge %']
    if not pd.notna(gap) or gap == 0:
        return None
    side = 'Over' if gap > 0 else 'Under'
    odds = row[f'{side} odds']
    if pd.isna(odds) or (side == 'Under' and row.Stat in ('STL','BLK') and odds < -160):
        return None
    quote = row.get('Updated UTC')
    pick = dict(day=day.isoformat(), event=str(event['id']), game=f"{event['Away']} @ {event['Home']}",
                book=book, player_id=int(row.PLAYER_ID), player=row.Player, stat=row.Stat, side=side,
                line=float(row['Book line']), odds=int(odds), projected=float(row.Projected),
                edge=float(row['Edge %']), quoted=str(quote) if pd.notna(quote) else '', status='Review')
    return dict(pick, id=prop_id(pick))


def valid_day(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError('Invalid slate date.')
    return value


def clean_picks(day, picks, limit=MAX_PLAYS):
    if not isinstance(picks, list) or len(picks) > limit:
        raise ValueError(f'A slate can contain up to {limit:,} plays for this action.')
    result, seen = [], set()
    for raw in picks:
        try:
            fields = ('id','day','event','game','book','player_id','player','stat','side','line',
                      'odds','projected','edge','quoted','status')
            p = {k: raw[k] for k in fields}
            if (p['day'] != day or p['book'] not in BOOKS or p['stat'] not in STATS
                    or p['side'] not in ('Over','Under') or p['status'] not in ('Review','Confirmed')
                    or type(p['player_id']) is not int or type(p['odds']) is not int or abs(p['odds']) < 100
                    or any(not isinstance(p[k],str) or not 0 < len(p[k]) <= 120 for k in ('player','game','event'))
                    or not isinstance(p['quoted'],str) or len(p['quoted']) > 60
                    or any(type(p[k]) not in (float,int) or not math.isfinite(p[k]) or p[k] < 0
                           for k in ('line','projected','edge'))
                    or prop_id(p) != p['id'] or p['id'] in seen):
                raise ValueError
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError('A saved play is invalid. Remove it and select it again from Edges.') from exc
        seen.add(p['id'])
        result.append(p)
    return result


def plays_workbook(day, picks):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    day = valid_day(day)
    picks = clean_picks(day, picks, limit=10000)
    if not picks:
        raise ValueError('Save a play before downloading this slate.')
    wb = Workbook()
    sheet = wb.active
    sheet.title = "Today's Plays"
    sheet.merge_cells('A1:K1')
    sheet['A1'] = f"Today's Plays · {day}"
    sheet['A1'].font = Font(bold=True, size=16, color='17354A')
    sheet.row_dimensions[1].height = 30
    sheet.append(['Player', 'Game', 'Stat', 'Side', 'Line', 'Odds', 'Sportsbook',
                  'Projected', 'Edge', 'Status', 'Partner feedback'])
    for cell in sheet[2]:
        cell.font = Font(bold=True, color='FFFFFF')
        cell.fill = PatternFill('solid', fgColor='264A60')
    sheet.row_dimensions[2].height = 22
    for n, p in enumerate(sorted(picks, key=lambda p: p['status'] != 'Review'), start=3):
        values = [p['player'], p['game'], p['stat'], p['side'], p['line'], p['odds'], p['book'],
                  p['projected'], p['edge'] / 100, 'In review' if p['status'] == 'Review' else 'Confirmed']
        for column, value in enumerate(values, start=1):
            cell = sheet.cell(n, column, value)
            if isinstance(value, str):
                cell.data_type = 's'  # Treat exported labels as literal text, including a leading '='.
        sheet.cell(n, 6).number_format = '+0;-0;0'
        sheet.cell(n, 8).number_format = '0.00'
        sheet.cell(n, 9).number_format = '0.00%'
        sheet.cell(n, 11).alignment = Alignment(wrap_text=True, vertical='top')
    sheet.freeze_panes = 'B3'
    sheet.auto_filter.ref = f'A2:K{sheet.max_row}'
    for i, width in enumerate([25, 16, 9, 10, 10, 10, 18, 14, 12, 15, 44], start=1):
        sheet.column_dimensions[get_column_letter(i)].width = width
    output = BytesIO()
    wb.save(output)
    return output.getvalue()


def public_token(owner_key, day):
    if not isinstance(owner_key, str) or not TOKEN.fullmatch(owner_key):
        raise ValueError('A valid browser sharing key is required.')
    return sha256(f'plays-v1/{valid_day(day)}/{owner_key}'.encode()).hexdigest()


def db():
    path = os.environ.get('TODAYS_PLAYS_DB_PATH') or str(Path(tempfile.gettempdir())/'nba-props-shared-plays-v1.sqlite3')
    conn = sqlite3.connect(path, timeout=8)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS boards (
            token TEXT PRIMARY KEY, day TEXT NOT NULL, picks TEXT NOT NULL, expires INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS feedback (
            token TEXT NOT NULL REFERENCES boards(token) ON DELETE CASCADE, nonce TEXT NOT NULL,
            play TEXT NOT NULL, author TEXT NOT NULL, note TEXT NOT NULL, posted INTEGER NOT NULL,
            PRIMARY KEY(token, nonce));
    """)
    return conn


def publish(day, owner_key, picks):
    token = public_token(owner_key, day)
    payload = json.dumps(clean_picks(day, picks), allow_nan=False)
    now = int(time.time())
    with closing(db()) as conn, conn:
        conn.execute('DELETE FROM boards WHERE expires <= ?', (now,))
        conn.execute('''INSERT INTO boards VALUES (?,?,?,?) ON CONFLICT(token) DO UPDATE SET
                     picks=excluded.picks, expires=excluded.expires''', (token,day,payload,now+WEEK))
    return token


def read_board(token):
    if not isinstance(token, str) or not TOKEN.fullmatch(token):
        return None
    with closing(db()) as conn, conn:
        conn.execute('BEGIN')
        board = conn.execute('SELECT day,picks FROM boards WHERE token=? AND expires>?',
                             (token,int(time.time()))).fetchone()
        if board is None:
            return None
        notes = conn.execute('SELECT nonce AS id,play,author,note,posted FROM feedback WHERE token=? ORDER BY posted,rowid',
                             (token,)).fetchall()
    return dict(day=board['day'], picks=json.loads(board['picks']), notes=[dict(n) for n in notes])


def comment(token, play, author, note, nonce):
    author, note = author.strip(), note.strip()
    if not (0 < len(author) <= 60 and 0 < len(note) <= 1000):
        raise ValueError('Enter your name and a comment of up to 1,000 characters.')
    with closing(db()) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute('SELECT picks FROM boards WHERE token=? AND expires>?', (token,int(time.time()))).fetchone()
        if not row or play not in {p['id'] for p in json.loads(row[0])}:
            raise ValueError('This play is no longer shared. Refresh the list.')
        if conn.execute('SELECT COUNT(*) FROM feedback WHERE token=?',(token,)).fetchone()[0] >= 2000:
            raise ValueError('This review has reached its comment limit.')
        conn.execute('INSERT OR IGNORE INTO feedback VALUES (?,?,?,?,?,?)',
                     (token,nonce,play,author,note,int(time.time())))


def public_url(token):
    base = os.environ.get('NBA_PROPS_PUBLIC_URL','https://cktest2.streamlit.app/').rstrip('/')
    return f'{base}/?plays={token}'


def label(p):
    return f"{p['player']} · {p['side']} {p['line']:g} {p['stat']} ({p['odds']:+d}) · {p['book']}"


@lru_cache(maxsize=1)
def shortlist_component(runtime, js, css):
    # Long-running sessions need a fresh browser component when its assets change.
    version = sha256((js + '\0' + css).encode()).hexdigest()[:12]
    return st.components.v2.component(f'todays_plays_{version}', html='<section></section>',
                                      js=js, css=css)


@st.fragment
def render_shortlist(ui):
    ui.subheader("Today's Plays")
    key = 'todays_plays_list'
    def on_excel():
        request = st.session_state.get(key, {}).get('excel') or {}
        if not isinstance(request, dict):
            return
        day = request.get('day')
        reply = dict(day=day, nonce=request.get('nonce'))
        try:
            reply['base64'] = b64encode(plays_workbook(day, request.get('picks'))).decode('ascii')
        except (ValueError, TypeError):
            reply['error'] = 'Could not export this slate. Check the saved picks and try again.'
        st.session_state[key+'_workbook'] = reply

    def on_action():
        action = st.session_state.get(key, {}).get('action') or {}
        if not isinstance(action, dict):
            return
        day = action.get('day')
        result = {'nonce':action.get('nonce'), 'day':day}
        try:
            token = public_token(action.get('owner'), day)
            if action.get('type') == 'publish':
                publish(day, action['owner'], action.get('picks'))
            elif action.get('type') != 'feedback':
                raise ValueError('Unknown review action.')
            board = read_board(token)
            if not board:
                raise ValueError('This shared list is unavailable. Share it again to restore the link.')
            result.update(url=public_url(token), notes=board['notes'])
        except (TypeError, ValueError, sqlite3.Error, OSError) as exc:
            result['error'] = str(exc) if isinstance(exc, ValueError) else 'Sharing is unavailable. Your browser list is still saved.'
        st.session_state[key+'_result'] = result
    js = (ROOT/'play_storage.mjs').read_text() + '\n' + (ROOT/'todays_plays.mjs').read_text()
    component = shortlist_component(st.runtime.get_instance(), js, (ROOT/'todays_plays.css').read_text())
    component(key=key, data=dict(
        today=datetime.now(ZoneInfo('America/Denver')).date().isoformat(),
        result=st.session_state.get(key+'_result'), workbook=st.session_state.pop(key+'_workbook', None)),
        on_action_change=on_action, on_excel_change=on_excel)


def render_partner_review(ui, token):
    ui.title("Today's Plays")
    try:
        board = read_board(token)
    except (sqlite3.Error, OSError):
        ui.error('The shared list is temporarily unavailable. Try again.')
        return
    if not board:
        ui.info('This review link is unavailable or expired. Ask the list owner to share it again.')
        return
    picks = {p['id']: p for p in board['picks']}
    ui.caption(f"Partner review · {date.fromisoformat(board['day']):%B %d, %Y} · Lines and prices are saved at selection.")
    ui.button('Refresh shared list')
    if not picks:
        ui.info('There are no plays on this shared list.')
        return
    ui.dataframe(pd.DataFrame([{'Player':p['player'],'Game':p['game'],'Prop':f"{p['side']} {p['line']:g} {p['stat']}",
        'Price':f"{p['odds']:+d}",'Book':p['book'],'Projected':round(p['projected'],2),
        'Edge':f"{p['edge']:.2f}%",'Status':p['status']} for p in picks.values()]), hide_index=True,width='stretch')
    ui.subheader('Partner feedback')
    if ui.session_state.pop('plays_comment_sent', False):
        ui.success('Feedback posted.')
    visible_notes = [n for n in board['notes'] if n['play'] in picks]
    if not visible_notes:
        ui.caption('No feedback yet.')
    for item in visible_notes:
        ui.text(f"{item['author']} · {label(picks[item['play']])}\n{item['note']}")
    if ui.session_state.pop('plays_clear_note', False):
        ui.session_state['partner_play_note'] = ''
    with ui.form('partner_play_feedback'):
        play = ui.selectbox('Play to review',list(picks),format_func=lambda i:label(picks[i]))
        author = ui.text_input('Your name',max_chars=60)
        note = ui.text_area('Your feedback',max_chars=1000, key='partner_play_note')
        submit = ui.form_submit_button('Post feedback')
    if submit:
        nonce = ui.session_state.setdefault('plays_feedback_nonce',uuid4().hex)
        try:
            comment(token,play,author,note,nonce)
        except ValueError as exc:
            ui.warning(str(exc))
        except (sqlite3.Error,OSError):
            ui.error('Feedback could not be saved. Try posting again.')
        else:
            ui.session_state.pop('plays_feedback_nonce')
            ui.session_state['plays_comment_sent'] = True
            ui.session_state['plays_clear_note'] = True
            ui.rerun()
