"""Game-scoped questionable players from the NBA's quarter-hour injury reports."""
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from io import BytesIO
import re
from zoneinfo import ZoneInfo

import pandas as pd
import pdfplumber
import requests

from approved_minutes import for_matchup
from live_projections import name_key
from sportsbook_lines import TEAM_NAMES

ET = ZoneInfo('America/New_York')
REPORT_BASE = 'https://ak-static.cms.nba.com/referee/injury/Injury-Report_'
KEY_MINUTES = 20
HEADERS = ('GameDate', 'GameTime', 'Matchup', 'Team', 'PlayerName', 'CurrentStatus', 'Reason')
STATUSES = {'Available', 'Probable', 'Questionable', 'Doubtful', 'Out'}


def compact(name):
    return name_key(name).replace(' ', '')


TEAM_CODES = {compact(name): code for name, code in TEAM_NAMES.items()}


def report_cells(page, words, columns, body_top):
    """Use the published row dividers for wrapped names and multipage reports."""
    boundaries = sorted({e['top'] for e in page.lines if abs(e['top'] - e['bottom']) < 1
                         and e['x0'] <= columns[-1] < e['x1']})
    groups = {}
    last_top = None
    for word in sorted(words, key=lambda w: (w['top'], w['x0'])):
        top = word['top']
        if top <= body_top or top > page.height - 40:
            continue
        if boundaries:
            group = bisect_right(boundaries, top)
        else:
            if last_top is None or top - last_top > 3:
                last_top = top
            group = last_top
        cells = groups.setdefault(group, [''] * len(columns))
        col = max(0, bisect_right(columns, word['x0']) - 1)
        cells[col] = (cells[col] + ' ' + word['text']).strip()
    for cells in groups.values():
        if not re.fullmatch(r'page\d+of\d+', compact(''.join(cells))):
            yield cells


def parse_report(content, stamp):
    reports = {}
    columns = None
    day = matchup = team = None
    with pdfplumber.open(BytesIO(content)) as pdf:
        if not pdf.pages or stamp.strftime('%m/%d/%y %I:%M %p') not in pdf.pages[0].extract_text():
            raise ValueError('Unexpected NBA report timestamp')
        for page in pdf.pages:
            words = page.extract_words()
            heads = {w['text']: w for w in words if w['text'] in HEADERS}
            if len(heads) == len(HEADERS):
                columns = [heads[k]['x0'] - 2 for k in HEADERS]
                top = heads['PlayerName']['bottom']
            elif columns:
                top = next((w['bottom'] for w in words if w['text'] == 'Report:'), 35)
            else:
                raise ValueError('Missing injury report columns')
            for date, _, pair, club, name, status, reason in report_cells(page, words, columns, top):
                if date:
                    day = datetime.strptime(date, '%m/%d/%Y').date()
                if pair:
                    matchup = pair.replace(' ', '')
                if club:
                    team = TEAM_CODES.get(compact(club))
                if not day or not matchup or team not in matchup.split('@'):
                    continue
                entry = reports.setdefault((day, matchup, team), {'submitted': False, 'players': {}, 'partial': False})
                if 'notyetsubmitted' in compact(name + status + reason):
                    entry['submitted'] = False
                elif name and status in STATUSES:
                    last, comma, first = name.partition(',')
                    display = (first.strip() + ' ' + last.strip()) if comma else name
                    entry['players'][display] = status
                    entry['submitted'] = True
                elif compact(name + status + reason) in ('none', 'noinjuries', 'na'):
                    entry['submitted'] = True
                elif name or status:
                    entry['partial'] = True
    if not reports:
        raise ValueError('No team reports found')
    return reports


def latest_report(now=None):
    now = now or datetime.now(timezone.utc)
    slot = datetime.fromtimestamp(int(now.timestamp()) // 900 * 900, timezone.utc)
    for offset in range(4):
        stamp = (slot - timedelta(minutes=15 * offset)).astimezone(ET)
        url = REPORT_BASE + stamp.strftime('%Y-%m-%d_%I_%M%p') + '.pdf'
        response = requests.get(url, timeout=(2, 4))
        if response.status_code in (403, 404):
            continue
        response.raise_for_status()
        if not response.content.startswith(b'%PDF-'):
            raise ValueError('Unexpected NBA report format')
        try:
            return stamp, url, parse_report(response.content, stamp)
        except pdfplumber.utils.exceptions.PdfminerException as exc:
            raise ValueError('Unreadable NBA injury report') from exc
    raise ValueError('No recent official report found')


@dataclass
class TeamInjuries:
    submitted: bool
    complete: bool
    statuses: dict = field(default_factory=dict)
    key_minutes: dict = field(default_factory=dict)
    questionables: list = field(default_factory=list)
    unverified: list = field(default_factory=list)


def game_injuries(game, reports, roster, automatic, bundle):
    """Match each official status to this game's roster, approvals and minutes."""
    away, home, tip = game['Away'], game['Home'], game['GameTimeUTC']
    day = pd.Timestamp(tip).tz_convert(ET).date()
    pair = f'{away}@{home}'
    approved = for_matchup(bundle, roster, away, home, tip)
    observations = {}
    for team in (away, home):
        entry = reports.get((day, pair, team), {})
        submitted = entry.get('submitted', False)
        info = TeamInjuries(submitted, submitted and not entry.get('partial', False))
        observations[team] = info
        if not submitted:
            continue
        if entry['partial']:
            info.unverified.append(f'{team} report incomplete')
        names = {}
        for player in roster[roster.TEAM_ABBREVIATION.eq(team)].itertuples():
            for alias in (player.PLAYER_NAME, getattr(player, 'WorkbookPlayerName', None)):
                if pd.notna(alias):
                    names.setdefault(compact(alias), set()).add(player.PLAYER_NAME)
            minutes = approved.get(player.PLAYER_ID, automatic.get(player.PLAYER_ID))
            known = [m for m in (minutes, player.PROJECTED_MIN) if pd.notna(m)]
            if known and max(known) > KEY_MINUTES:
                info.key_minutes[player.PLAYER_NAME] = (minutes, max(known))
            elif len(known) < 2:
                info.key_minutes[player.PLAYER_NAME] = None
        for reported, status in entry['players'].items():
            matches = names.get(compact(reported), set())
            if len(matches) != 1:
                info.complete = False  # An unmatched name could be a previously listed player.
                if status == 'Questionable':
                    info.unverified.append(f'{reported} ({team})')
                continue
            name = next(iter(matches))
            info.statuses[name] = status
            if status == 'Questionable' and name in info.key_minutes:
                if info.key_minutes[name] is None:
                    info.unverified.append(f'{name} ({team})')
                else:
                    info.questionables.append((pair, team, name, *info.key_minutes[name]))
    return observations


def slate_questionables(games, reports, roster, projections, bundle):
    rows, pending, unverified = [], [], []
    automatic = projections.set_index('PLAYER_ID')['Projected minutes'].to_dict()
    for game in games:
        for team, info in game_injuries(game, reports, roster, automatic, bundle).items():
            rows.extend(info.questionables)
            unverified.extend(info.unverified)
            if not info.submitted:
                pending.append(team)
    return rows, pending, unverified


def injury_alert_changes(previous, stamp, observations):
    """Compare verified editions for one game. First sighting establishes a quiet baseline."""
    if previous is not None and stamp <= previous[0]:
        return previous, []
    old_teams = previous[1] if previous else {}
    updated, changes = {}, []
    for team, info in observations.items():
        old = old_teams.get(team, {})
        if not info.submitted:
            updated[team] = old
            continue
        current = info.statuses
        if previous is not None:
            for name, status in current.items():
                before = old.get(name)
                if (info.key_minutes.get(name) is not None and before != status
                        and (before is not None or status != 'Available')):
                    changes.append((team, name, before, status))
            if info.complete:
                for name in old.keys() - current.keys():
                    if info.key_minutes.get(name) is not None and old[name] != 'Available':
                        changes.append((team, name, old[name], 'Off report'))
        updated[team] = current.copy() if info.complete else {**old, **current}
    return (stamp, updated), sorted(changes)
