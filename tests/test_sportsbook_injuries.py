from datetime import date, datetime, timedelta, timezone
from unittest import TestCase
from unittest.mock import Mock, patch

import pandas as pd

from sportsbook_injuries import (ET, HEADERS, TeamInjuries, injury_alert_changes,
                                 latest_report, parse_report, slate_questionables)


def word(text, x, top):
    return dict(text=text, x0=x, top=top, bottom=top + 4)


class InjuryReportTests(TestCase):
    def test_key_alerts_follow_verified_editions_and_preserve_history_through_gaps(self):
        now = datetime(2026, 11, 9, 13, tzinfo=ET)
        key = {'Starter': (0, 30), 'New late scratch': (25, 25), 'Unknown minutes': None}

        def report(previous, minutes, statuses, submitted=True, complete=True):
            team = TeamInjuries(submitted, complete, statuses, key)
            return injury_alert_changes(previous, now + timedelta(minutes=minutes), {'CLE': team})

        history, changes = report(None, 0, {'Starter': 'Questionable'})
        self.assertFalse(changes)
        history, changes = report(history, 15, {}, submitted=False, complete=False)
        self.assertFalse(changes)
        history, changes = report(history, 30, {'Bench': 'Out', 'Unknown minutes': 'Out'}, complete=False)
        self.assertFalse(changes)
        current = {'Starter': 'Out', 'New late scratch': 'Doubtful', 'Bench': 'Doubtful'}
        history, changes = report(history, 45, current)
        self.assertCountEqual(changes, [('CLE', 'Starter', 'Questionable', 'Out'),
                                        ('CLE', 'New late scratch', None, 'Doubtful')])
        for minute in (45, 30):
            self.assertEqual(report(history, minute, {}), (history, []))
        history, changes = report(history, 60, {'Starter': 'Available'})
        self.assertCountEqual(changes, [('CLE', 'Starter', 'Out', 'Available'),
                                        ('CLE', 'New late scratch', 'Doubtful', 'Off report')])
        _, changes = report(history, 75, {})
        self.assertFalse(changes)

    def test_pdf_continuation_and_pending_team(self):
        stamp = datetime(2026, 11, 9, 16, 30, tzinfo=ET)
        columns = [20, 120, 200, 260, 420, 580, 660]
        headings = [word(label, x, 105) for label, x in zip(HEADERS, columns)]
        first = Mock(height=600, lines=[])
        first.extract_text.return_value = 'Injury Report: 11/09/26 04:30 PM'
        first.extract_words.return_value = headings + [
            word(value, x, 124) for x, value in zip(columns, [
                '11/09/2026','07:00(ET)','CLE@CHA','ClevelandCavaliers','Allen,Jarrett','Questionable','Illness'])]
        second = Mock(height=600, lines=[dict(top=110, bottom=110, x0=1, x1=800)])
        second.extract_words.return_value = [word('Report:', 350, 46),
            word('Mobley,Evan', 422, 80), word('Available', 581, 80),
            word('CharlotteHornets', 262, 125), word('NOT YET SUBMITTED', 421, 125)]
        with patch('sportsbook_injuries.pdfplumber.open') as pdf:
            pdf.return_value.__enter__.return_value.pages = [first, second]
            reports = parse_report(b'example', stamp)
            with self.assertRaises(ValueError):
                parse_report(b'example', stamp.replace(hour=15))
        self.assertEqual(reports[stamp.date(),'CLE@CHA','CLE']['players'],
                         {'Jarrett Allen':'Questionable','Evan Mobley':'Available'})
        self.assertFalse(reports[stamp.date(),'CLE@CHA','CHA']['submitted'])

    def test_dashboard_popups_follow_scope_and_start_silently(self):
        from streamlit.testing.v1 import AppTest
        script = """
import pandas as pd
import streamlit as st
from projection_dashboard import render_edge_injuries
roster = pd.DataFrame(dict(TEAM_ABBREVIATION=['CLE','NYK','NYK'], PLAYER_ID=[1,2,3],
    PLAYER_NAME=['First','Second','Reserve'], PROJECTED_MIN=[30.,31.,20.]))
projections = pd.DataFrame({'PLAYER_ID':[1,2,3], 'Projected minutes':[0.,31.,20.]})
games = st.session_state['fixtures']
render_edge_injuries(st, games[st.session_state['selection']], games, roster, projections, None)
"""
        now = datetime.now(timezone.utc)
        tip = now + timedelta(hours=3)
        day = tip.astimezone(ET).date()
        fixtures = [dict(id='a', Away='CLE', Home='CHA', GameTimeUTC=tip),
                    dict(id='b', Away='NYK', Home='BOS', GameTimeUTC=tip)]
        games = [(day, 'CLE@CHA', 'CLE'), (day, 'NYK@BOS', 'NYK')]
        def edition(minutes, first, second):
            reports = {k: dict(submitted=True, partial=False, players=players) for k, players in
                       zip(games, [{'First': first}, {'Second': second, 'Reserve': second}])}
            return (now + timedelta(minutes=minutes), 'https://example.com/report.pdf', reports)
        app = AppTest.from_string(script, default_timeout=30)
        app.session_state['fixtures'] = fixtures
        app.session_state['selection'] = 0
        with (patch('projection_dashboard.edge_injury_report', return_value=edition(0,'Questionable','Questionable')) as feed,
              patch('projection_dashboard.render_injury_voice') as voice):
            app.run()
            self.assertFalse(app.exception)
            self.assertFalse(app.toast)
            self.assertEqual(voice.call_args.args[-1], [])
            self.assertEqual(app.radio(key='injury_alert_scope').value, 'All games on this slate')
            app.radio(key='injury_alert_scope').set_value('Selected game').run()
            feed.return_value = edition(15, 'Out', 'Available')
            app.run()
            self.assertEqual(len(app.toast), 1)
            self.assertEqual(voice.call_args.args[-1], ['Cleveland Cavaliers. First is now out.'])
            self.assertIn('First (CLE)', app.toast[0].value)
            self.assertIn('Questionable → Out', app.toast[0].value)
            self.assertNotIn('Second', app.toast[0].value)
            app.run()
            self.assertFalse(app.toast)
            self.assertEqual(voice.call_args.args[-1], [])
            app.session_state['selection'] = 1
            app.run()
            self.assertFalse(app.toast)
            app.radio(key='injury_alert_scope').set_value('All games on this slate').run()
            self.assertFalse(app.toast)
            feed.return_value = edition(30, 'Doubtful', 'Out')
            app.run()
            self.assertEqual(len(app.toast), 2)
            self.assertEqual(voice.call_args.args[-1], ['Cleveland Cavaliers. First is now doubtful.',
                                                      'New York Knicks. Second is now out.'])
            self.assertFalse(any('Reserve' in t.value for t in app.toast))
            self.assertFalse(any('CLE @ CHA' in m.value for m in app.markdown))
            app.radio(key='injury_alert_scope').set_value('Off').run()
            feed.return_value = edition(45, 'Available', 'Available')
            app.run()
            self.assertFalse(app.toast)
            self.assertFalse(voice.call_args.args[0])
            app.radio(key='injury_alert_scope').set_value('Selected game').run()
            self.assertFalse(app.toast)

    def test_only_exact_game_unresolved_and_key_in_either_rotation(self):
        roster = pd.DataFrame({'TEAM_ABBREVIATION':'CLE', 'PLAYER_ID':[1,2,3,4,5],
            'PLAYER_NAME':['A One','B Two','C Three','D Four','E Five'],
            'WorkbookPlayerName':['A One','B Two Jr.','C Three','D Four','E Five'],
            'PROJECTED_MIN':[30.,0.,20.,40.,10.]})
        projected = pd.DataFrame({'PLAYER_ID':[1,2,3,4,5], 'Projected minutes':[0.,31.,20.,40.,1.]})
        tip = pd.Timestamp('2026-11-10T00:00Z')
        games = [dict(Away='CLE',Home='CHA',GameTimeUTC=tip)]
        key = date(2026,11,9), 'CLE@CHA', 'CLE'
        entry = {'submitted':True,'partial':False,'players':{'A One':'Questionable',
            'B TwoJr.':'Questionable','C Three':'Questionable','D Four':'Out','E Five':'Questionable',
            'Unknown Player':'Questionable'}}
        reports = {key:entry, (date(2026,11,10),'CHA@CLE','CLE'):dict(entry, players={'D Four':'Questionable'})}
        rows, pending, unverified = slate_questionables(games,reports,roster,projected,None)
        self.assertEqual([(r[2],r[3]) for r in rows],[('A One',0.),('B Two',31.)])
        self.assertEqual(pending,['CHA'])
        self.assertEqual(unverified,['Unknown Player (CLE)'])
        entry['players']['A One'] = 'Available'
        rows, _, _ = slate_questionables(games,reports,roster,projected,None)
        self.assertEqual([r[2] for r in rows],['B Two'])

    def test_latest_available_quarter_hour_and_bounded_outage(self):
        now = datetime(2026,11,9,21,2,tzinfo=timezone.utc)
        found = Mock(status_code=200, content=b'%PDF-example')
        missing = Mock(status_code=404)
        with patch('sportsbook_injuries.requests.get',side_effect=[missing,found]) as get, \
             patch('sportsbook_injuries.parse_report',return_value={'ok':True}):
            stamp, url, _ = latest_report(now)
            self.assertTrue(url.endswith('2026-11-09_03_45PM.pdf'))
            self.assertEqual(stamp.minute,45)
            self.assertEqual(get.call_count,2)
        with patch('sportsbook_injuries.requests.get', return_value=missing) as get:
            with self.assertRaises(ValueError):
                latest_report(now)
            self.assertEqual(get.call_count,4)
