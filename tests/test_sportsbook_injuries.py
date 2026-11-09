from datetime import date, datetime, timezone
from unittest import TestCase
from unittest.mock import Mock, patch

import pandas as pd

from sportsbook_injuries import ET, HEADERS, latest_report, parse_report, slate_questionables


def word(text, x, top):
    return dict(text=text, x0=x, top=top, bottom=top + 4)


class InjuryReportTests(TestCase):
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
