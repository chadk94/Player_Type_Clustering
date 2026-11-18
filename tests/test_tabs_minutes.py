import copy
import unittest

import pandas as pd

from tabs_minutes import parse_tabs_minutes, parse_tabs_browser_snapshot
from edge_review import minute_reference

NOW=pd.Timestamp('2026-11-18T00:00:00Z')
GAME=dict(Away='POR',Home='MIN',GameTimeUTC=pd.Timestamp('2026-11-18T01:00:00Z'))
ROSTER=pd.DataFrame([
    dict(PLAYER_ID=1,PLAYER_NAME='Robert Williams III',TEAM_ABBREVIATION='POR'),
    dict(PLAYER_ID=2,PLAYER_NAME='Deni Avdija',TEAM_ABBREVIATION='POR'),
    dict(PLAYER_ID=3,PLAYER_NAME='Wrong Team',TEAM_ABBREVIATION='LAL')])
CSV=b'Player,ID,Team,Opp,Tabs Min,My Min,Boost %\nRobert Williams,44956202,POR,MIN,19,33,20\n'

def browser():
    return dict(schema=1,source='TabsNBA',source_url='https://app.tabsnba.com/projections',
        slate_date='2026-11-17',scenario='POR: Avdija OUT (Q)',source_updated_at=(NOW-pd.Timedelta(minutes=5)).isoformat(),
        observed_at=NOW.isoformat(),manual_edits=False,scope='POR @ MIN',
        rows=[dict(Player='Robert Williams III',Team='POR',Opp='MIN',minutes=19,status='Projected'),
              dict(Player='Deni Avdija',Team='POR',Opp='MIN',minutes=0,status='Out')])


class TabsTests(unittest.TestCase):
    def parse(self,content=CSV,**kwargs):
        args=dict(filename='Tabs_DK_2026-11-17.csv',roster=ROSTER,games=[GAME],captured_at=NOW,
                  scenario='POR: Avdija OUT (Q)',source_updated_at=NOW-pd.Timedelta(minutes=5))
        return parse_tabs_minutes(content,**(args|kwargs))

    def test_tabs_column_not_edited_minutes_or_fantasy_id_and_no_missing_zero(self):
        saved=ROSTER.copy(deep=True)
        refs,details=self.parse()
        self.assertEqual(refs.PLAYER_ID.tolist(),[1]);self.assertEqual(refs.minutes.tolist(),[19])
        self.assertNotIn(2,refs.PLAYER_ID.tolist())
        self.assertEqual(details['scenario'],'POR: Avdija OUT (Q)')
        pd.testing.assert_frame_equal(ROSTER,saved)

    def test_same_game_date_team_identity_required(self):
        refs,_=self.parse(filename='Tabs_DK_2026-11-16.csv');self.assertTrue(refs.empty)
        refs,_=self.parse(CSV.replace(b',POR,MIN,',b',POR,LAL,'));self.assertTrue(refs.empty)
        refs,details=self.parse(CSV.replace(b'Robert Williams',b'Wrong Team'))
        self.assertTrue(refs.empty);self.assertEqual(len(details['unmatched']),1)

    def test_csv_invalid_and_duplicate_rows_fail_closed(self):
        for content in [CSV.replace(b',19,',b',nan,'),CSV.replace(b',19,',b',49,'),CSV.replace(b',19,',b',-1,'),
                        CSV+CSV.splitlines()[-1]+b'\n']:
            with self.assertRaises(ValueError):self.parse(content)
        for changes in [dict(scenario=''),dict(filename='unknown.csv'),dict(source_updated_at=NOW+pd.Timedelta(minutes=1))]:
            with self.assertRaises(ValueError):self.parse(**changes)

    def test_freshness_uses_publication_not_download_time(self):
        row=pd.Series(dict(PLAYER_ID=1,Team='POR',Opponent='MIN'))
        for publication in [None,NOW-pd.Timedelta(hours=3)]:
            refs,_=self.parse(source_updated_at=publication)
            self.assertIsNone(minute_reference(refs,row,GAME,NOW))
        refs,_=self.parse()
        self.assertEqual(minute_reference(refs,row,GAME,NOW).minutes,19)

    def test_browser_preserves_explicit_out_and_conditional_scenario(self):
        snapshot=browser();saved=copy.deepcopy(snapshot)
        refs,details=parse_tabs_browser_snapshot(snapshot,ROSTER,[GAME],NOW)
        self.assertEqual(refs.minutes.tolist(),[19,0]);self.assertEqual(snapshot,saved)
        self.assertEqual(details['scope'],'POR @ MIN')
        self.assertTrue(refs.independent.all());self.assertEqual(refs.source_mode.unique().tolist(),['Browser comparison'])

    def test_browser_stale_edited_missing_bad_origin_or_status_fails_closed(self):
        for changes in [dict(manual_edits=True),dict(source_url='https://other.example'),dict(rows=[]),
                        dict(source_updated_at=None),dict(observed_at=(NOW-pd.Timedelta(hours=3)).isoformat()),
                        dict(source_updated_at=(NOW+pd.Timedelta(minutes=1)).isoformat())]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                parse_tabs_browser_snapshot(browser()|changes,ROSTER,[GAME],NOW)
        b=browser();b['rows'][1]['minutes']=18
        with self.assertRaises(ValueError):parse_tabs_browser_snapshot(b,ROSTER,[GAME],NOW)

if __name__=='__main__':unittest.main()
