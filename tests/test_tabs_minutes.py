import copy
import unittest
from unittest.mock import patch

import pandas as pd

from tabs_minutes import parse_tabs_minutes, parse_tabs_browser_snapshot
import tabs_minutes
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

    def test_unmatched_zero_reserve_does_not_block_complete_rotation(self):
        snapshot=browser()
        snapshot['rows'].append(dict(Player='Unknown Reserve',Team='POR',Opp='MIN',minutes=0,status='Out'))
        refs,details=parse_tabs_browser_snapshot(snapshot,ROSTER,[GAME],NOW)
        self.assertEqual(refs.attrs['incomplete_teams'],[])
        self.assertEqual(details['unmatched'][0]['Minutes'],0)
        self.assertEqual(refs.PLAYER_ID.tolist(),[1,2])
        snapshot['rows'][-1].update(minutes=5,status='Projected')
        refs,_=parse_tabs_browser_snapshot(snapshot,ROSTER,[GAME],NOW)
        self.assertEqual(refs.attrs['incomplete_teams'],['POR'])

    def test_philon_suffix_alias_is_scoped_to_his_team(self):
        roster=pd.DataFrame([dict(PLAYER_ID=1642889,PLAYER_NAME='Labaron Philon',TEAM_ABBREVIATION='PHI')])
        game=dict(Away='PHI',Home='ORL',GameTimeUTC=GAME['GameTimeUTC'])
        data=b'Player,Team,Opp,Tabs Min\nLabaron Philon Jr.,PHI,ORL,3.5\n'
        refs,_=self.parse(data,roster=roster,games=[game])
        self.assertEqual(refs.PLAYER_ID.tolist(),[1642889])

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
        self.assertEqual(refs.source_status.tolist(),['Projected','Out'])
        self.assertEqual(details['scope'],'POR @ MIN')
        self.assertTrue(refs.independent.all());self.assertEqual(refs.source_mode.unique().tolist(),['Browser comparison'])

    def test_healthy_zero_is_not_an_out_designation(self):
        b=browser();b['rows'][1]['status']='Projected'
        refs,_=parse_tabs_browser_snapshot(b,ROSTER,[GAME],NOW)
        self.assertEqual(refs.loc[refs.PLAYER_ID.eq(2),'source_status'].iloc[0],'Projected')
        refs,_=self.parse()
        self.assertEqual(refs.source_status.tolist(),['Unknown'])

    def test_browser_stale_edited_missing_bad_origin_or_status_fails_closed(self):
        for changes in [dict(manual_edits=True),dict(source_url='https://other.example'),dict(rows=[]),
                        dict(source_updated_at=None),dict(observed_at=(NOW-pd.Timedelta(hours=3)).isoformat()),
                        dict(source_updated_at=(NOW+pd.Timedelta(minutes=1)).isoformat())]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                parse_tabs_browser_snapshot(browser()|changes,ROSTER,[GAME],NOW)
        b=browser();b['rows'][1]['minutes']=18
        with self.assertRaises(ValueError):parse_tabs_browser_snapshot(b,ROSTER,[GAME],NOW)

    def test_timestamp_and_row_order_updates_refresh_source_without_rebuilding_dashboard(self):
        original=browser()
        refs,_=parse_tabs_browser_snapshot(original,ROSTER,[GAME],NOW)
        fresh=original|dict(observed_at=(NOW+pd.Timedelta(seconds=30)).isoformat(),
                            source_updated_at=NOW.isoformat(),rows=list(reversed(original['rows'])))
        state={'tabs_minutes_reference':refs,'tabs_minutes_browser':{'result':{'ok':True,'snapshot':fresh}}}
        real_parse=parse_tabs_browser_snapshot
        with patch.object(tabs_minutes.st,'session_state',state), \
             patch.object(tabs_minutes,'parse_tabs_browser_snapshot',side_effect=lambda s,r,g,a=None,**kw:real_parse(s,r,g,NOW+pd.Timedelta(minutes=1),**kw)), \
             patch.object(tabs_minutes.st,'rerun') as rerun:
            tabs_minutes._remember_browser_result(ROSTER,[GAME])
            rerun.assert_not_called()
        self.assertEqual(state['tabs_minutes_reference'].source_updated_at.unique().tolist(),[NOW.isoformat()])
        self.assertEqual(state['tabs_browser_last_result']['snapshot'],fresh)

    def test_freshness_recovery_changes_eligibility_even_with_identical_minutes(self):
        refs,_=self.parse()
        stale=refs.assign(source_updated_at=(NOW-pd.Timedelta(hours=3)).isoformat())
        self.assertNotEqual(tabs_minutes._reference_signature(refs,NOW),
                            tabs_minutes._reference_signature(stale,NOW))

    def test_disconnect_and_page_restore_keep_original_minutes_and_timestamps(self):
        refs,details,saved,error=tabs_minutes.resolve_browser_result({'ok':True,'snapshot':browser()},None,ROSTER,[GAME],NOW)
        self.assertFalse(error);self.assertFalse(refs.source_saved.any())
        for state,last in [({'ok':False,'error':'Disconnected'},saved),
                           ({'ok':False,'saved_snapshot':saved},None), (None,saved)]:
            retained,_,stored,error=tabs_minutes.resolve_browser_result(state,last,ROSTER,[GAME],NOW+pd.Timedelta(hours=5))
            self.assertEqual(retained.minutes.tolist(),refs.minutes.tolist())
            self.assertEqual(retained.source_updated_at.tolist(),refs.source_updated_at.tolist())
            self.assertEqual(retained.updated_at.tolist(),refs.updated_at.tolist())
            self.assertTrue(retained.source_saved.all());self.assertEqual(stored,saved)
            row=pd.Series(dict(PLAYER_ID=1,Team='POR',Opponent='MIN'))
            self.assertIsNone(minute_reference(retained,row,GAME,NOW+pd.Timedelta(hours=5)))
            self.assertEqual(minute_reference(retained,row,GAME,NOW+pd.Timedelta(hours=5),allow_saved=True).minutes,19)

    def test_invalid_update_does_not_replace_saved_data_and_reconnect_replaces_it(self):
        saved=browser()
        bad=saved|dict(manual_edits=True)
        refs,_,stored,error=tabs_minutes.resolve_browser_result({'ok':True,'snapshot':bad},saved,ROSTER,[GAME],NOW)
        self.assertTrue(error);self.assertTrue(refs.source_saved.all());self.assertEqual(stored['snapshots'][0]['rows'],saved['rows'])
        fresh=copy.deepcopy(saved);fresh['rows'][0]['minutes']=24
        refs,_,stored,error=tabs_minutes.resolve_browser_result({'ok':True,'snapshot':fresh},saved,ROSTER,[GAME],NOW)
        self.assertFalse(error);self.assertFalse(refs.source_saved.any());self.assertEqual(refs.minutes.tolist(),[24,0])
        self.assertEqual(stored['snapshots'][0]['rows'],fresh['rows'])

    def test_all_30_teams_survive_filtered_views_partial_updates_reload_and_disconnect(self):
        from minutes_blend import minutes_plan
        teams='ATL BOS BKN CHA CHI CLE DAL DEN DET GSW HOU IND LAC LAL MEM MIA MIL MIN NOP NYK OKC ORL PHI PHX POR SAC SAS TOR UTA WAS'.split()
        roster=[];rows=[];base=[];games=[]
        for n,team in enumerate(teams):
            opponent=teams[n^1]
            if n%2==0:games.append(dict(Away=team,Home=opponent,GameTimeUTC=GAME['GameTimeUTC']))
            for i in range(7):
                pid=n*10+i+1;name=f'{team} Player {i}';minutes=40 if i<6 else 0
                roster.append(dict(PLAYER_ID=pid,PLAYER_NAME=name,TEAM_ABBREVIATION=team))
                rows.append(dict(Player=name,Team=team,Opp=opponent,minutes=minutes,status='Projected' if minutes else 'Out'))
                base.append(dict(PLAYER_ID=pid,Player=name,Team=team,**{'Projected minutes':minutes}))
        roster=pd.DataFrame(roster);base=pd.DataFrame(base)
        snapshot=browser()|dict(rows=rows,scenario='Default',scope='All games')
        initial,_,cache,_=tabs_minutes.resolve_browser_result({'ok':True,'snapshot':snapshot},None,roster,games,NOW)
        original=[minutes_plan(base[base.Team.isin([g['Away'],g['Home']])],initial,g,NOW)[0] for g in games]
        selected=copy.deepcopy(snapshot);selected['rows']=selected['rows'][:7]
        selected['rows'][0]['minutes']=35;selected['rows'][1]['minutes']=45
        current,_,cache,_=tabs_minutes.resolve_browser_result({'ok':True,'snapshot':selected},cache,roster,games,NOW)
        self.assertEqual(len(current),210);self.assertEqual(len(cache['snapshots']),30)
        self.assertEqual(set(current.loc[~current.source_saved,'Team']),{'ATL'})
        self.assertEqual(current.loc[current.PLAYER_ID.eq(1),'minutes'].iloc[0],35)
        partial=selected|dict(rows=[selected['rows'][0]|dict(minutes=42)])
        current,_,cache,_=tabs_minutes.resolve_browser_result({'ok':True,'snapshot':partial},cache,roster,games,NOW)
        self.assertEqual(len(current),210);self.assertTrue(current.source_saved.all())
        self.assertEqual(current.loc[current.PLAYER_ID.eq(1),'minutes'].iloc[0],35)
        # On refresh a raw filtered reply can arrive before the restore callback.
        current,_,cache,_=tabs_minutes.resolve_browser_result({'ok':True,'snapshot':partial,'saved_snapshot':cache},None,roster,games,NOW)
        self.assertEqual(len(current),210)
        restored,_,cache,_=tabs_minutes.resolve_browser_result({'ok':False,'saved_snapshot':cache},None,roster,games,NOW+pd.Timedelta(hours=4))
        for i,g in enumerate(games):
            plan,meta=minutes_plan(base[base.Team.isin([g['Away'],g['Home']])],restored,g,NOW+pd.Timedelta(hours=4))
            self.assertEqual(len(plan['blended']),14);self.assertTrue(meta['Tabs minutes saved'].all())
            if i: self.assertEqual(plan,original[i])
        self.assertEqual(restored.loc[restored.source_status.eq('Out'),'minutes'].sum(),0)
        fresh,_,_,_=tabs_minutes.resolve_browser_result({'ok':True,'snapshot':snapshot},cache,roster,games,NOW)
        self.assertFalse(fresh.source_saved.any())

    def test_filtered_team_cannot_replace_complete_snapshot_with_unmapped_player_or_old_update(self):
        rows=[dict(Player=f'Player {i}',Team='POR',Opp='MIN',minutes=40,status='Projected') for i in range(6)]
        roster=pd.DataFrame([dict(PLAYER_ID=i,PLAYER_NAME=r['Player'],TEAM_ABBREVIATION='POR') for i,r in enumerate(rows)])
        snapshot=browser()|dict(rows=rows)
        _,_,cache,_=tabs_minutes.resolve_browser_result({'ok':True,'snapshot':snapshot},None,roster,[GAME],NOW)
        for changed in [snapshot|dict(rows=rows[:-1]+[rows[-1]|dict(Player='Unknown')]),
                        snapshot|dict(source_updated_at=(NOW-pd.Timedelta(minutes=10)).isoformat())]:
            kept,_,_,_=tabs_minutes.resolve_browser_result({'ok':True,'snapshot':changed},cache,roster,[GAME],NOW)
            self.assertEqual(kept.minutes.tolist(),[40]*6);self.assertTrue(kept.source_saved.all())

    def test_saved_snapshot_stays_bound_to_its_game_and_must_have_been_valid_at_capture(self):
        next_game=GAME|dict(GameTimeUTC=GAME['GameTimeUTC']+pd.Timedelta(days=1))
        self.assertIsNone(tabs_minutes.resolve_browser_result(None,browser(),ROSTER,[next_game],NOW+pd.Timedelta(days=1))[0])
        for changes in [dict(manual_edits=True),dict(source_url='https://other.example'),
                        dict(source_updated_at=(NOW-pd.Timedelta(hours=3)).isoformat()),
                        dict(observed_at=(NOW+pd.Timedelta(days=3)).isoformat())]:
            self.assertIsNone(tabs_minutes.resolve_browser_result(None,browser()|changes,ROSTER,[GAME],NOW)[0])

    def test_real_minutes_out_scenario_disconnect_and_stale_changes_refresh_projections(self):
        original=browser()
        refs,_=parse_tabs_browser_snapshot(original,ROSTER,[GAME],NOW)
        cases=[]
        for field,value in [('minutes',20),('status','Out')]:
            changed=copy.deepcopy(original)
            changed['rows'][0][field]=value
            if field=='status':changed['rows'][0]['minutes']=0
            cases.append({'ok':True,'snapshot':changed})
        cases.extend([{'ok':True,'snapshot':original|{'scenario':'Default'}},
                      {'ok':True,'snapshot':original|{'rows':original['rows'][:1]}},
                      {'ok':True,'snapshot':original|{'source_updated_at':(NOW-pd.Timedelta(hours=3)).isoformat()}},
                      {'ok':False,'error':'Source tab closed'}])
        real_parse=parse_tabs_browser_snapshot
        for result in cases:
            with self.subTest(result=result):
                state={'tabs_minutes_reference':refs,'tabs_minutes_browser':{'result':result}}
                with patch.object(tabs_minutes.st,'session_state',state), \
                     patch.object(tabs_minutes,'parse_tabs_browser_snapshot',side_effect=lambda s,r,g,a=None,**kw:real_parse(s,r,g,NOW,**kw)), \
                     patch.object(tabs_minutes.st,'rerun') as rerun:
                    tabs_minutes._remember_browser_result(ROSTER,[GAME])
                    rerun.assert_called_once_with(scope='projection_dashboard')

if __name__=='__main__':unittest.main()
