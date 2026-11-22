import copy
import unittest
from datetime import date

import numpy as np
import pandas as pd

from tabs_minutes import parse_tabs_browser_snapshot, parse_tabs_minutes, resolve_browser_result
from tabs_projection_review import add_tabs_projection_review, tabs_projection_check
from edge_rating_table import table_payload
from edge_review import filter_review

NOW = pd.Timestamp('2026-11-22T21:00:00Z')
GAME = dict(id='demo', Away='BOS', Home='SAC', GameTimeUTC=NOW+pd.Timedelta(hours=3))
ROSTER = pd.DataFrame([dict(PLAYER_ID=1, PLAYER_NAME='Example Wing', TEAM_ABBREVIATION='BOS')])


def snapshot():
    return dict(schema=1, source='TabsNBA', source_url='https://app.tabsnba.com/projections',
        slate_date='2026-11-22', scenario='Default', source_updated_at=(NOW-pd.Timedelta(minutes=5)).isoformat(),
        observed_at=NOW.isoformat(), manual_edits=False, rows=[dict(Player='Example Wing', Team='BOS', Opp='SAC',
            minutes=30, status='Projected', PTS=18, REB=5, AST=4, **{'3PM':2.2, 'STL':.8, 'BLK':.3, 'TOV':1.4})])


def edge(**kwargs):
    return pd.Series(dict(PLAYER_ID=1, Player='Example Wing', Team='BOS', Opponent='SAC', Stat='3PM',
        Play='Over', Projected=3.2, **{'Book line':2.5, 'Book implied total':2.1, 'Over odds':-110,
        'Under odds':-110, 'Edge %':20., 'Signed edge %':20., 'Robust play':True, 'Review group':'More robust',
        'Exp. P(over)':.65}) | kwargs)


class TabsProjectionTests(unittest.TestCase):
    def refs(self, data=None):
        return parse_tabs_browser_snapshot(data or snapshot(), ROSTER, [GAME], NOW)[0]

    def test_opposite_sides_use_actual_line_not_book_implied_total(self):
        refs=self.refs()
        over=tabs_projection_check(edge(),refs,GAME,NOW)
        self.assertEqual(over['Tabs projection flag'],'OPPOSES')
        self.assertIn('Tabs projects 2.20 3PM — below the 2.5 line.',over['Tabs projection detail'])
        self.assertIn('Our projection: 3.20',over['Tabs projection detail'])
        under=tabs_projection_check(edge(Play='Under',Projected=1.5,**{'Book line':2.0}),refs,GAME,NOW)
        self.assertEqual(under['Tabs projection flag'],'OPPOSES')
        for side,line,status in [('Over',2.0,'Agrees'),('Under',2.5,'Agrees'),('Over',2.2,'Neutral'),('Under',2.2,'Neutral')]:
            with self.subTest(side=side,line=line):
                self.assertEqual(tabs_projection_check(edge(Play=side,**{'Book line':line}),refs,GAME,NOW)['Tabs projection flag'],status)

    def test_every_supported_stat_and_pra_uses_matching_stat_totals(self):
        refs=self.refs()
        expected={'PTS':18,'REB':5,'AST':4,'3PM':2.2,'STL':.8,'BLK':.3,'TOV':1.4,'PRA':27}
        for stat,value in expected.items():
            result=tabs_projection_check(edge(Stat=stat,**{'Book line':value+.5}),refs,GAME,NOW)
            self.assertEqual(result['Tabs stat projection'],value)
            self.assertEqual(result['Tabs projection flag'],'OPPOSES')
        refs['Tabs projection AST']=np.nan
        self.assertEqual(tabs_projection_check(edge(Stat='PRA'),refs,GAME,NOW)['Tabs projection flag'],'Unavailable')

    def test_missing_stale_wrong_game_duplicate_and_out_never_flag(self):
        refs=self.refs()
        cases=[None, refs.iloc[:0],pd.concat([refs,refs]), refs.assign(Team='LAL'),
               refs.assign(Opponent='LAC'),refs.assign(game_date='2026-11-21'),
               refs.assign(source='Other'),refs.assign(minutes=0),refs.assign(source_status='Out'),
               refs.assign(source_saved=True),refs.assign(source_updated_at=(NOW-pd.Timedelta(hours=3)).isoformat()),
               refs.assign(source_updated_at=(NOW+pd.Timedelta(minutes=1)).isoformat()),
               refs.assign(**{'Tabs projection 3PM':np.nan}),refs.assign(**{'Tabs projection 3PM':-1}),
               refs.assign(**{'Tabs projection 3PM':np.inf})]
        for i,case in enumerate(cases):
            with self.subTest(i=i):
                self.assertEqual(tabs_projection_check(edge(),case,GAME,NOW)['Tabs projection flag'],'Unavailable')

    def test_csv_edits_quarantine_stats_but_keep_independent_minutes(self):
        header='Player,Team,Opp,Tabs Min,My Min,Boost %,PTS,REB,AST,3PM\n'
        def parse(mine='30',boost='0'):
            raw=(header+f'Example Wing,BOS,SAC,30,{mine},{boost},18,5,4,2.2\n').encode()
            return parse_tabs_minutes(raw,'Tabs_DK_2026-11-22.csv',ROSTER,[GAME],NOW,'Default',NOW)[0]
        self.assertEqual(parse()['Tabs projection PTS'].iloc[0],18)
        for mine,boost in [('32','0'),('30','10'),('30','-10'),('','0'),('30','')]:
            refs=parse(mine,boost)
            self.assertEqual(refs.minutes.iloc[0],30)
            self.assertTrue(refs['Tabs projection PTS'].isna().all())

    def test_stat_only_refresh_and_saved_reconnect_preserve_source_truth(self):
        data=snapshot()
        refs,_,saved,_=resolve_browser_result({'ok':True,'snapshot':data},None,ROSTER,[GAME],NOW)
        changed=copy.deepcopy(data);changed['rows'][0]['3PM']=3.1
        new,_,backup,_=resolve_browser_result({'ok':True,'snapshot':changed},saved,ROSTER,[GAME],NOW)
        self.assertEqual(new['Tabs projection 3PM'].iloc[0],3.1)
        retained,_,_,_=resolve_browser_result({'ok':False},backup,ROSTER,[GAME],NOW)
        self.assertEqual(retained['Tabs projection 3PM'].iloc[0],3.1)
        self.assertEqual(tabs_projection_check(edge(),retained,GAME,NOW)['Tabs projection flag'],'Unavailable')
        old=snapshot();old['rows'][0]={k:v for k,v in old['rows'][0].items() if k in ('Player','Team','Opp','minutes','status')}
        self.assertEqual(self.refs(old).minutes.iloc[0],30)
        self.assertEqual(tabs_projection_check(edge(),self.refs(old),GAME,NOW)['Tabs projection flag'],'Unavailable')

    def test_review_preserves_model_values_order_and_reaches_both_badges(self):
        rows=pd.DataFrame([edge(),edge(Play='Under',Projected=1.8)])
        original=rows.copy(deep=True)
        result=add_tabs_projection_review(rows,self.refs(),GAME,NOW)
        model_columns=original.columns.drop(['Robust play','Review group'])
        pd.testing.assert_frame_equal(result[model_columns],original[model_columns])
        pd.testing.assert_frame_equal(rows,original)
        self.assertEqual(result['Robust play'].tolist(),[False,True])
        payload=table_payload(result,['Rank','Player','Stat','Play','Projected','Edge %'],ROSTER,{},'BOS',GAME,'FanDuel',date(2026,11,22))
        badges=[r['reviewBadges'] for r in payload['rows']]
        self.assertEqual(badges[0][0]['label'],'Tabs disagrees')
        self.assertIn('Scenario: Default',badges[0][0]['technical'])
        self.assertEqual(badges[1][0]['label'],'Tabs agrees')
        self.assertEqual(badges[1][0]['color'],'tabs-agrees')
        self.assertTrue(add_tabs_projection_review(rows.iloc[:0],None,GAME,NOW).empty)

    def test_neutral_badge_is_distinct_from_missing_data(self):
        rows=pd.DataFrame([edge(**{'Book line':2.2}),edge(Stat='AST')])
        refs=self.refs();refs['Tabs projection AST']=np.nan
        result=add_tabs_projection_review(rows,refs,GAME,NOW)
        self.assertEqual(result['Tabs projection flag'].tolist(),['Neutral','Unavailable'])
        payload=table_payload(result,['Rank','Player','Stat','Play','Projected','Edge %'],ROSTER,{},'BOS',GAME,'FanDuel',date(2026,11,22))
        self.assertEqual(payload['rows'][0]['reviewBadges'][0]['label'],'Tabs neutral')
        self.assertEqual(payload['rows'][0]['reviewBadges'][0]['color'],'tabs-neutral')
        self.assertFalse(any(b['label'].startswith('Tabs ') for b in payload['rows'][1]['reviewBadges']))
        self.assertFalse(result['Robust play'].any())
        self.assertEqual(result['Review group'].tolist(),['Needs review','Unchecked'])

    def test_agreement_is_required_in_addition_to_all_existing_checks(self):
        rows=pd.DataFrame([
            edge(**{'Book line':2.0}),
            edge(Play='Under'),
            edge(**{'Book line':2.0,'Robust play':False,'Review group':'Waiting on news','Injury flag':'Q'}),
            edge(**{'Book line':2.0,'Robust play':False,'Review group':'Needs review','Minutes flag':'MIN'}),
            edge(**{'Book line':2.0,'Robust play':False,'Review group':'Unchecked','Role flag':'Unchecked'}),
            edge(),
            edge(**{'Book line':2.2}),
            edge(Stat='AST'),
        ],index=[8,2,7,5,4,9,1,6])
        rows['Review flags']='existing';rows['Review detail']='existing detail'
        refs=self.refs();refs['Tabs projection AST']=np.nan
        result=add_tabs_projection_review(rows,refs,GAME,NOW)
        self.assertEqual(result['Robust play'].tolist(),[True,True,False,False,False,False,False,False])
        self.assertEqual(filter_review(result,'More robust plays').index.tolist(),[8,2])
        self.assertEqual(filter_review(result,'All plays').index.tolist(),rows.index.tolist())
        self.assertEqual(filter_review(result,'Waiting on news').index.tolist(),[7])
        self.assertEqual(filter_review(result,'Needs review / unchecked').index.tolist(),[5,4,9,1,6])
        for idx in [7,5,4]:
            for col in ['Review group','Review flags','Review detail']:
                self.assertEqual(result.loc[idx,col],rows.loc[idx,col])
        self.assertEqual(result.loc[9,'Review flags'],'Review')
        self.assertIn('Tabs favors the opposite side',result.loc[9,'Review detail'])
        self.assertIn('exactly on the line',result.loc[1,'Review detail'])
        self.assertIn('still needed',result.loc[6,'Review detail'])

    def test_stat_refresh_rechecks_robust_eligibility_without_changing_cached_review(self):
        original=pd.DataFrame([edge()])
        refs=self.refs()
        first=add_tabs_projection_review(original,refs,GAME,NOW)
        self.assertFalse(first['Robust play'].iloc[0])
        refs['Tabs projection 3PM']=3.1
        refreshed=add_tabs_projection_review(original,refs,GAME,NOW)
        self.assertTrue(refreshed['Robust play'].iloc[0])
        self.assertTrue(original['Robust play'].iloc[0])
        self.assertEqual(refreshed['Projected'].iloc[0],3.2)


if __name__=='__main__':unittest.main()
