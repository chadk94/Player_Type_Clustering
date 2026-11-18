import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from edge_review import (assess_edges, unavailable_review, side_edge, rebalance_minutes,
                         scenario_projection, role_evidence, minute_reference, filter_review)
from edge_review_sources import parse_minutes, review_passing
from sportsbook_injuries import TeamInjuries
from market_points import apply_market_points
from team_assists import apply_team_assists
from team_threes import apply_team_threes
from team_rebounds import apply_team_rebounds

NOW=pd.Timestamp('2026-11-17T20:00:00Z')
GAME=dict(Away='BOS',Home='DET',GameTimeUTC=NOW+pd.Timedelta(hours=4))


def raw_team():
    f=pd.DataFrame(dict(PLAYER_ID=range(1,10),Player=[f'Player {i}' for i in range(1,10)],Team='BOS',Opponent='DET',
        **{'Projected minutes':[30.]*8+[0.]}))
    for s,n in dict(PTS=15,AST=4,FG3M=2,FG3A=5,OREB=1,DREB=4,STL=1,BLK=.5).items():f[s]=[n]*8+[0.]
    f['REB']=f.OREB+f.DREB;f['PRA']=f.PTS+f.REB+f.AST
    return f


def finalized(f):
    rates=pd.DataFrame({'Assist rate':.6,'Allowed assist rate':.6,'Points per FGM':2.5,'Allowed points per FGM':2.5,
        'Three attempt rate':.4,'Allowed three attempt rate':.4,'Points per FGA':1.2,'Allowed points per FGA':1.2,
        'FG misses per FGM':1.,'FT misses per FGM':.1,'Player credited fraction':.8,'Offensive rebound share':.25},index=['BOS','DET'])
    targets={'BOS':120.,'DET':110.};opp={'BOS':'DET','DET':'BOS'}
    f,_,_=apply_market_points(f,targets)
    f,_,_=apply_team_assists(f,targets,rates,opp)
    f,_,_=apply_team_threes(f,targets,rates,opp)
    f,_,_=apply_team_rebounds(f,targets,rates,rates,opp)
    return f


def edge(mean=15.,implied=10.,side='Over',stat='PTS'):
    return pd.DataFrame([dict(PLAYER_ID=1,Player='Player 1',Team='BOS',Opponent='DET',Stat=stat,Play=side,
        Projected=mean,Rank=1,**{'Projected minutes':30.,'Book implied total':implied,'EV %':12.,'Edge %':abs(mean/implied-1)*100,
                               'Exp. P(over)':.65,'Exp. P(under)':.35,'Book line':9.5,'Over odds':-110,'Under odds':-105})])


def reference(minutes=30.):
    return pd.DataFrame([dict(PLAYER_ID=1,Team='BOS',Opponent='DET',game_date='2026-11-17',minutes=minutes,
        source='independent',updated_at=NOW.isoformat(),independent=True)])


def healthy():return {'BOS':TeamInjuries(True,True),'DET':TeamInjuries(True,True)}


def logs():
    return pd.DataFrame([dict(ESPN_PLAYER_ID='11',PLAYER_NAME='Player 1',Team='BOS',GAME_ID=str(i),GAME_DATE=f'2026-11-{i:02d}T12:00:00Z',
                             MIN=30,FGA=10,FGM=4,FG3A=4,FG3M=1,FTA=2,FTM=1,AST=3) for i in range(1,11)])


class ReviewTests(unittest.TestCase):
    def test_full_rotation_scenarios_match_real_recalibration_and_preserve_original(self):
        raw=raw_team();original=finalized(raw); saved=original.copy(deep=True)
        minutes=rebalance_minutes(original,1,18.)
        self.assertEqual(minutes[9],0.);self.assertAlmostEqual(sum(minutes.values()),240.)
        self.assertTrue(all(0<=m<=48 for m in minutes.values()))
        changed=raw.copy();stats=['PTS','AST','FG3M','FG3A','OREB','DREB','REB','STL','BLK','PRA']
        for i,p in changed.iterrows():
            if p['Projected minutes']>0:changed.loc[i,stats]*=minutes[p.PLAYER_ID]/p['Projected minutes']
            changed.loc[i,'Projected minutes']=minutes[p.PLAYER_ID]
        expected=finalized(changed);actual=scenario_projection(original,minutes)
        np.testing.assert_allclose(actual[['PTS','AST','REB','FG3M','PRA','STL','BLK']],expected[['PTS','AST','REB','FG3M','PRA','STL','BLK']])
        pd.testing.assert_frame_equal(saved,original)
        pd.testing.assert_frame_equal(scenario_projection(original)[['PTS','AST']],original[['PTS','AST']])
        role=scenario_projection(original,role={1:{'PTS':.5,'AST':.6,'FG3A':.5}})
        self.assertAlmostEqual(role.PTS.sum(),original.PTS.sum());self.assertAlmostEqual(role.AST.sum(),original.AST.sum())
        self.assertAlmostEqual(role.REB.sum(),original.REB.sum())
        self.assertAlmostEqual(role.FG3M.sum(),original.FG3M.sum()) # equal shooting conversion fixture
        np.testing.assert_allclose(role.PRA,role.PTS+role.AST+role.REB)

    def test_invalid_rotation_does_not_invent_minutes_or_revive_excluded_players(self):
        f=raw_team();f.loc[0,'Projected minutes']=29
        with self.assertRaises(ValueError):rebalance_minutes(f,1,20)
        with self.assertRaises(ValueError):rebalance_minutes(raw_team(),9,20)
        with self.assertRaises(ValueError):rebalance_minutes(raw_team().iloc[:5].assign(**{'Projected minutes':48}),1,47)

    def test_direction_threshold_and_model_invariance(self):
        f=finalized(raw_team());e=edge();before=e.copy(deep=True)
        signal=dict(ratios={'PTS':1.},trends={'PTS':0.},detail='Stable')
        with patch('edge_review.role_evidence',return_value=signal):
            r=assess_edges(e,f,GAME,pd.DataFrame(),{},reference(),healthy(),NOW,NOW)
            self.assertTrue(r.iloc[0]['Robust play'])
            pd.testing.assert_frame_equal(r[e.columns],before)
            self.assertLess(r.iloc[0]['Minutes test projection'],e.iloc[0].Projected)
            # Equality at 5% never qualifies, and a reversed direction never uses abs().
            r=assess_edges(edge(10.5,10),f,GAME,pd.DataFrame(),{},reference(),healthy(),NOW,NOW)
            self.assertFalse(r.iloc[0]['Robust play'])
            under=assess_edges(edge(15,20,'Under'),f,GAME,pd.DataFrame(),{},reference(30),healthy(),NOW,NOW)
            self.assertGreater(under.iloc[0]['Minutes test projection'],15)
        self.assertAlmostEqual(side_edge(12,10,'Under'),-20)
        self.assertAlmostEqual(side_edge(8,10,'Under'),20)

    def test_missing_stale_wrong_game_and_dependent_sources_are_not_clear(self):
        row=edge().iloc[0]
        for changes in [dict(independent=False),dict(updated_at=(NOW-pd.Timedelta(hours=7)).isoformat()),
                        dict(updated_at=(NOW+pd.Timedelta(seconds=1)).isoformat()),dict(Opponent='NYK'),dict(game_date='2026-11-16')]:
            self.assertIsNone(minute_reference(reference().assign(**changes),row,GAME,NOW))
        self.assertIsNone(minute_reference(pd.concat([reference(),reference()]),row,GAME,NOW))
        r=assess_edges(edge(),finalized(raw_team()),GAME,pd.DataFrame(),{},None,healthy(),NOW,NOW)
        self.assertFalse(r.iloc[0]['Robust play']);self.assertEqual(r.iloc[0]['Review group'],'Unchecked')
        fallback=unavailable_review(edge());pd.testing.assert_frame_equal(fallback[edge().columns],edge())

    def test_combined_stress_can_fail_even_when_each_individual_test_passes(self):
        signal=dict(ratios={'PTS':.9},trends={'PTS':-.05},detail='Small adverse shift')
        with patch('edge_review.role_evidence',return_value=signal):
            row=assess_edges(edge(15,12.6),finalized(raw_team()),GAME,pd.DataFrame(),{},reference(),healthy(),NOW,NOW).iloc[0]
        self.assertGreater(row['Minutes test edge %'],5)
        self.assertGreater(row['Recent test edge %'],5)
        self.assertLess(row['Combined test edge %'],5)
        self.assertFalse(row['Robust play']);self.assertEqual(row['Review group'],'Needs review')

    def test_recent_role_uses_nonoverlapping_volume_not_makes(self):
        f=finalized(raw_team()).iloc[0];observed=pd.Series(dict(ESPN_PLAYER_ID='11',GP=10))
        sample=logs();sample.loc[sample.index>=5,'FG3M']=4
        evidence=role_evidence(f,observed,sample,NOW)
        self.assertEqual(evidence['ratios']['FG3A'],1.)
        self.assertNotIn('AST',evidence['ratios'])
        sample.loc[sample.index>=5,'FG3A']=2;sample.loc[sample.index>=5,'FG3M']=1
        self.assertAlmostEqual(role_evidence(f,observed,sample,NOW)['ratios']['FG3A'],2/3)
        passing=pd.DataFrame(dict(PLAYER_ID=1,GAME_ID=sample.GAME_ID,POTENTIAL_AST=[8]*5+[4]*5,PASSES_MADE=[40]*10))
        self.assertAlmostEqual(role_evidence(f,observed,sample,NOW,passing)['ratios']['AST'],(2/3+1)/2)
        self.assertNotIn('AST',role_evidence(f,observed,sample,NOW,passing.iloc[:-1])['ratios'])
        self.assertEqual(role_evidence(f,observed,sample.iloc[:7],NOW)['ratios'],{})
        self.assertEqual(role_evidence(f,observed,pd.concat([sample,sample.iloc[:1]]),NOW)['ratios'],{})
        self.assertIn('PTS',role_evidence(f,observed,sample,pd.Timestamp('2027-01-01T00:00:00Z'))['ratios'])

    def test_injury_uncertainty_needs_review_and_does_not_guess_projections(self):
        info=healthy();info['DET'].questionables=[('BOS@DET','DET','Key teammate',30,30)]
        r=assess_edges(edge(),finalized(raw_team()),GAME,pd.DataFrame(),{},reference(),info,NOW,NOW)
        self.assertEqual(r.iloc[0]['Injury flag'],'Q');self.assertFalse(r.iloc[0]['Robust play'])
        self.assertEqual(r.iloc[0]['Review group'],'Waiting on news')
        self.assertIn('No verified in/out',r.iloc[0]['Injury detail'])
        self.assertNotIn('Injury in projection',r)
        info=healthy();info['BOS'].statuses={'Player 1':'Questionable'} # below key-player threshold, still own risk
        self.assertEqual(assess_edges(edge(),finalized(raw_team()),GAME,pd.DataFrame(),{},reference(),info,NOW,NOW).iloc[0]['Injury flag'],'Q')
        info=healthy();info['BOS'].statuses={'Player 1':'Out'}
        self.assertEqual(assess_edges(edge(),finalized(raw_team()),GAME,pd.DataFrame(),{},reference(),info,NOW,NOW).iloc[0]['Injury flag'],'Q')
        stale=assess_edges(edge(),finalized(raw_team()),GAME,pd.DataFrame(),{},reference(),healthy(),NOW-pd.Timedelta(minutes=31),NOW)
        self.assertEqual(stale.iloc[0]['Injury flag'],'Unchecked')

    def test_teammate_q_is_directional_and_preserves_every_model_value(self):
        frame=finalized(raw_team()); saved=frame.copy(deep=True)
        info=healthy()
        info['BOS'].statuses={'Player 2':'Questionable'}
        info['BOS'].key_minutes={'Player 2':(30.,30.)}
        signal=dict(ratios={'PTS':1.,'AST':1.,'FG3A':1.},trends={'PTS':0.,'AST':0.,'FG3A':0.},detail='Stable')
        with patch('edge_review.role_evidence',return_value=signal):
            for stat,mean in [('PTS',15.),('PRA',24.),('3PM',2.)]:
                for side in ['Over','Under']:
                    edges=edge(mean,mean*.6 if side=='Over' else mean*1.5,side,stat)
                    before=edges.copy(deep=True)
                    result=assess_edges(edges,frame,GAME,pd.DataFrame(),{},reference(),info,NOW,NOW)
                    row=result.iloc[0]
                    self.assertEqual(row['Injury flag'],'Clear' if side=='Over' else 'Q')
                    self.assertEqual(row['Review group'],'More robust' if side=='Over' else 'Waiting on news')
                    self.assertEqual(row['Robust play'],side=='Over')
                    self.assertIn('already included' if side=='Over' else 'could gain',row['Injury detail'])
                    pd.testing.assert_frame_equal(result[edges.columns],before)
                    pd.testing.assert_frame_equal(edges,before)
        pd.testing.assert_frame_equal(frame,saved)

    def test_q_rotation_assumptions_unknowns_and_multiple_risks(self):
        frame=finalized(raw_team())
        info=healthy();info['BOS'].statuses={'Player 9':'Questionable'}
        # A key player excluded from this game's rotation can return and hurt an Over.
        info['BOS'].key_minutes={'Player 9':(0.,30.)}
        def review(info=info,frame=frame,stamp=NOW):
            return assess_edges(edge(),frame,GAME,pd.DataFrame(),{},reference(),info,stamp,NOW).iloc[0]
        self.assertEqual(review()['Injury flag'],'Q')
        self.assertIn('currently excluded',review()['Injury detail'])
        info['BOS'].statuses={'Missing teammate':'Questionable'}
        for minutes in [None,float('nan'),float('inf'),-1.,49.]:
            info['BOS'].key_minutes={'Missing teammate':None if minutes is None else (minutes,30.)}
            self.assertEqual(review()['Injury flag'],'Q')
            self.assertIn('could not be verified',review()['Injury detail'])
        # Valid game minutes can still be used when a zero-stat player is absent from final rows.
        info['BOS'].key_minutes={'Missing teammate':(30.,30.)}
        self.assertEqual(review()['Injury flag'],'Clear')
        self.assertEqual(review(stamp=NOW-pd.Timedelta(minutes=31))['Injury flag'],'Unchecked')
        info['DET'].complete=False
        self.assertEqual(review()['Injury flag'],'Unchecked')
        info['DET'].complete=True
        info['DET'].statuses={'Opponent':'Questionable'}
        info['DET'].key_minutes={'Opponent':(30.,30.)}
        self.assertEqual(review()['Injury flag'],'Q')
        self.assertIn('Opponent availability',review()['Injury detail'])
        info['DET']=TeamInjuries(True,True)
        info['BOS'].statuses['Player 1']='Doubtful'
        self.assertIn('own availability',review()['Injury detail'])
        self.assertEqual(review()['Injury flag'],'Q')

    def test_own_status_conflicts_and_doubtful_teammates(self):
        frame=finalized(raw_team())
        for status in ['Questionable','Doubtful','Out']:
            for side in ['Over','Under']:
                info=healthy();info['BOS'].statuses={'Player 1':status}
                row=assess_edges(edge(side=side),frame,GAME,pd.DataFrame(),{},reference(),info,NOW,NOW).iloc[0]
                self.assertEqual(row['Injury flag'],'Q')
        info=healthy();info['BOS'].statuses={'Player 2':'Doubtful'}
        info['BOS'].key_minutes={'Player 2':(30.,30.)}
        row=assess_edges(edge(),frame,GAME,pd.DataFrame(),{},reference(),info,NOW,NOW).iloc[0]
        self.assertEqual(row['Injury flag'],'Clear')
        info['BOS'].statuses={'Player 2':'Out'}
        row=assess_edges(edge(),frame,GAME,pd.DataFrame(),{},reference(),info,NOW,NOW).iloc[0]
        self.assertEqual(row['Injury flag'],'Q')
        self.assertIn('still has projected minutes',row['Injury detail'])
        info['BOS'].statuses={'Player 9':'Out'}
        info['BOS'].key_minutes={'Player 9':(0.,30.)}
        # Resolved Out at zero minutes, and a Q in another game, should not warn this game.
        info['NYK']=TeamInjuries(True,True,{'Other game':'Questionable'},{'Other game':(30.,30.)})
        row=assess_edges(edge(),frame,GAME,pd.DataFrame(),{},reference(),info,NOW,NOW).iloc[0]
        self.assertEqual(row['Injury flag'],'Clear')

    def test_filters_preserve_order_ranks_and_empty_case(self):
        rows=pd.concat([unavailable_review(edge()),unavailable_review(edge())],ignore_index=True)
        rows.loc[0,'Rank']=7;rows.loc[1,'Rank']=2;rows.loc[1,'Robust play']=True
        self.assertEqual(filter_review(rows,'More robust plays').Rank.tolist(),[2])
        pd.testing.assert_frame_equal(rows,filter_review(rows,'All plays'))
        empty=assess_edges(edge().iloc[:0],finalized(raw_team()),GAME,pd.DataFrame(),{},as_of=NOW)
        self.assertTrue(filter_review(empty,'More robust plays').empty)

    def test_public_minutes_parser_never_uses_paywall_or_different_day(self):
        roster=pd.DataFrame([dict(PLAYER_ID=1,PLAYER_NAME='Player One',TEAM_ABBREVIATION='BOS')])
        html='''<div class="lineup proj-min"><img class="lineup__logo" alt="BOS"><li class="lineups-viz__player"><div class="lineups-viz__player-name"><a><span>Player One</span></a></div><div class="minutes-meter__range" data-proj="31"></div></li></div>'''
        rows=parse_minutes(html,roster,[GAME],NOW)
        self.assertEqual(rows.minutes.tolist(),[31.]);self.assertEqual(rows.PLAYER_ID.tolist(),[1])
        self.assertTrue(parse_minutes(html.replace('<li','<div class="lineups-viz-wall"></div><li'),roster,[GAME],NOW).empty)
        self.assertTrue(parse_minutes(html,roster,[GAME],NOW-pd.Timedelta(days=1)).empty)
        self.assertTrue(parse_minutes(html.replace('31','100'),roster,[GAME],NOW).empty)

    def test_passing_match_requires_boxscore_confirmation(self):
        sample=logs().iloc[:1];matches={1:pd.Series(dict(ESPN_PLAYER_ID='11'))}
        tracking=pd.DataFrame([dict(PLAYER_ID=1,GP=1,AST=3,MIN=30,POTENTIAL_AST=7,PASSES_MADE=40)]).set_index('PLAYER_ID')
        read=lambda season,days:{next(iter(days)):tracking}
        self.assertEqual(len(review_passing('2026-27',sample,matches,read)),1)
        tracking.loc[1,'AST']=4
        self.assertTrue(review_passing('2026-27',sample,matches,read).empty)

if __name__=='__main__':unittest.main()
