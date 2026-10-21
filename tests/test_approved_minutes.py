from copy import deepcopy
import json
import unittest
import pandas as pd
from approved_minutes import digest, validate_bundle, for_matchup, projection_gaps
from live_projections import build_projections, STAT_NAMES
from import_minutes import TEAM_IDS


def example():
    away,home = TEAM_IDS['BOS'],TEAM_IDS['DET']
    key=f'2026-10-21_{away}_{home}'
    teams=[];roster=[]
    for name,tid,opp,offset in [('BOS',away,home,0),('DET',home,away,10)]:
        players=[]
        for i in range(7):
            pid=offset+i+1;minutes=40. if i<6 else 0.
            players.append(dict(player_id=pid,player_name=f'Player {pid}',minutes=minutes,starter=i<5,injury_status='Available'))
            roster.append(dict(PLAYER_ID=pid,PLAYER_NAME=f'Player {pid}',TEAM_ID=tid,TEAM_ABBREVIATION=name,
                PROJECTED_MIN=30.,OffArchetype='Guard',DefArchetype='Guard'))
        teams.append(dict(schema='nba_approved_minutes_v1',game_date='2026-10-21',game_key=key,team_id=tid,team=name,
            opponent_id=opp,players=players,approved_at='2026-10-21T12:00:00+00:00',approved_by='test',approval_version=1,warnings=[]))
    bundle=dict(schema='nba_approved_slate_v1',games=[dict(game_key=key,game_date='2026-10-21',away_team_id=away,home_team_id=home)],approvals=teams)
    bundle['bundle_id']=digest(bundle)
    return bundle,pd.DataFrame(roster)


def reseal(bundle):
    bundle['bundle_id']=digest({k:v for k,v in bundle.items() if k!='bundle_id'})
    return bundle


class ApprovalTests(unittest.TestCase):
    def test_coverage_uses_approved_active_players_and_preserves_profiles(self):
        bundle, roster = example()
        profiles = roster.rename(columns={'PLAYER_NAME':'Player', 'TEAM_ABBREVIATION':'Team',
            'OffArchetype':'Offensive archetype', 'DefArchetype':'Defensive archetype'}).assign(
                Status='Available', **{'Stat baseline':'2025–26 full regular season'})
        profiles.loc[profiles.PLAYER_ID.eq(1), 'Offensive archetype'] = None
        profiles.loc[profiles.PLAYER_ID.eq(2), 'Status'] = 'Needs statistical profile'
        profiles.loc[profiles.PLAYER_ID.eq(2), 'Stat baseline'] = 'Unavailable — no statistical profile'
        profiles.loc[profiles.PLAYER_ID.eq(3), 'Status'] = 'Needs current-season minutes (no matched season appearances)'
        profiles.loc[profiles.PLAYER_ID.eq(7), 'Defensive archetype'] = None
        before = profiles.copy(deep=True)
        gaps = projection_gaps(bundle, profiles)
        self.assertEqual(gaps.Player.tolist(), ['Player 1', 'Player 2'])
        self.assertEqual(gaps['Approved minutes'].tolist(), [40., 40.])
        self.assertEqual(gaps['Missing data'].tolist(), ['offensive archetype', 'Needs statistical profile'])
        self.assertTrue(projection_gaps(None, profiles).empty)
        pd.testing.assert_frame_equal(profiles, before)

    def test_round_trip_and_exact_pairing_eastern_date(self):
        original,roster=example();bundle=validate_bundle(json.dumps(original),roster,'2026-27')
        expected={p['player_id']:p['minutes'] for a in bundle['approvals'] for p in a['players']}
        self.assertEqual(for_matchup(bundle,roster,'DET','BOS','2026-10-22T01:00:00Z'),expected)
        for opponent,start in [('BOS','2026-10-23T01:00:00Z'),('CHI','2026-10-22T01:00:00Z'),('BOS',None),('BOS','2026-10-21T18:00:00')]:
            self.assertEqual(for_matchup(bundle,roster,'DET',opponent,start),{})
        self.assertEqual(bundle,original)

    def test_bad_or_incomplete_files_never_apply(self):
        base,roster=example()
        cases=[]
        bad=deepcopy(base);bad['approvals'].pop();cases.append(bad)
        bad=deepcopy(base);bad['approvals'][0]['players'][0]['minutes']=39;cases.append(bad)
        bad=deepcopy(base);bad['approvals'][0]['players'][0]['player_id']=99999;cases.append(bad)
        bad=deepcopy(base);bad['approvals'][0]['players'][0]['injury_status']='Out';cases.append(bad)
        bad=deepcopy(base);bad['approvals'][0]['approval_version']=0;cases.append(bad)
        bad=deepcopy(base);bad['approvals'][0]['players'][1]['player_id']=1;cases.append(bad)
        for bad in cases:
            with self.assertRaises(ValueError):validate_bundle(reseal(bad),roster,'2026-27')
        with self.assertRaises(ValueError):validate_bundle(base,roster,'2025-26')
        bad=deepcopy(base);bad['approvals'][0]['players'][0]['minutes']=0
        with self.assertRaisesRegex(ValueError,'checksum'):validate_bundle(bad,roster,'2026-27')

    def test_approved_minutes_override_both_policies_preserve_exact_allocation(self):
        bundle,roster=example()
        minutes=for_matchup(bundle,roster,'BOS','DET','2026-10-21T23:00:00Z')
        minutes[1]=39.9;minutes[2]=40.1
        history=pd.DataFrame([dict(PLAYER_ID=p,MIN=20.,**{s:10. for s in STAT_NAMES}) for p in roster.PLAYER_ID])
        current=pd.DataFrame([dict(PLAYER_NAME=f'Player {p}',ESPN_PLAYER_ID=str(p),GP=4,MIN=25.,**{s:10. for s in STAT_NAMES}) for p in roster.PLAYER_ID])
        for completed in (0,8):
            teams=pd.DataFrame([dict(TEAM_ID=TEAM_IDS[n],GAMES_PLAYED=completed) for n in ['BOS','DET']])
            before=roster.copy(deep=True)
            projection=build_projections(roster,history,teams,current,approved_minutes=minutes)
            self.assertEqual(projection.groupby('Team')['Projected minutes'].sum().tolist(),[240.,240.])
            self.assertTrue(projection['Minutes source'].eq('Approved game minutes').all())
            self.assertEqual(projection.iloc[0]['Projected minutes'],39.9)
            self.assertEqual(projection[projection.PLAYER_ID.eq(7)].iloc[0].PTS,0.)
            pd.testing.assert_frame_equal(roster,before)
