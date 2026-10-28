"""The saved-minute fix must work for every NBA team and both minute policies."""
import json
import unittest
import numpy as np
import pandas as pd

from approved_minutes import digest, validate_bundle, for_matchup
from approved_minutes_storage import new_state, accept_reply
from import_minutes import TEAM_IDS
from live_projections import build_projections, STAT_NAMES
from matchup_projections import apply_matchup
from team_projection_view import team_projection_tables


class LeagueCoverageTests(unittest.TestCase):
    def test_all_30_teams_restore_complete_rotations_and_keep_out_players_zero(self):
        names = sorted(TEAM_IDS)
        roster, history, current, approvals, games = [], [], [], [], []
        for index in range(0,len(names),2):
            away,home = names[index:index+2]
            game_key=f'2026-10-28_{TEAM_IDS[away]}_{TEAM_IDS[home]}'
            games.append(dict(game_date='2026-10-28',game_key=game_key,
                              away_team_id=TEAM_IDS[away],home_team_id=TEAM_IDS[home]))
            for side,opponent,offset in [(away,home,index),(home,away,index+1)]:
                players=[]
                for i in range(8):
                    pid=offset*100+i+1
                    # Six active players, one out player with positive workbook
                    # minutes but no appearances, and one zero-minute reserve.
                    minutes=40. if i<6 else 0.
                    player_name=f'{side} player {i}'
                    players.append(dict(player_id=pid,player_name=player_name,minutes=minutes,
                                        starter=i<5,injury_status='Out' if i==6 else 'Available'))
                    roster.append(dict(PLAYER_ID=pid,PLAYER_NAME=player_name,TEAM_ID=TEAM_IDS[side],
                        TEAM_ABBREVIATION=side,PROJECTED_MIN=0. if i==7 else 30.,
                        OffCluster=1,DefCluster=1,OffArchetype='Role Player',DefArchetype='Non-POA Wing'))
                    if i<6:
                        history.append(dict(PLAYER_ID=pid,MIN=20.,**{s:2. for s in STAT_NAMES}))
                        current.append(dict(PLAYER_NAME=player_name,ESPN_PLAYER_ID=str(pid),GP=5,MIN=25.,
                                            **{s:3. for s in STAT_NAMES}))
                approvals.append(dict(schema='nba_approved_minutes_v1',game_date='2026-10-28',game_key=game_key,
                    team_id=TEAM_IDS[side],team=side,opponent_id=TEAM_IDS[opponent],players=players,
                    approved_at='2026-10-28T12:00:00Z',approved_by='test',approval_version=1,warnings=[]))
        bundle=dict(schema='nba_approved_slate_v1',games=games,approvals=approvals)
        bundle['bundle_id']=digest(bundle)
        roster=pd.DataFrame(roster); history=pd.DataFrame(history); current=pd.DataFrame(current)
        state=new_state('2026-27')
        accept_reply(state,dict(request=state['request']['request'],status='ok',bundle=json.dumps(bundle)),roster,'2026-27')
        self.assertEqual(len(state['bundle']['approvals']),30)
        original=roster.copy(deep=True)
        for completed in (0,5):
            counts=pd.DataFrame([dict(TEAM_ID=tid,TEAM_ABBREVIATION=name,GAMES_PLAYED=completed) for name,tid in TEAM_IDS.items()])
            all_totals=[]
            for game in games:
                sides=[name for name,tid in TEAM_IDS.items() if tid in (game['away_team_id'],game['home_team_id'])]
                minutes=for_matchup(state['bundle'],roster,*sides,'2026-10-29T01:00:00Z')
                base=build_projections(roster[roster.PLAYER_ID.isin(minutes)],history,counts,current,approved_minutes=minutes)
                base['Opponent']=base.Team.map({sides[0]:sides[1],sides[1]:sides[0]})
                # Missing category history must retain real baseline production.
                matched=apply_matchup(base,roster,counts,{}, {})
                totals,breakdown,issues=team_projection_tables(base,matched,matched,sides)
                self.assertFalse(issues)
                self.assertTrue(np.isfinite(totals[['Minutes','PTS','REB','AST','3PM','STL','BLK','TOV','PRA']]).all().all())
                self.assertTrue(totals.Minutes.eq(240.).all())
                self.assertTrue(totals['Stat coverage'].eq('Complete').all())
                self.assertTrue(base['Minutes source'].eq('Approved game minutes').all())
                self.assertEqual(base['Projected minutes'].eq(0).sum(),4)
                self.assertTrue(np.allclose(matched.PRA,matched.PTS+matched.REB+matched.AST,equal_nan=True))
                all_totals.append(totals)
            self.assertEqual(set(pd.concat(all_totals).Team),set(TEAM_IDS))
        pd.testing.assert_frame_equal(roster,original)
