from datetime import date
from hashlib import sha256
import json
from unittest import TestCase

import pandas as pd

from approved_minutes import digest, for_matchup, validate_bundle
from approved_minutes_storage import new_state, update_upload
from import_minutes import TEAM_IDS
from roster_membership import roster_on_date
from season_pipeline import ROOT, apply_roster_rules


class CliffordRosterTests(TestCase):
    def setUp(self):
        self.player_id = 1642363
        self.roster = apply_roster_rules(pd.read_csv(ROOT / 'preseason_archetypes.csv'))
        self.live = roster_on_date(self.roster, '2026-27', date(2026, 11, 13))

    def slate(self, day, team):
        away, home = TEAM_IDS[team], TEAM_IDS['LAC']
        key = f'{day}_{away}_{home}'
        teams = []
        for code, tid, opp in [(team, away, home), ('LAC', home, away)]:
            ids = self.roster.loc[self.roster.TEAM_ID.eq(tid) & self.roster.PLAYER_ID.ne(self.player_id), 'PLAYER_ID'].iloc[:6].tolist()
            if tid == away:
                ids[-1] = self.player_id
            minutes = [45,45,44,44,44,18]
            players = [dict(player_id=pid, player_name='Nique Clifford' if pid == self.player_id else str(pid),
                            minutes=m, starter=i<5, injury_status='Available')
                       for i,(pid,m) in enumerate(zip(ids, minutes))]
            teams.append(dict(game_key=key, team=code, team_id=tid, opponent_id=opp, game_date=day,
                              schema='nba_approved_minutes_v1', approval_version=1,
                              approved_at=day, approved_by='test', players=players))
        result = dict(schema='nba_approved_slate_v1', approvals=teams, games=[dict(game_key=key, game_date=day, away_team_id=away, home_team_id=home)])
        result['bundle_id'] = digest(result)
        return result

    def test_only_membership_changes_and_historical_rotations_still_resolve(self):
        original = self.roster.set_index('PLAYER_ID')
        current = self.live.set_index('PLAYER_ID', verify_integrity=True)
        expected = original.copy()
        expected.loc[self.player_id, ['TEAM_ID','TEAM_ABBREVIATION']] = [TEAM_IDS['OKC'], 'OKC']
        pd.testing.assert_frame_equal(current, expected)
        self.assertEqual(current.loc[self.player_id, ['OffArchetype','DefArchetype']].tolist(),
                         ['Spot-Up Role Player','Long Wing/Secondary POA'])
        for day, right, wrong in [('2026-11-07','SAC','OKC'), ('2026-11-08','OKC','SAC'), ('2026-11-13','OKC','SAC')]:
            with self.subTest(day):
                bundle = validate_bundle(self.slate(day, right), self.live, '2026-27')
                minutes = for_matchup(bundle, self.live, right, 'LAC', day+'T23:00:00Z')
                self.assertEqual(minutes[self.player_id], 18)
                self.assertEqual(sum(minutes.values()), 480)
                with self.assertRaisesRegex(ValueError, 'matching prop-model roster'):
                    validate_bundle(self.slate(day, wrong), self.live, '2026-27')

    def test_already_rejected_upload_is_retried_after_roster_update(self):
        content = json.dumps(self.slate('2026-11-13','OKC')).encode()
        state = new_state('2026-27')
        state.update(seen_upload=sha256(content).hexdigest(), error='missing profile')
        update_upload(state, content, self.live, '2026-27')
        self.assertEqual(state['request']['action'], 'save')
        self.assertIsNone(state['error'])
