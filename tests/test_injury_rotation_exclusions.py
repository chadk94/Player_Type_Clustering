"""Excluded Out players must not invalidate an otherwise verified game report."""
from copy import deepcopy
from datetime import date
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from approved_minutes import digest, validate_bundle
from import_minutes import TEAM_IDS
from minutes_sync import validate_feed
from sportsbook_injuries import game_injuries, pending_key_players


DAY = '2026-11-18'
TIP = pd.Timestamp(DAY + 'T23:30:00Z')


def fixture(away='CHI', home='BKN'):
    game = dict(Away=away, Home=home, GameTimeUTC=TIP)
    a, h = TEAM_IDS[away], TEAM_IDS[home]
    key = f'{DAY}_{a}_{h}'
    roster, approvals = [], []
    for team, tid, opponent, offset in ((away, a, h, 0), (home, h, a, 100)):
        players = []
        for i in range(1, 9):
            pid = offset + i
            name = f'Player {pid}'
            roster.append(dict(PLAYER_ID=pid, PLAYER_NAME=name, TEAM_ID=tid,
                               TEAM_ABBREVIATION=team, PROJECTED_MIN=30.))
            players.append(dict(player_id=pid, player_name=name, minutes=30.,
                                starter=i <= 5, injury_status='Available'))
        approvals.append(dict(schema='nba_approved_minutes_v1', game_date=DAY,
                              game_key=key, team=team, team_id=tid, opponent_id=opponent,
                              players=players, approved_at=DAY+'T12:00Z',
                              approved_by='test', approval_version=1))
    bundle = dict(schema='nba_approved_slate_v1', approvals=approvals,
                  games=[dict(game_date=DAY, game_key=key, away_team_id=a, home_team_id=h)])
    bundle['bundle_id'] = digest(bundle)
    reports = {(date.fromisoformat(DAY), f'{away}@{home}', team):
               dict(submitted=True, partial=False, players={name: 'Out'})
               for team, name in ((away, 'Orlando Robinson'), (home, 'Grant Nelson'))}
    return game, reports, pd.DataFrame(roster), bundle


def add_excluded(bundle, minutes=0.):
    bundle['approvals'][0]['players'].append(dict(player_id=999,
        player_name='Orlando Robinson', minutes=minutes, starter=False, injury_status='Out'))
    bundle['approvals'][0]['players'][0]['minutes'] -= minutes
    bundle.pop('bundle_id', None)
    bundle['bundle_id'] = digest(bundle)


class InjuryRotationExclusionTests(TestCase):
    def test_both_bulls_nets_exclusions_clear_without_changing_inputs(self):
        game, reports, roster, bundle = fixture()
        before = deepcopy((game, reports, bundle))
        roster_before = roster.copy(deep=True)
        for source in (validate_bundle(bundle, roster, '2026-27'),
                       validate_feed(dict(bundles=[bundle]), roster, '2026-27')):
            info = game_injuries(game, reports, roster, {}, source)
            self.assertTrue(all(t.complete for t in info.values()))
            self.assertEqual(info['CHI'].statuses['Orlando Robinson'], 'Out')
            self.assertEqual(info['BKN'].statuses['Grant Nelson'], 'Out')
            self.assertFalse(pending_key_players(info))
            self.assertFalse(any(t.unverified for t in info.values()))
        self.assertEqual((game, reports, bundle), before)
        pd.testing.assert_frame_equal(roster, roster_before)

    def test_explicit_zero_without_a_prop_profile_is_also_resolved(self):
        game, reports, roster, bundle = fixture()
        add_excluded(bundle)
        validate_bundle(bundle, roster, '2026-27')
        self.assertTrue(game_injuries(game, reports, roster, {}, bundle)['CHI'].complete)
        # Neither a conflicting live allocation nor an active approved row is ignored.
        self.assertFalse(game_injuries(game, reports, roster, {999: 8.}, bundle)['CHI'].complete)
        bundle['approvals'][0]['players'][-1]['minutes'] = 8.
        bundle['approvals'][0]['players'][0]['minutes'] -= 8.
        self.assertFalse(game_injuries(game, reports, roster, {}, bundle)['CHI'].complete)

    def test_exclusion_rule_works_for_every_team(self):
        for team in TEAM_IDS:
            opponent = 'BOS' if team != 'BOS' else 'DET'
            with self.subTest(team=team):
                game, reports, roster, bundle = fixture(team, opponent)
                self.assertTrue(game_injuries(game, reports, roster, {}, bundle)[team].complete)

    def test_only_exact_complete_team_approval_can_resolve_an_unknown_out(self):
        for case in ('none', 'date', 'opponent', 'approval_date', 'approval_team',
                     'approval_opponent', 'duplicate', 'missing_team', 'partial_rotation'):
            game, reports, roster, bundle = fixture()
            if case == 'none': bundle = None
            if case == 'date': bundle['games'][0]['game_date'] = '2026-11-17'
            if case == 'opponent': bundle['games'][0]['home_team_id'] = TEAM_IDS['NYK']
            if case == 'approval_date': bundle['approvals'][0]['game_date'] = '2026-11-17'
            if case == 'approval_team': bundle['approvals'][0]['team'] = 'NYK'
            if case == 'approval_opponent': bundle['approvals'][0]['opponent_id'] = TEAM_IDS['NYK']
            if case == 'duplicate': bundle['approvals'].append(deepcopy(bundle['approvals'][0]))
            if case == 'missing_team': bundle['approvals'].pop(0)
            if case == 'partial_rotation': bundle['approvals'][0]['players'][0]['minutes'] = 20.
            with self.subTest(case=case):
                info = game_injuries(game, reports, roster, {}, bundle)['CHI']
                self.assertFalse(info.complete)
                self.assertIn('Orlando Robinson (CHI)', info.unverified)

    def test_unresolved_statuses_ambiguity_and_incomplete_reports_stay_unchecked(self):
        game, reports, roster, bundle = fixture()
        key = date.fromisoformat(DAY), 'CHI@BKN', 'CHI'
        for status in ('Questionable', 'Doubtful', 'Probable', 'Available'):
            with self.subTest(status=status):
                reports[key]['players']['Orlando Robinson'] = status
                info = game_injuries(game, reports, roster, {}, bundle)['CHI']
                self.assertFalse(info.complete)
                self.assertIn('Orlando Robinson (CHI)', info.unverified)
        reports[key]['players']['Orlando Robinson'] = 'Out'
        for field in ('partial', 'submitted'):
            changed = deepcopy(reports)
            changed[key][field] = field == 'partial'
            self.assertFalse(game_injuries(game, changed, roster, {}, bundle)['CHI'].complete)
        add_excluded(bundle)
        duplicate = dict(bundle['approvals'][0]['players'][-1], player_id=998)
        bundle['approvals'][0]['players'].append(duplicate)
        self.assertFalse(game_injuries(game, reports, roster, {}, bundle)['CHI'].complete)
        roster.loc[:1, 'PLAYER_NAME'] = ['Orlando Robinson', 'OrlandoRobinson']
        self.assertFalse(game_injuries(game, reports, roster, {}, bundle)['CHI'].complete)

    def test_edge_injury_check_clears_but_projections_and_real_q_flags_do_not_change(self):
        from edge_review import assess_edges
        from test_edge_review import edge, finalized, raw_team, reference
        game, reports, roster, bundle = fixture('BOS', 'DET')
        now = TIP - pd.Timedelta(hours=3)
        info = game_injuries(game, reports, roster, {}, bundle)
        e, f = edge(), finalized(raw_team())
        before_e, before_f = e.copy(deep=True), f.copy(deep=True)
        refs = reference().assign(game_date=DAY, updated_at=now.isoformat())
        with patch('edge_review.role_evidence', return_value=dict(ratios={'PTS':1.},
                                                                 trends={'PTS':0.}, detail='Stable')):
            result = assess_edges(e, f, game, pd.DataFrame(), {}, refs, info, now, now)
            self.assertEqual(result.iloc[0]['Injury flag'], 'Clear')
            pd.testing.assert_frame_equal(result[e.columns], before_e)
            stale = assess_edges(e, f, game, pd.DataFrame(), {}, refs, info,
                                 now-pd.Timedelta(minutes=31), now)
            self.assertEqual(stale.iloc[0]['Injury flag'], 'Unchecked')
            report = reports[date.fromisoformat(DAY), 'BOS@DET', 'BOS']['players']
            for status in ('Questionable', 'Out'):
                report['Player 1'] = status
                info = game_injuries(game, reports, roster, {}, bundle)
                result = assess_edges(e, f, game, pd.DataFrame(), {}, refs, info, now, now)
                self.assertEqual(result.iloc[0]['Injury flag'], 'Q')
        pd.testing.assert_frame_equal(e, before_e)
        pd.testing.assert_frame_equal(f, before_f)
