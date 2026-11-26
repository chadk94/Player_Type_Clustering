"""Approved roster changes and missing profiles must not take down the slate."""
from copy import deepcopy
from datetime import date
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch, Mock

import pandas as pd
from streamlit.testing.v1 import AppTest

from approved_minutes import digest, for_matchup, blocked_matchup_message, apply_bundle_change
from import_minutes import TEAM_IDS
from minutes_sync import validate_feed
from background_feed import BackgroundFeed
from test_background_feed import ImmediateExecutor
from roster_membership import roster_for_approved_games
from season_pipeline import ROOT, apply_roster_rules
from live_projections import build_projections, STAT_NAMES


DAY = '2026-11-18'
TIP = DAY + 'T23:00:00Z'


def slate(roster, pair=('UTA', 'LAL'), mover=1641723):
    a, b = (TEAM_IDS[t] for t in pair)
    key = f'{DAY}_{a}_{b}'
    records = []
    names = roster.set_index('PLAYER_ID').PLAYER_NAME
    for team, tid, opp in ((pair[0], a, b), (pair[1], b, a)):
        ids = roster.loc[roster.TEAM_ID.eq(tid) & roster.PLAYER_ID.ne(mover), 'PLAYER_ID'].head(6).tolist()
        if tid == a and mover is not None:
            ids[-1] = mover
        players = [dict(player_id=int(pid), player_name=names[pid], minutes=40., starter=i<5,
                        injury_status='Available') for i, pid in enumerate(ids)]
        records.append(dict(schema='nba_approved_minutes_v1', game_date=DAY, game_key=key,
                            team=team, team_id=tid, opponent_id=opp, players=players,
                            approved_at=DAY+'T12:00:00Z', approved_by='test', approval_version=1))
    raw = dict(schema='nba_approved_slate_v1', approvals=records,
               games=[dict(game_date=DAY, game_key=key, away_team_id=a, home_team_id=b)])
    raw['bundle_id'] = digest(raw)
    return raw


def seal(raw):
    raw.pop('bundle_id', None)
    raw['bundle_id'] = digest(raw)
    return raw


class LinkedMinutesResilienceTests(TestCase):
    def setUp(self):
        self.roster = apply_roster_rules(pd.read_csv(ROOT / 'preseason_archetypes.csv'))

    def test_bufkin_uses_approved_utah_membership_and_preserves_every_profile_field(self):
        original = self.roster.copy(deep=True)
        raw = slate(self.roster)
        linked = validate_feed(dict(bundles=[raw]), self.roster, '2026-27')
        self.assertFalse(linked['blocked_games'])
        game_roster = roster_for_approved_games(self.roster, '2026-27', date.fromisoformat(DAY), linked)
        before = self.roster.set_index('PLAYER_ID').loc[1641723]
        after = game_roster.set_index('PLAYER_ID').loc[1641723]
        self.assertEqual(after.TEAM_ABBREVIATION, 'UTA')
        pd.testing.assert_series_equal(before.drop(['TEAM_ID','TEAM_ABBREVIATION']),
                                       after.drop(['TEAM_ID','TEAM_ABBREVIATION']))
        minutes = for_matchup(linked, self.roster, 'UTA', 'LAL', TIP)
        self.assertEqual(minutes[1641723], 40.)
        self.assertEqual(sum(minutes.values()), 480.)
        history = pd.DataFrame([dict(PLAYER_ID=int(p), MIN=20., **{s:3. for s in STAT_NAMES})
                                for p in self.roster.PLAYER_ID])
        teams = self.roster[['TEAM_ID']].drop_duplicates().assign(GAMES_PLAYED=10)
        result = build_projections(game_roster[game_roster.PLAYER_ID.isin(minutes)], history, teams,
                                   pd.DataFrame(), approved_minutes=minutes)
        self.assertEqual(result.groupby('Team')['Projected minutes'].sum().to_dict(), {'UTA':240., 'LAL':240.})
        self.assertEqual(result.loc[result.PLAYER_ID.eq(1641723), 'Minutes source'].iloc[0], 'Approved game minutes')
        next_day = roster_for_approved_games(self.roster, '2026-27', date(2026,11,19), linked)
        self.assertEqual(next_day.loc[next_day.PLAYER_ID.eq(1641723), 'TEAM_ABBREVIATION'].iloc[0], 'NOP')
        self.assertEqual(for_matchup(linked, self.roster, 'UTA', 'LAL', '2026-11-19T23:00:00Z'), {})
        pd.testing.assert_frame_equal(self.roster, original)

    def test_known_profile_can_follow_approved_rotation_for_all_30_teams(self):
        for team in TEAM_IDS:
            opponent = 'LAL' if team != 'LAL' else 'BOS'
            with self.subTest(team=team):
                raw = slate(self.roster, (team,opponent))
                linked = validate_feed(dict(bundles=[raw]), self.roster, '2026-27')
                self.assertFalse(linked['blocked_games'])
                self.assertEqual(for_matchup(linked,self.roster,team,opponent,TIP)[1641723],40.)

    def mixed_slate(self):
        bad = slate(self.roster)
        bad['approvals'][0]['players'][-1].update(player_id=99999999, player_name='Unknown player')
        good = slate(self.roster, ('BOS','DET'), None)
        bad['games'].extend(good['games'])
        bad['approvals'].extend(good['approvals'])
        return seal(bad)

    def test_unknown_player_pauses_only_its_game_and_never_uses_fallback_minutes(self):
        raw = self.mixed_slate()
        linked = validate_feed(dict(bundles=[raw]), self.roster, '2026-27')
        self.assertEqual(len(linked['games']),1)
        self.assertEqual(len(linked['approvals']),2)
        self.assertEqual(len(linked['blocked_games']),1)
        self.assertIn('Unknown player',blocked_matchup_message(linked,'UTA','LAL',TIP))
        self.assertFalse(blocked_matchup_message(linked,'BOS','DET',TIP))
        self.assertFalse(blocked_matchup_message(linked,'UTA','LAL','2026-11-19T23:00:00Z'))
        self.assertEqual(for_matchup(linked,self.roster,'UTA','LAL',TIP),{})
        self.assertEqual(sum(for_matchup(linked,self.roster,'BOS','DET',TIP).values()),480.)
        # Same-game snapshots are invalidated when a previously valid approval is rejected.
        state = dict(approved_minutes_bundle='old', team_projection_snapshots={'old':1}, custom_minutes_1=30)
        apply_bundle_change(state,linked)
        self.assertNotIn('team_projection_snapshots',state)
        from projection_dashboard import team_game_projection
        ui, frames = Mock(), Mock()
        team_game_projection(ui,dict(Away='UTA',Home='LAL',GameTimeUTC=TIP),None,None,None,{},None,frames,linked)
        frames.assert_not_called()
        ui.warning.assert_called_once()

    def test_identity_checksum_outs_and_duplicate_players_stay_protected(self):
        raw = slate(self.roster)
        raw['approvals'][0]['players'][-1]['player_name'] = 'Different person'
        linked = validate_feed(dict(bundles=[seal(raw)]), self.roster, '2026-27')
        self.assertFalse(linked['games'])
        self.assertIn('does not match the name',linked['blocked_games'][0]['reason'])
        for kind in ('checksum','out','duplicate','total'):
            raw = slate(self.roster)
            players = raw['approvals'][0]['players']
            if kind == 'checksum': raw['bundle_id'] = 'bad'
            if kind == 'out': players[-1]['injury_status'] = 'Out'
            if kind == 'duplicate': players[-1]['player_id'] = players[0]['player_id']
            if kind == 'total': players[-1]['minutes'] = 39.
            if kind != 'checksum': seal(raw)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_feed(dict(bundles=[raw]),self.roster,'2026-27')

    def test_partial_sync_restores_page_and_automatically_recovers(self):
        raw = self.mixed_slate()
        def fake_component(**kw):
            return SimpleNamespace(connection={'ready':True,'connection':'test-connection'})
        code = '''
import streamlit as st
import pandas as pd
from season_pipeline import ROOT, apply_roster_rules
from approved_minutes import render_import
bundle = render_import(st,apply_roster_rules(pd.read_csv(ROOT/'preseason_archetypes.csv')),'2026-27')
st.success(f"Projection page ready: {len(bundle['games'])} games")
'''
        with patch('minutes_sync.minutes_feed_worker', return_value=BackgroundFeed(ImmediateExecutor())), \
             patch('minutes_sync.connection_component',return_value=fake_component), \
             patch('minutes_sync.read_feed',return_value=dict(bundles=[raw])) as read:
            app = AppTest.from_string(code).run(timeout=10)
            self.assertFalse(app.exception)
            self.assertEqual(app.success[0].value, 'Projection page ready: 1 games')
            self.assertTrue(any('Only those games are paused' in w.value for w in app.warning))
            fixed = deepcopy(raw)
            fixed['approvals'][0]['players'][-1].update(player_id=1641723,player_name='Kobe Bufkin')
            read.return_value = dict(bundles=[seal(fixed)])
            app.session_state['nba_minutes_link']['checked'] = 0
            app.run(timeout=10)
            self.assertFalse(app.exception)
            self.assertEqual(app.success[0].value, 'Projection page ready: 2 games')
            self.assertFalse(app.warning)
