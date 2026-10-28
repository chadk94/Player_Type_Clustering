from copy import deepcopy
import json
import unittest
import pandas as pd

from test_approved_minutes import example, reseal
from approved_minutes import for_matchup
from approved_minutes_storage import new_state, update_upload, accept_reply
from live_projections import build_projections, STAT_NAMES
from import_minutes import TEAM_IDS


def reply(state, **fields):
    return dict(request=state['request']['request'], status='ok', **fields)


class BrowserMinutesTests(unittest.TestCase):
    def setUp(self):
        self.bundle, self.roster = example()
        self.season = '2026-27'

    def test_fresh_session_restores_18_instead_of_workbook_8(self):
        players = self.bundle['approvals'][0]['players']
        for p, minutes in zip(players, [18., 48., 48., 48., 38., 40., 0.]):
            p['minutes'] = minutes
        reseal(self.bundle)
        self.roster.loc[self.roster.PLAYER_ID.eq(1), 'PROJECTED_MIN'] = 8.
        before = self.roster.copy(deep=True)
        session = new_state(self.season)
        update_upload(session, json.dumps(self.bundle).encode(), self.roster, self.season)
        saved = deepcopy(session['request']['bundle'])
        accept_reply(session, reply(session), self.roster, self.season)
        # Refresh discards all server session/widget state; only browser data survives.
        refreshed = new_state(self.season)
        self.assertTrue(refreshed['pending'])
        accept_reply(refreshed, reply(refreshed, bundle=saved), self.roster, self.season)
        minutes = for_matchup(refreshed['bundle'], self.roster, 'BOS', 'DET', '2026-10-21T23:00:00Z')
        history = pd.DataFrame([dict(PLAYER_ID=p, MIN=20., **{s:10. for s in STAT_NAMES}) for p in self.roster.PLAYER_ID])
        teams = pd.DataFrame([dict(TEAM_ID=TEAM_IDS[t], GAMES_PLAYED=0) for t in ('BOS', 'DET')])
        projection = build_projections(self.roster, history, teams, pd.DataFrame(), approved_minutes=minutes)
        self.assertEqual(projection.iloc[0]['Projected minutes'], 18.)
        self.assertTrue(projection['Minutes source'].eq('Approved game minutes').all())
        self.assertEqual(projection.groupby('Team')['Projected minutes'].sum().tolist(), [240., 240.])
        self.assertFalse(for_matchup(refreshed['bundle'], self.roster, 'BOS', 'DET', '2026-10-22T23:00:00Z'))
        pd.testing.assert_frame_equal(self.roster, before)

    def test_new_upload_wins_over_late_restore_response(self):
        state = new_state(self.season)
        late = reply(state, bundle=None)
        update_upload(state, json.dumps(self.bundle).encode(), self.roster, self.season)
        accept_reply(state, late, self.roster, self.season)
        self.assertTrue(state['pending'])
        self.assertEqual(state['bundle'], self.bundle)
        accept_reply(state, reply(state), self.roster, self.season)
        self.assertFalse(state['pending'])

    def test_invalid_restored_or_replacement_bundle_blocks_old_projection(self):
        state = new_state(self.season)
        tampered = deepcopy(self.bundle)
        tampered['approvals'][0]['players'][0]['minutes'] = 8.
        accept_reply(state, reply(state, bundle=tampered), self.roster, self.season)
        self.assertIsNone(state['bundle'])
        self.assertIn('checksum', state['error'])
        update_upload(state, json.dumps(self.bundle).encode(), self.roster, self.season)
        accept_reply(state, reply(state), self.roster, self.season)
        update_upload(state, b'{broken', self.roster, self.season)
        self.assertIsNone(state['bundle'])
        self.assertIsNotNone(state['error'])
        self.assertEqual(state['request']['action'], 'clear')

    def test_remove_and_explicit_clear_erase_browser_copy(self):
        for explicit in (False, True):
            state = new_state(self.season)
            update_upload(state, json.dumps(self.bundle).encode(), self.roster, self.season)
            accept_reply(state, reply(state), self.roster, self.season)
            update_upload(state, None, self.roster, self.season, clear=explicit)
            self.assertEqual(state['request']['action'], 'clear')
            self.assertIsNone(state['bundle'])
            accept_reply(state, reply(state), self.roster, self.season)
            self.assertFalse(state['pending'])

    def test_blocked_storage_keeps_upload_but_reports_not_saved(self):
        state = new_state(self.season)
        update_upload(state, json.dumps(self.bundle).encode(), self.roster, self.season)
        failed = reply(state); failed['status'] = 'unavailable'
        accept_reply(state, failed, self.roster, self.season)
        self.assertEqual(state['bundle'], self.bundle)
        self.assertEqual(state['storage_error'], 'unavailable')
        self.assertFalse(state['pending'])

    def test_empty_restore_does_not_invent_minutes(self):
        state = new_state(self.season)
        accept_reply(state, reply(state, bundle=None), self.roster, self.season)
        self.assertIsNone(state['bundle'])
        self.assertIsNone(state['error'])
        self.assertFalse(state['pending'])

    def test_another_tab_replaces_clears_and_validates_without_resaving_old_upload(self):
        state = new_state(self.season)
        old_content = json.dumps(self.bundle).encode()
        update_upload(state, old_content, self.roster, self.season)
        accept_reply(state, reply(state), self.roster, self.season)
        changed = deepcopy(self.bundle)
        changed['approvals'][0]['players'][0]['minutes'] = 30.
        changed['approvals'][0]['players'][1]['minutes'] = 48.
        changed['approvals'][0]['players'][5]['minutes'] = 42.
        reseal(changed)
        event = reply(state, source='storage', revision=1, bundle=json.dumps(changed))
        accept_reply(state,event,self.roster,self.season)
        self.assertEqual(state['bundle'],changed)
        update_upload(state,old_content,self.roster,self.season)
        self.assertEqual(state['bundle'],changed)
        self.assertFalse(state['pending'])
        corrupt = deepcopy(changed)
        corrupt['approvals'][0]['players'][0]['minutes'] = 1.
        accept_reply(state,reply(state,source='storage',revision=2,bundle=corrupt),self.roster,self.season)
        self.assertIsNone(state['bundle'])
        self.assertIn('checksum',state['error'])
        accept_reply(state,reply(state,source='storage',revision=3,bundle=changed),self.roster,self.season)
        self.assertEqual(state['bundle'],changed)
        self.assertIsNone(state['error'])
        accept_reply(state,reply(state,source='storage',revision=4,bundle=None),self.roster,self.season)
        self.assertIsNone(state['bundle'])
        self.assertFalse(state['pending'])
        accept_reply(state,reply(state,source='storage',revision=4,bundle=changed),self.roster,self.season)
        self.assertEqual(state['bundle'],changed)

    def test_pending_upload_ignores_external_event_and_external_invalid_blocks_old_minutes(self):
        state = new_state(self.season)
        update_upload(state,json.dumps(self.bundle).encode(),self.roster,self.season)
        accept_reply(state,reply(state,source='storage',revision=1,bundle=None),self.roster,self.season)
        self.assertEqual(state['bundle'],self.bundle)
        self.assertTrue(state['pending'])
        accept_reply(state,reply(state),self.roster,self.season)
        invalid=reply(state,source='storage',revision=2); invalid['status']='invalid'
        accept_reply(state,invalid,self.roster,self.season)
        self.assertIsNone(state['bundle'])
        self.assertIsNotNone(state['error'])
