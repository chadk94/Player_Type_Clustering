from copy import deepcopy
from datetime import date
import unittest
from unittest.mock import Mock

import requests
import pandas as pd

from schedule_feed import load_daily_schedule, validate_daily_schedule, ScheduleUnavailable, ScheduleMemory
from team_projection_view import upcoming_team_games
from sportsbook_edges import scheduled_book_games


class DailyScheduleTests(unittest.TestCase):
    def setUp(self):
        self.day = date(2026, 10, 27)
        self.payload = {'events': [dict(id='game1', date='2026-10-28T03:00Z',
            season={'year': 2027, 'type': 2}, status={'type': {'completed': False, 'description': 'Scheduled'}},
            competitions=[{'competitors': [
                {'team': {'abbreviation': 'LAL'}, 'homeAway': 'home'},
                {'team': {'abbreviation': 'POR'}, 'homeAway': 'away'}]}])]}

    def test_timeout_retries_once_then_returns_both_teams(self):
        fetch = Mock(side_effect=[requests.Timeout('temporary'), self.payload])
        result = load_daily_schedule('2026-27', self.day, fetch)
        self.assertEqual(result.Opponent.tolist(), ['POR', 'LAL'])
        self.assertEqual(result.Venue.tolist(), ['home', 'away'])
        self.assertEqual(fetch.call_count, 2)
        self.assertTrue(all(call.args[0].endswith('dates=20261027') for call in fetch.call_args_list))

    def test_repeated_failure_raises_instead_of_returning_an_empty_slate(self):
        fetch = Mock(side_effect=requests.Timeout('temporary'))
        with self.assertRaisesRegex(ScheduleUnavailable, 'two attempts'):
            load_daily_schedule('2026-27', self.day, fetch)
        self.assertEqual(fetch.call_count, 2)

    def test_verified_empty_and_completed_slates_do_not_retry(self):
        fetch = Mock(return_value={'events': []})
        self.assertTrue(load_daily_schedule('2026-27', self.day, fetch).empty)
        fetch.assert_called_once()
        self.payload['events'][0]['status']['type']['completed'] = True
        self.assertTrue(validate_daily_schedule(self.payload, 2027, self.day).empty)

    def test_malformed_wrong_date_and_wrong_season_are_unavailable(self):
        variants = [{}, {'events': None}, {'events': [None]}]
        for field, value in [('date', '2026-10-29T03:00Z'), ('date', '2026-10-27T19:00'),
                             ('season', {'year': 2026, 'type': 2}), ('season', None),
                             ('competitions', []), ('competitions', [{'competitors': []}])]:
            bad = deepcopy(self.payload)
            bad['events'][0][field] = value
            variants.append(bad)
        duplicate = deepcopy(self.payload)
        duplicate['events'] *= 2
        variants.append(duplicate)
        for bad in variants:
            with self.subTest(payload=bad):
                fetch = Mock(return_value=bad)
                with self.assertRaises(ScheduleUnavailable):
                    load_daily_schedule('2026-27', self.day, fetch)
                self.assertEqual(fetch.call_count, 2)

    def test_transient_malformed_json_can_recover(self):
        fetch = Mock(side_effect=[ValueError('invalid JSON'), self.payload])
        self.assertEqual(len(load_daily_schedule('2026-27', self.day, fetch)), 2)
        self.assertEqual(fetch.call_count, 2)

    def verified_frame(self, checked='2026-10-27T17:00:00Z'):
        frame = load_daily_schedule('2026-27', self.day, Mock(return_value=self.payload))
        self.assertEqual(frame.attrs['schedule_date'], self.day.isoformat())
        self.assertEqual(frame.attrs['schedule_season'], '2026-27')
        self.assertIsNotNone(pd.Timestamp(frame.attrs['schedule_checked_at']).tzinfo)
        frame.attrs['schedule_checked_at'] = checked
        return frame

    def test_backup_is_date_and_season_scoped_and_does_not_reage(self):
        memory = ScheduleMemory()
        frame = self.verified_frame()
        memory.remember('2026-27', self.day, frame)
        backup = memory.recover('2026-27', self.day, '2026-10-27T17:29:00Z')
        pd.testing.assert_frame_equal(frame, backup)
        memory.remember('2026-27', self.day, backup)  # Re-reading a cache is not verification.
        self.assertIsNone(memory.recover('2026-27', self.day, '2026-10-27T17:30:01Z'))
        self.assertIsNone(memory.recover('2026-27', date(2026, 10, 28), '2026-10-27T17:10:00Z'))
        self.assertIsNone(memory.recover('2025-26', self.day, '2026-10-27T17:10:00Z'))
        self.assertIsNone(memory.recover('2026-27', self.day, '2026-10-27T16:59:00Z'))

    def test_backup_rejects_unverified_frames_and_returns_isolated_copies(self):
        memory = ScheduleMemory()
        frame = self.verified_frame()
        memory.remember('2025-26', self.day, frame)
        self.assertIsNone(memory.recover('2025-26', self.day))
        bare = frame.copy()
        bare.attrs.clear()
        memory.remember('2026-27', self.day, bare)
        self.assertIsNone(memory.recover('2026-27', self.day))
        memory.remember('2026-27', self.day, frame)
        frame.loc[:, 'Opponent'] = 'BAD'
        recovered = memory.recover('2026-27', self.day, '2026-10-27T17:10:00Z')
        self.assertEqual(recovered.Opponent.tolist(), ['POR', 'LAL'])
        recovered.loc[:, 'Opponent'] = 'BAD'
        self.assertEqual(memory.recover('2026-27', self.day, '2026-10-27T17:10:00Z').Opponent.tolist(), ['POR', 'LAL'])

    def test_new_verified_empty_slate_replaces_backup_and_older_responses_cannot_restore_it(self):
        memory = ScheduleMemory()
        frame = self.verified_frame()
        memory.remember('2026-27', self.day, frame)
        empty = frame.iloc[:0].copy()
        empty.attrs['schedule_checked_at'] = '2026-10-27T17:05:00Z'
        memory.remember('2026-27', self.day, empty)
        memory.remember('2026-27', self.day, frame)
        self.assertTrue(memory.recover('2026-27', self.day, '2026-10-27T17:10:00Z').empty)

    def test_fallback_keeps_tipoff_cutoff_and_postponement_checks(self):
        memory = ScheduleMemory()
        frame = self.verified_frame('2026-10-28T03:40:00Z')
        memory.remember('2026-27', self.day, frame)
        backup = memory.recover('2026-27', self.day, '2026-10-28T03:59:00Z')
        self.assertEqual(len(upcoming_team_games(backup, now=pd.Timestamp('2026-10-28T03:59:00Z'))), 1)
        self.assertEqual(upcoming_team_games(backup, now=pd.Timestamp('2026-10-28T04:00:00Z')), [])
        frame.loc[:, 'GameStatus'] = 'Postponed'
        frame.attrs['schedule_checked_at'] = '2026-10-28T03:50:00Z'
        memory.remember('2026-27', self.day, frame)
        backup = memory.recover('2026-27', self.day, '2026-10-28T03:59:00Z')
        self.assertEqual(upcoming_team_games(backup, now=pd.Timestamp('2026-10-28T03:59:00Z')), [])

    def test_fallback_cannot_match_a_sportsbook_rematch_or_started_game(self):
        frame = self.verified_frame('2026-10-28T02:45:00Z')
        memory = ScheduleMemory()
        memory.remember('2026-27', self.day, frame)
        backup = memory.recover('2026-27', self.day, '2026-10-28T03:01:00Z')
        event = dict(id='book', home_team='Los Angeles Lakers', away_team='Portland Trail Blazers',
                     commence_time='2026-10-28T03:00:00Z')
        self.assertEqual(len(scheduled_book_games([event], backup, now=pd.Timestamp('2026-10-28T02:59:00Z'))), 1)
        self.assertEqual(scheduled_book_games([event], backup, now=pd.Timestamp('2026-10-28T03:01:00Z')), [])
        event['commence_time'] = '2026-10-28T03:10:00Z'
        self.assertEqual(scheduled_book_games([event], backup, now=pd.Timestamp('2026-10-28T03:01:00Z')), [])
        event['commence_time'] = '2026-11-01T03:00:00Z'
        self.assertEqual(scheduled_book_games([event], backup, now=pd.Timestamp('2026-10-28T02:59:00Z')), [])


if __name__ == '__main__':
    unittest.main()
