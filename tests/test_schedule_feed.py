from copy import deepcopy
from datetime import date
import unittest
from unittest.mock import Mock

import requests

from schedule_feed import load_daily_schedule, validate_daily_schedule, ScheduleUnavailable


class DailyScheduleTests(unittest.TestCase):
    def setUp(self):
        self.day = date(2026, 10, 27)
        self.payload = {'events': [dict(id='game1', date='2026-10-28T03:00Z',
            season={'year': 2027, 'type': 2}, status={'type': {'completed': False}},
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


if __name__ == '__main__':
    unittest.main()
