from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

from saved_sportsbook_lines import (read_saved_feed, cached_feed, source_notice, uses_saved_lines,
                                    render_source, SOURCE, SAVED)
from sportsbook_lines import OddsFeedError, player_quotes
from sportsbook_edges import game_edges
from market_points import market_targets
from points_tracking import snapshot


class SavedLinesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.now = datetime(2026, 11, 21, 19, 10, tzinfo=timezone.utc)
        self.event = dict(id='nyk-phi', home_team='New York Knicks',
                          away_team='Philadelphia 76ers',
                          commence_time=(self.now + timedelta(hours=4)).isoformat())
        self.market = dict(key='player_points', last_update=self.now.isoformat(), outcomes=[
            dict(description='Test Player', name=side, point=20.5, price=-110)
            for side in ('Over', 'Under')])
        self.payload = dict(self.event, bookmakers=[dict(key='draftkings', markets=[self.market])])
        self.path = '/events/nyk-phi/odds'
        self.params = dict(bookmakers='draftkings', markets='player_points', oddsFormat='american')
        self.fetch = Mock(return_value=self.payload)

    def read(self, **kwargs):
        return read_saved_feed(kwargs.pop('path', self.path), kwargs.pop('api_key', 'secret-test-key'),
            self.fetch, cache_dir=self.temp.name, now=kwargs.pop('now', self.now),
            **(self.params | kwargs))

    def fail(self):
        self.fetch.side_effect = OddsFeedError('Sportsbook feed unavailable (HTTP 401).')

    def test_hourly_reuse_survives_repeated_reads_until_exact_deadline(self):
        first = self.read(refresh_after=3600)
        for seconds in (0, 1, 300, 900, 1800, 3599):
            later = self.now + timedelta(seconds=seconds)
            saved = self.read(now=later, refresh_after=3600)
            self.assertEqual(saved[SOURCE]['saved_at'], first[SOURCE]['saved_at'])
            self.assertTrue(uses_saved_lines(saved, later))
        self.fetch.assert_called_once()
        refreshed = self.read(now=self.now + timedelta(hours=1), refresh_after=3600)
        self.assertEqual(self.fetch.call_count, 2)
        self.assertEqual(refreshed[SOURCE]['saved_at'], (self.now + timedelta(hours=1)).isoformat())
        self.assertFalse(refreshed[SOURCE]['reused'])
        self.read(now=self.now + timedelta(minutes=119), refresh_after=3600)
        self.assertEqual(self.fetch.call_count, 2)

    def test_explicit_refresh_resets_only_this_request_hour(self):
        self.read(refresh_after=3600)
        self.read(markets='player_points,player_assists', refresh_after=3600)
        later = self.now + timedelta(minutes=20)
        forced = self.read(now=later, refresh_after=3600, force=True)
        self.assertEqual(self.fetch.call_count, 3)
        self.assertEqual(forced[SOURCE]['saved_at'], later.isoformat())
        other = self.read(now=later, markets='player_points,player_assists', refresh_after=3600)
        self.assertEqual(other[SOURCE]['saved_at'], self.now.isoformat())
        self.read(now=self.now + timedelta(minutes=79), refresh_after=3600)
        self.assertEqual(self.fetch.call_count, 3)
        self.read(now=self.now + timedelta(minutes=80), refresh_after=3600)
        self.assertEqual(self.fetch.call_count, 4)

    def test_failed_hourly_attempt_backs_off_without_rejuvenating_quote(self):
        self.read(refresh_after=3600)
        self.fail()
        for minute in (60, 61, 90, 119):
            saved = self.read(now=self.now + timedelta(minutes=minute), refresh_after=3600)
            self.assertTrue(saved[SOURCE]['fallback'])
            self.assertEqual(saved[SOURCE]['saved_at'], self.now.isoformat())
        self.assertEqual(self.fetch.call_count, 2)
        self.read(now=self.now + timedelta(hours=2), refresh_after=3600)
        self.assertEqual(self.fetch.call_count, 3)
        self.read(now=self.now + timedelta(minutes=121), refresh_after=3600, force=True)
        self.assertEqual(self.fetch.call_count, 4)

    def test_successful_empty_markets_are_reused_for_an_hour(self):
        self.fetch.return_value = dict(self.event, bookmakers=[])
        for minute in (0, 1, 10, 59):
            self.assertEqual(self.read(now=self.now + timedelta(minutes=minute),
                                      refresh_after=3600)['bookmakers'], [])
        self.fetch.assert_called_once()
        self.read(now=self.now + timedelta(hours=1), refresh_after=3600)
        self.assertEqual(self.fetch.call_count, 2)

    def test_parallel_sessions_share_one_provider_request(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        barrier = Barrier(8)
        def visit(_):
            barrier.wait(timeout=5)
            return self.read(refresh_after=3600)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(visit, range(8)))
        self.fetch.assert_called_once()
        self.assertEqual(sum(not r[SOURCE]['reused'] for r in results), 1)
        self.assertTrue(all(r[SOURCE]['saved_at'] == self.now.isoformat() for r in results))

    def test_read_only_lookup_expires_without_fetching(self):
        self.read(refresh_after=3600)
        def cached(minute):
            return cached_feed(self.path, 'secret-test-key', cache_dir=self.temp.name,
                               now=self.now + timedelta(minutes=minute),
                               max_age_seconds=3600, **self.params)
        self.assertEqual(cached(59)['id'], self.event['id'])
        self.assertIsNone(cached(60))
        self.fetch.assert_called_once()

    def test_manual_game_total_wins_until_hourly_refresh_resumes(self):
        import os
        import projection_dashboard as dashboard
        def game(spread):
            return dict(self.event, bookmakers=[dict(key='draftkings', markets=[
                dict(key='spreads', last_update=self.now.isoformat(), outcomes=[
                    dict(name=self.event['home_team'], point=-spread, price=-110),
                    dict(name=self.event['away_team'], point=spread, price=-110)]),
                dict(key='totals', last_update=self.now.isoformat(), outcomes=[
                    dict(name=side, point=220, price=-110) for side in ('Over', 'Under')])])])
        with patch.dict(os.environ, {'SPORTSBOOK_LINES_CACHE_DIR': self.temp.name}), \
             patch('saved_sportsbook_lines.datetime', wraps=datetime) as clock, \
             patch.object(dashboard, 'read_feed') as paid:
            clock.now.return_value = self.now
            paid.return_value = [game(2)]
            dashboard.sportsbook_slate_lines('draftkings', 'secret-test-key')
            clock.now.return_value = self.now + timedelta(minutes=10)
            paid.return_value = game(8)
            dashboard.sportsbook_game_lines('nyk-phi', 'draftkings', 'secret-test-key', force=True)
            self.assertEqual(paid.call_count, 2)
            clock.now.return_value = self.now + timedelta(minutes=65)
            result = dashboard.load_team_game_lines(self.event, 'draftkings', 'secret-test-key')
            self.assertEqual(result['bookmakers'][0]['markets'][0]['outcomes'][0]['point'], -8)
            self.assertEqual(paid.call_count, 2)
            clock.now.return_value = self.now + timedelta(minutes=70)
            paid.return_value = [game(4)]
            result = dashboard.load_team_game_lines(self.event, 'draftkings', 'secret-test-key')
            self.assertEqual(result['bookmakers'][0]['markets'][0]['outcomes'][0]['point'], -4)
            self.assertEqual(paid.call_count, 3)

    def test_round_trip_across_new_reader_preserves_original_times_and_prices(self):
        first = self.read()
        raw = next(Path(self.temp.name).glob('*.json'))
        self.assertEqual(raw.stat().st_mode & 0o777, 0o600)
        self.assertNotIn('secret-test-key', raw.read_text())
        self.assertNotIn('secret-test-key', raw.name)
        first['bookmakers'][0]['markets'][0]['outcomes'][0]['price'] = 999
        self.fail()
        later = self.now + timedelta(hours=2, minutes=5)
        saved = self.read(now=later)
        self.assertEqual(saved[SOURCE]['saved_at'], self.now.isoformat())
        self.assertTrue(uses_saved_lines(saved, later))
        self.assertEqual(saved['bookmakers'], self.payload['bookmakers'])
        quote = player_quotes(saved, 'Test Player', 'draftkings', now=later).set_index('Stat').loc['PTS']
        self.assertEqual(quote['Market status'], SAVED)
        self.assertEqual(quote['Updated UTC'], self.now.isoformat())
        self.assertEqual(quote['Saved UTC'], self.now.isoformat())
        self.assertGreater(quote['Book implied total'], 0)
        notice = source_notice(saved, later)
        self.assertIn('Nov 21, 2026, 12:10 PM MST', notice)
        self.assertIn('2 hr 5 min ago', notice)
        ui = Mock()
        render_source(ui, saved, 'DraftKings', later)
        ui.warning.assert_called_once()
        self.assertIn('not live', ui.warning.call_args.args[0])

    def test_outage_marks_even_recent_prices_saved_then_recovers_to_live(self):
        self.read()
        self.fail()
        saved = self.read(now=self.now + timedelta(minutes=2))
        q = player_quotes(saved, 'Test Player', 'draftkings', now=self.now + timedelta(minutes=2))
        self.assertEqual(q.set_index('Stat').loc['PTS', 'Market status'], SAVED)
        self.fetch.side_effect = None
        self.payload['bookmakers'][0]['markets'][0]['outcomes'][0]['point'] = 21.5
        self.payload['bookmakers'][0]['markets'][0]['outcomes'][1]['point'] = 21.5
        live = self.read(now=self.now + timedelta(minutes=3))
        self.assertFalse(live[SOURCE]['fallback'])
        q = player_quotes(live, 'Test Player', 'draftkings', now=self.now + timedelta(minutes=3))
        self.assertEqual(q.set_index('Stat').loc['PTS', 'Market status'], 'Current')
        self.assertEqual(q.set_index('Stat').loc['PTS', 'Book line'], 21.5)

    def test_account_event_book_and_market_never_cross_match(self):
        self.read()
        self.fail()
        for change in (dict(api_key='other-key'), dict(bookmakers='fanduel'),
                       dict(path='/events/rematch/odds'), dict(markets='player_threes')):
            with self.subTest(change=change), self.assertRaises(OddsFeedError):
                self.read(**change)

    def test_expired_future_timestamp_and_corrupt_backups_rejected(self):
        self.read()
        self.fail()
        for when in (self.now + timedelta(hours=4), self.now - timedelta(minutes=1)):
            with self.assertRaises(OddsFeedError):
                self.read(now=when)
        file = next(Path(self.temp.name).glob('*.json'))
        record = json.loads(file.read_text())
        record['payload']['id'] = 'wrong-game'
        file.write_text(json.dumps(record))
        with self.assertRaises(OddsFeedError):
            self.read()
        file.write_text('{broken')
        with self.assertRaises(OddsFeedError):
            self.read()

    def test_failed_and_mismatched_responses_do_not_overwrite_backup(self):
        self.read()
        self.fetch.return_value = dict(self.payload, id='wrong-game')
        result = self.read(now=self.now + timedelta(minutes=1))
        self.assertTrue(result[SOURCE]['fallback'])
        self.fail()
        self.assertEqual(self.read(now=self.now + timedelta(minutes=2))['id'], 'nyk-phi')

    def test_empty_success_is_authoritative_and_never_revives_withdrawn_quotes(self):
        self.read()
        self.fetch.return_value = dict(self.event, bookmakers=[])
        self.assertEqual(self.read()['bookmakers'], [])
        self.fail()
        self.assertEqual(self.read()['bookmakers'], [])

    def test_unwritable_cache_keeps_live_feed_and_warns(self):
        with patch('saved_sportsbook_lines._write', side_effect=OSError('Read-only disk')):
            current = self.read()
        self.assertFalse(current[SOURCE]['persisted'])
        self.assertIn('Backup could not be saved', source_notice(current, self.now))

    def test_event_list_backup_filters_started_games(self):
        old = dict(self.event, id='old', commence_time=(self.now - timedelta(hours=1)).isoformat())
        self.fetch.return_value = [old, self.event]
        self.read(path='/events')
        self.fail()
        self.assertEqual([e['id'] for e in self.read(path='/events')], ['nyk-phi'])

    def test_age_recomputed_without_fetch_and_unmarked_stale_quotes_still_rejected(self):
        payload = self.read()
        later = self.now + timedelta(minutes=20)
        self.assertTrue(uses_saved_lines(payload, later))
        self.assertIn('20 min ago', source_notice(payload, later))
        q = player_quotes(self.payload, 'Test Player', 'draftkings', now=later)
        self.assertTrue(q['Book implied total'].isna().all())
        # Never allow a future or timezone-less price through the fallback.
        for stamp in ((later + timedelta(minutes=10)).isoformat(), '2026-11-21T19:10:00'):
            bad = deepcopy(payload)
            bad['bookmakers'][0]['markets'][0]['last_update'] = stamp
            q = player_quotes(bad, 'Test Player', 'draftkings', now=later)
            self.assertTrue(q['Book implied total'].isna().all())

    def test_no_unverified_partial_pairs_are_manufactured(self):
        self.payload['bookmakers'][0]['markets'][0]['outcomes'].pop()
        self.read()
        self.fail()
        q = player_quotes(self.read(), 'Test Player', 'draftkings', now=self.now)
        self.assertTrue(q['Book implied total'].isna().all())

    def test_saved_team_targets_require_explicit_opt_in(self):
        payload = dict(self.event, bookmakers=[dict(key='draftkings', markets=[
            dict(key='spreads', last_update=self.now.isoformat(), outcomes=[
                dict(name=self.event['home_team'], point=-5, price=-110),
                dict(name=self.event['away_team'], point=5, price=-110)]),
            dict(key='totals', last_update=self.now.isoformat(), outcomes=[
                dict(name=side, point=220, price=-110) for side in ('Over', 'Under')])])])
        self.fetch.return_value = payload
        self.read(markets='spreads,totals')
        self.fail()
        later = self.now + timedelta(hours=1)
        saved = self.read(markets='spreads,totals', now=later)
        with self.assertRaises(ValueError):
            market_targets(saved, self.event, 'draftkings', later)
        self.assertEqual(market_targets(saved, self.event, 'draftkings', later, allow_saved=True),
                         {'NYK': 112.5, 'PHI': 107.5})
        with self.assertRaises(ValueError):
            market_targets(saved, self.event, 'fanduel', later, allow_saved=True)

    def test_saved_comparisons_reach_edges_with_explicit_provenance(self):
        from tests.test_sportsbook_edges import GameEdgeTests
        fixture = GameEdgeTests()
        fixture.setUp()
        saved = deepcopy(fixture.payload)
        saved[SOURCE] = dict(saved_at=fixture.now.isoformat(), fallback=True)
        later = fixture.now + timedelta(hours=1)
        ranked, total = game_edges(fixture.players, saved, fixture.event, 'fanduel', now=later)
        self.assertGreater(total, 0)
        self.assertTrue(ranked['Market status'].eq(SAVED).all())
        self.assertTrue(ranked['Saved UTC'].eq(fixture.now.isoformat()).all())

    def test_saved_prices_or_saved_game_targets_never_enter_live_tracking(self):
        from tests.test_points_probability import PointsProbabilityTests
        from points_probability import points_quote
        fixture = PointsProbabilityTests()
        fixture.setUp()
        self.assertIsNotNone(fixture.record())
        for change in ({'Market status': SAVED}, {'Saved game lines': True}):
            quote = fixture.quote | change
            probability = points_quote(fixture.row, quote)
            self.assertEqual(probability['Probability status'], 'Experimental')
            row = fixture.row | quote | probability
            self.assertIsNone(snapshot(row, fixture.event, 'fanduel', {}, 'test', now=fixture.now))


if __name__ == '__main__':
    unittest.main()
