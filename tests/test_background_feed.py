from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from time import perf_counter
from unittest import TestCase
from unittest.mock import Mock

from background_feed import BackgroundFeed


class ImmediateExecutor:
    """Deterministic completed jobs for existing UI-state regression tests."""
    def submit(self, function):
        future = Future()
        try:
            future.set_result(function())
        except Exception as exc:
            future.set_exception(exc)
        return future


class BackgroundFeedTests(TestCase):
    def test_slow_refresh_does_not_block_or_duplicate_and_keeps_original_timestamp(self):
        entered, release = Event(), Event()
        with ThreadPoolExecutor(max_workers=1) as pool:
            cache = BackgroundFeed(pool)
            def slow():
                entered.set()
                release.wait(5)
                return {'source_time': '2026-11-26T10:00:00Z', 'minutes': [30]}
            fetch = Mock(side_effect=slow)
            try:
                started = perf_counter()
                result = cache.read(fetch, refresh_after=55)
                self.assertLess(perf_counter() - started, .5)
                self.assertTrue(entered.wait(1))
                self.assertTrue(result.pending)
                for _ in range(25):
                    self.assertTrue(cache.read(fetch, refresh_after=55, force=True).pending)
                fetch.assert_called_once()
            finally:
                release.set()
            cache.future.result(timeout=2)
            result = cache.read(fetch, refresh_after=55)
            self.assertEqual(result.revision, 1)
            self.assertEqual(result.value['source_time'], '2026-11-26T10:00:00Z')
            result.value['minutes'][0] = 999
            self.assertEqual(cache.read(fetch, refresh_after=55).value['minutes'], [30])
            fetch.assert_called_once()

    def test_failure_keeps_verified_value_and_retry_recovers(self):
        cache = BackgroundFeed(ImmediateExecutor())
        original = cache.read(lambda: {'minutes': [32]}, refresh_after=55)
        broken = cache.read(Mock(side_effect=ValueError('Unavailable')), refresh_after=55, force=True)
        self.assertEqual(broken.value, original.value)
        self.assertEqual(broken.revision, 2)
        self.assertIsInstance(broken.error, ValueError)
        fetch = Mock(return_value={'minutes': [28]})
        cache.read(fetch, refresh_after=55)
        fetch.assert_not_called()
        cache.invalidate()
        recovered = cache.read(fetch, refresh_after=55)
        self.assertEqual(recovered.value['minutes'], [28])
        self.assertIsNone(recovered.error)

    def test_different_connections_cannot_share_results(self):
        a, b = BackgroundFeed(ImmediateExecutor()), BackgroundFeed(ImmediateExecutor())
        self.assertEqual(a.read(lambda: 'a', refresh_after=60).value, 'a')
        self.assertEqual(b.read(lambda: 'b', refresh_after=60).value, 'b')
        self.assertEqual(a.read(lambda: 'b', refresh_after=60).value, 'a')
