"""Bounded background reads that never make a UI rerun wait for the network."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from time import monotonic


_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix='prop-feed')


@dataclass(frozen=True)
class FeedSnapshot:
    value: object
    error: Exception | None
    pending: bool
    revision: int
    checked_at: datetime | None


class BackgroundFeed:
    """One in-flight request per source; failures retain the last good result.

    Readers do not touch Streamlit or session state. Callers still validate the
    source timestamp and apply their existing freshness/coverage rules.
    """
    def __init__(self, executor=None):
        self.executor = executor or _POOL
        self.lock = Lock()
        self.future = None
        self.value = self.error = self.checked_at = None
        self.revision = 0
        self.next_check = 0.

    def invalidate(self):
        with self.lock:
            self.next_check = 0.

    def _collect(self, refresh_after):
        if self.future is not None and self.future.done():
            try:
                value = self.future.result()
            except Exception as exc:
                self.error = exc
            else:
                self.value, self.error = value, None
            self.future = None
            self.checked_at = datetime.now(timezone.utc)
            self.revision += 1
            self.next_check = monotonic() + refresh_after

    def read(self, fetch, *, refresh_after, force=False):
        with self.lock:
            self._collect(refresh_after)
            if self.future is None and (force or monotonic() >= self.next_check):
                self.future = self.executor.submit(fetch)
            # A fast or already-completed read can be used on this same render.
            self._collect(refresh_after)
            return FeedSnapshot(deepcopy(self.value), self.error, self.future is not None,
                                self.revision, self.checked_at)
