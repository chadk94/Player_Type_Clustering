"""Bounded, private-session reuse of unchanged game calculations.

Inputs are fingerprinted by value (including DataFrame attributes). Clock
boundaries retain the exact existing freshness rules, including equality at
expiration; this is not a grace period for stale prices or injury reports.
"""
from bisect import bisect_left, bisect_right
from copy import deepcopy
import hashlib
import pickle

import pandas as pd

from edge_review import CONFIG


class GameCalculationCache:
    def __init__(self, limit=48):
        self.limit = limit
        self.entries = {}

    def clear(self):
        self.entries.clear()

    def get(self, inputs, now, calculate, deadlines=lambda: ()):
        key = hashlib.sha256(pickle.dumps(inputs, protocol=5)).digest()
        stamp = pd.Timestamp(now).value
        entry = self.entries.get(key)
        boundaries = entry[0] if entry else tuple(sorted(set(deadlines())))
        period = (bisect_left(boundaries, stamp), bisect_right(boundaries, stamp))
        if entry is None or entry[1] != period:
            result = calculate()
            self.entries.pop(key, None)
            if len(self.entries) >= self.limit:
                del self.entries[next(iter(self.entries))]
            self.entries[key] = (boundaries, period, deepcopy(result))
        else:
            # Keep frequently revisited games when a whole slate is inspected.
            self.entries[key] = self.entries.pop(key)
        return deepcopy(self.entries[key][2])


def source_deadlines(references=None, injury_stamp=None):
    """All instants at which existing reference/report validity can change."""
    values = []
    def add(raw, hours):
        try:
            stamp = pd.Timestamp(raw)
            if pd.notna(stamp) and stamp.tzinfo is not None:
                values.extend((stamp.value, (stamp + pd.Timedelta(hours=hours)).value))
        except (TypeError, ValueError, OverflowError):
            pass
    if references is not None:
        for row in references.to_dict('records'):
            add(row.get('updated_at'), CONFIG['minutes_source_max_hours'])
            if row.get('requires_source_timestamp', False):
                add(row.get('source_updated_at'), row.get('source_max_age_hours', CONFIG['minutes_source_max_hours']))
    if injury_stamp is not None:
        add(injury_stamp, CONFIG['injury_max_age_minutes'] / 60)
    return values
