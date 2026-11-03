"""Player-baseline exposure policy, independent of opponent-effect smoothing."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

PLAYER_RATE_POLICY_VERSION = 'stocks-1000-prior-if-500-last-season-v1'
STOCKS_MIN_PRIOR_SEASON_MINUTES = 500.
STOCKS_PRIOR_MINUTES = 1000.


@lru_cache(maxsize=2)
def load_prior_season_minutes(season):
    """Exact NBA regular-season totals by NBA player ID, not rounded ESPN MPG."""
    path = Path(__file__).parent / 'data' / 'player_rate_prior_minutes.json'
    payload = json.loads(path.read_text())
    start = int(season[:4])
    expected = f'{start - 1}-{str(start)[-2:]}'
    if payload['target_season'] != season or payload['source_season'] != expected:
        raise ValueError('Player baseline minutes have the wrong source season')
    result = {int(key): float(row['MIN']) for key, row in payload['players'].items()}
    if any(not np.isfinite(value) or value < 0 for value in result.values()):
        raise ValueError('Invalid prior-season player minutes')
    return result


def stocks_prior_minutes(last_season_minutes, default=200.):
    """Inclusive 500-minute cutoff; unknown/invalid exposure keeps the base prior."""
    try:
        minutes = float(last_season_minutes)
    except (TypeError, ValueError):
        return default
    return (STOCKS_PRIOR_MINUTES if np.isfinite(minutes)
            and minutes >= STOCKS_MIN_PRIOR_SEASON_MINUTES else default)
