"""Player-baseline exposure policy, independent of opponent-effect smoothing."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

PLAYER_RATE_POLICY_VERSION = 'player-priors-taper8-current20-appearances-v1'
STOCKS_MIN_PRIOR_SEASON_MINUTES = 500.
STOCKS_PRIOR_MINUTES = 1000.
PLAYER_TAPER_START = 8.
PLAYER_FULL_CURRENT_APPEARANCES = 20.


def effective_player_prior(prior_minutes, appearances):
    """Keep early exposure weights; remove historical evidence by appearance 20.

    These are completed player appearances, never team games or projected
    minutes. Missing observations are handled by the caller's existing fallback.
    """
    if not np.isfinite([prior_minutes, appearances]).all() or min(prior_minutes, appearances) < 0:
        raise ValueError('Player prior and appearances must be finite and nonnegative')
    fraction = np.clip((PLAYER_FULL_CURRENT_APPEARANCES - appearances) /
                       (PLAYER_FULL_CURRENT_APPEARANCES - PLAYER_TAPER_START), 0., 1.)
    return float(prior_minutes * fraction)


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
