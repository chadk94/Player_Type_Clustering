"""Mean-preserving count distributions for non-points player props."""
from functools import lru_cache
from pathlib import Path
import json
import math

import numpy as np
from scipy.stats import betabinom, binom, nbinom, poisson

from points_probability import PROBABILITY_COLUMNS, points_quote

COUNT_STATS = ('REB', 'AST', '3PM', 'STL', 'BLK', 'TOV', 'PRA', 'PF')
COUNT_VERSION = 'count-distributions-v1'
LOW_WEIGHT = 0.08
LOW_SCALE = 0.25
HIGH_SCALE = (1 - LOW_WEIGHT * LOW_SCALE) / (1 - LOW_WEIGHT)


@lru_cache(maxsize=1)
def count_profile():
    return json.loads((Path(__file__).parent / 'data/prop_distribution_profile.json').read_text())


def count_cdf(k, mean, dispersion):
    """Var = mean + dispersion * mean². Use a soft binomial tail for underdispersion."""
    k, mu, a = np.broadcast_arrays(k, mean, dispersion)
    cdf = poisson.cdf(k, mu)
    over = (a > 1e-5) & (mu > 0)
    if over.any():
        size = 1 / a[over]
        cdf[over] = nbinom.cdf(k[over], size, size / (size + mu[over]))
    under = (a < -1e-5) & (mu > 0)
    if under.any():
        trials = np.maximum(np.ceil(mu[under]), np.ceil(-0.95 / a[under]))
        cdf[under] = 0.05 * cdf[under] + 0.95 * binom.cdf(k[under], trials, mu[under] / trials)
    return cdf


def distribution_cdf(k, mean, dispersion, method, stat):
    """Vectorized for historical scoring and for both sides of an integer line."""
    k, mu, a = np.broadcast_arrays(np.atleast_1d(k), mean, dispersion)
    if method == 'poisson':
        return poisson.cdf(k, mu)
    if stat == 'PF':
        # Model ordinary six-foul games, with a beta-binomial when variance warrants it.
        p = mu / 6
        cdf = binom.cdf(k, 6, p)
        varying = (a > 1e-5) & (p > 0) & (p < 1) & (k >= 0) & (k < 6)
        if varying.any():
            strength = 1 / a[varying] - 1
            alpha, beta = p[varying] * strength, (1 - p[varying]) * strength
            cdf[varying] = sum((j <= k[varying]) * betabinom.pmf(j, 6, alpha, beta) for j in range(6))
        return cdf
    if method.endswith('_mixture'):
        # A low-volume branch and a compensating normal-volume branch preserve the input mean.
        return (LOW_WEIGHT * count_cdf(k, mu * LOW_SCALE, a) +
                (1 - LOW_WEIGHT) * count_cdf(k, mu * HIGH_SCALE, a))
    return count_cdf(k, mu, a)


def prop_quote(player, quote, observed=None):
    stat = quote['Stat']
    if stat == 'PTS':
        return points_quote(player, quote, observed)
    result = dict.fromkeys(PROBABILITY_COLUMNS, np.nan)
    result['Probability status'] = 'Needs a current paired line'
    manual = stat == 'PF' and quote.get('Market status') == 'Manual line'
    if stat not in COUNT_STATS or (quote.get('Market status') != 'Current' and not manual):
        return result
    try:
        line = float(quote['Book line'])
        mean = float(quote['Projected'])
        minutes = float(player['Projected minutes'])
        if not math.isfinite(line) or line < 0 or not (line * 2).is_integer():
            raise ValueError('Needs an integer or half-point line')
        if not math.isfinite(mean) or mean < 0 or not math.isfinite(minutes) or minutes <= 0:
            raise ValueError('Needs a valid projection and positive minutes')
        if player.get('Estimated stat profile', False):
            raise ValueError('Needs an individual stat baseline')
        if stat == 'PF' and mean > 6:
            raise ValueError('Foul projection exceeds the modeled six-foul range')
        profile = count_profile()
        fit = profile['stats'][stat]
        pid = str(int(player['PLAYER_ID']))
        dispersion = fit['players'].get(pid, fit['pool'])
        k = math.floor(line)
        low, at = distribution_cdf([k-1, k], mean, dispersion, fit['method'], stat).clip(0, 1)
        under = float(low if line.is_integer() else at)
        result.update(zip(PROBABILITY_COLUMNS, [float(1 - at), under, float(at - under)]))
        source = 'Player history' if pid in fit['players'] else 'Stat pool'
        result.update({'Probability status': 'Experimental', 'Probability model': COUNT_VERSION,
                       'Probability profile': profile['profile_hash'], 'Variability source': source,
                       'Distribution': fit['method'], 'Dispersion': dispersion})
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        result['Probability status'] = str(exc)
    return result
