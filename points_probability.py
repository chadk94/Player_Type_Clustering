"""Experimental scoring-component probabilities; never modify projection means."""
from functools import lru_cache
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

from live_projections import match_player_stats
from sportsbook_lines import player_quotes

MODEL_VERSION = 'points-mixture-0.5-0.05-0.25-v1'
PROBABILITY_COLUMNS = ['Exp. P(over)', 'Exp. P(under)', 'Exp. P(push)']
COMPONENTS = ('FTM', 'FG2M', 'FG3M')


@lru_cache(maxsize=1)
def model_profile():
    return json.loads((Path(__file__).parent / 'data/points_probability_profile.json').read_text())


def scoring_context(roster, current):
    """Keep NBA IDs for the fitted profile and ESPN IDs for result matching."""
    matches = match_player_stats(roster, current)
    return {int(pid): row.to_dict() for pid, row in matches.items()}


def scoring_components(row, observed=None, profile=None):
    profile = model_profile() if profile is None else profile
    mean, threes = float(row['PTS']), float(row['FG3M'])
    minutes = float(row['Projected minutes'])
    if not all(np.isfinite([mean, threes, minutes])) or mean < 0 or threes < 0 or minutes <= 0:
        raise ValueError('Needs valid points, threes and positive minutes')
    remaining = mean - 3 * threes
    if remaining < -1e-9:
        raise ValueError('Points and threes estimates are inconsistent; probability unavailable')
    remaining = max(0., remaining)
    past = profile['scoring'].get(str(int(row['PLAYER_ID'])))
    prior = pd.Series(past['totals'], dtype=float) if past else None
    current = None
    if observed is not None and float(row.get('Actual sample minutes', 0)) > 0:
        current = pd.Series({s: float(observed.get(s, np.nan)) * float(observed.get('GP', 0))
                             for s in ['MIN', 'FTM', 'FGM', 'FG3M', 'FGA', 'FG3A']})
        current['FG2M'] = current.FGM - current.FG3M
        if not np.isfinite(current).all() or (current < 0).any() or current.MIN <= 0:
            raise ValueError('Incomplete observed scoring mix')
    if current is not None and (prior is None or str(row.get('Stat baseline', '')).startswith('Current season')):
        rates = current / current.MIN
    elif current is not None and prior is not None:
        rates = (200 * prior / prior.MIN + current) / (200 + current.MIN)
    elif prior is not None:
        rates = prior / prior.MIN
    else:
        raise ValueError('Needs individual scoring history')
    evidence = prior.add(current, fill_value=0) if prior is not None and current is not None else (prior if prior is not None else current)
    ratio = evidence.FG3A / evidence.FGA if evidence.FGA > 0 else .3
    style = 'twos-heavy' if ratio <= .1 else 'threes-heavy' if ratio >= .6 else 'mixed'
    weight = evidence.MIN / (evidence.MIN + 30)
    mix = weight * rates[list(COMPONENTS)].to_numpy(float) + (1-weight) * np.array(profile['peers'][style])
    denominator = mix[0] + 2 * mix[1]
    if not np.isfinite(mix).all() or (mix < 0).any() or (remaining > 0 and denominator <= 0):
        raise ValueError('Incomplete scoring mix')
    scale = remaining / denominator if denominator > 0 else 0.
    lambdas = (float(mix[0]*scale), float(mix[1]*scale), threes)
    fit = profile['fit']['players'].get(str(int(row['PLAYER_ID'])))
    phi = float(fit['phi'] if fit else profile['fit']['pools'][style])
    return lambdas, phi, style, 'Player history' if fit else 'Scoring-style pool'


def component_pmf(lambdas, phi, limit=256):
    lam = np.asarray(lambdas, dtype=float)
    if lam.shape != (3,) or not np.isfinite(lam).all() or (lam < 0).any() or not np.isfinite(phi) or phi < 0:
        raise ValueError('Invalid scoring distribution')
    a = float(lam.sum())
    den = 1 + phi * a
    p = np.zeros(limit+1)
    p[0] = math.exp(-math.log1p(phi*a)/phi) if phi > 1e-12 else math.exp(-a)
    for n in range(1, limit+1):
        p[n] = sum(lam[j-1] * (j + phi*(n-j)) * p[n-j] for j in range(1, min(3, n)+1)) / (den*n)
    if abs(p.sum()-1) > 1e-10 or abs(p @ np.arange(limit+1) - lam @ [1, 2, 3]) > 1e-7:
        if limit >= 2048:
            raise ValueError('Scoring distribution could not be resolved')
        return component_pmf(lambdas, phi, limit*2)
    return p


@lru_cache(maxsize=4096)
def points_pmf(lambdas, phi):
    """Selected offline mixture, preserving all three expected scoring components."""
    lam = np.asarray(lambdas)
    low = component_pmf(lam*.25, phi*.5)
    high = component_pmf(lam*((1-.05*.25)/.95), phi*.5)
    p = np.zeros(max(len(low), len(high)))
    p[:len(low)] += .05*low
    p[:len(high)] += .95*high
    p.setflags(write=False)
    return p


def line_probabilities(lambdas, phi, line):
    line = float(line)
    if not np.isfinite(line) or line < 0 or abs(line*2-round(line*2)) > 1e-8:
        raise ValueError('Needs an integer or half-point line')
    p = points_pmf(tuple(lambdas), float(phi))
    k = math.floor(line)
    push = float(p[k]) if line.is_integer() and k < len(p) else 0.
    under = float(p[:k if line.is_integer() else k+1].sum())
    over = float(p[k+1:].sum())
    return dict(zip(PROBABILITY_COLUMNS, [over, under, push]))


def points_quote(row, quote, observed=None):
    result = {column: np.nan for column in PROBABILITY_COLUMNS}
    result['Probability status'] = 'Needs a current paired points line'
    if quote.get('Market status') != 'Current':
        return result
    try:
        if row.get('Estimated stat profile', False):
            raise ValueError('Needs individual scoring history')
        lam, phi, style, source = scoring_components(row, observed)
        result.update(line_probabilities(lam, phi, quote['Book line']))
        result.update({'Probability status': 'Experimental', 'Probability model': MODEL_VERSION,
                       'Probability profile': model_profile()['profile_hash'],
                       'Scoring style': style, 'Variability source': source, 'Variability phi': phi,
                       'Expected FTM': lam[0], 'Expected 2PM': lam[1], 'Expected 3PM': lam[2]})
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        result['Probability status'] = str(exc)
    return result


def game_quote_probabilities(players, payload, bookmaker, context, evaluate, stats, now=None):
    """Use the same player means and fresh paired quotes before edge filtering."""
    result = []
    for _, row in players.iterrows():
        player = row.to_dict()
        observed = context.get(int(row.PLAYER_ID))
        quotes = player_quotes(payload, row.Player, bookmaker, now=now)
        for quote in quotes.to_dict('records'):
            stat = quote['Stat']
            if stat not in stats or (stat != 'PTS' and quote['Market status'] != 'Current'):
                continue
            quote['Projected'] = player.get('FG3M' if stat == '3PM' else stat, np.nan)
            result.append({**player, **quote, **evaluate(player, quote, observed)})
    if not result:
        return pd.DataFrame(columns=['PLAYER_ID', 'Player', 'Team', 'Stat', *PROBABILITY_COLUMNS])
    return pd.DataFrame(result)


def game_points_probabilities(players, payload, bookmaker, context, now=None):
    return game_quote_probabilities(players, payload, bookmaker, context, points_quote, ('PTS',), now=now)
