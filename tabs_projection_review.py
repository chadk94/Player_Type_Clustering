"""Tabs stat comparisons and agreement screening; never change model values."""
from math import isfinite

import numpy as np
import pandas as pd

from edge_review import minute_reference
from sportsbook_lines import PLAY_THRESHOLD_PCT, american_probability, signed_edge_pct

TABS_STATS = ('PTS', 'REB', 'AST', '3PM', 'STL', 'BLK', 'TOV')
TECHNICAL = (f'Compares TabsNBA’s unedited projection with the same price-adjusted market '
             f'projection used by Edge %. Paired Over/Under prices remove the book margin; '
             f'Over-only quotes retain it. Edge = (Tabs / market projection − 1) × 100, '
             f'reversed for Unders and rounded to two decimals. More than {PLAY_THRESHOLD_PCT:g}% '
             f'in our direction agrees; less than −{PLAY_THRESHOLD_PCT:g}% disagrees; the rest is neutral. '
             'This is a projection gap, not Tabs win probability or expected return. '
             'Uses Tabs’ own minutes and selected scenario. Missing, stale, OUT, zero-minute '
             'or unavailable-price comparisons produce no Tabs flag. More robust plays requires '
             'Tabs agreement plus the existing minutes, role and injury checks. Our projections, '
             'probabilities and edge rankings are unchanged.')


def finite_number(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
        return number if isfinite(number) and number >= 0 else None
    except (ValueError, TypeError):
        return None


def tabs_stat_values(row, *, unedited=False):
    """CSV stats can follow My Min / Boost edits, unlike the separate Tabs Min."""
    if not unedited:
        own, original = finite_number(row.get('My Min')), finite_number(row.get('Tabs Min'))
        boost = finite_number(row.get('Boost %'))
        if own is None or original is None or abs(own-original) > 1e-9 or boost != 0:
            return {f'Tabs projection {stat}': np.nan for stat in TABS_STATS}
    return {f'Tabs projection {stat}': value if (value := finite_number(row.get(stat))) is not None else np.nan
            for stat in TABS_STATS}


def tabs_projection_check(row, references, game, as_of):
    result = {'Tabs projection flag': 'Unavailable', 'Tabs stat projection': np.nan, 'Tabs edge %': np.nan,
              'Tabs projection detail': 'No fresh, unedited Tabs stat projection for this player and game.',
              'Tabs projection technical': TECHNICAL}
    side, stat = row.get('Play'), row.get('Stat')
    line = finite_number(row.get('Book line'))
    if side not in ('Over', 'Under') or line is None or stat not in (*TABS_STATS, 'PRA'):
        return result
    implied = finite_number(row.get('Book implied total'))
    try:
        price = float(row.get(f'{side} odds'))
        american_probability(price)
    except (TypeError, ValueError):
        return result
    if implied is None or implied <= 0:
        return result
    reference = minute_reference(references, row, game, as_of)
    if reference is None or reference.source != 'TabsNBA' or reference.minutes <= 0 or reference.get('source_status') == 'Out':
        return result
    if not reference.get('requires_source_timestamp', False):
        return result
    components = ('PTS', 'REB', 'AST') if stat == 'PRA' else (stat,)
    values = [finite_number(reference.get(f'Tabs projection {s}')) for s in components]
    if any(v is None for v in values):
        return result
    value = sum(values)
    edge = float(signed_edge_pct(value, implied)) * (1 if side == 'Over' else -1)
    status = 'Agrees' if edge > PLAY_THRESHOLD_PCT else 'OPPOSES' if edge < -PLAY_THRESHOLD_PCT else 'Neutral'
    ours = finite_number(row.get('Projected'))
    detail = f'Our pick: {side} {line:g} {stat} at {price:+g}.'
    if ours is not None:
        detail += f' Our projection: {ours:.2f}.'
    detail += f'\nTabs projects {value:.2f} {stat}; price-adjusted market: {implied:.2f}.'
    verdict = ('Agrees' if status == 'Agrees' else 'Disagrees' if status == 'OPPOSES'
               else f'Neutral — within ±{PLAY_THRESHOLD_PCT:g}%')
    detail += f'\nTabs edge: {edge:+.2f}% for this {side}. {verdict}.'
    stamp = pd.Timestamp(reference.source_updated_at).tz_convert('America/Denver')
    technical = (f'Tabs minutes: {reference.minutes:g}. Scenario: {reference.scenario}. '
                 f'Updated {stamp:%m/%d, %I:%M %p %Z}.\n' +
                 ('Tabs PRA = PTS + REB + AST from the same snapshot.\n' if stat == 'PRA' else '') + TECHNICAL)
    return dict(result, **{'Tabs projection flag': status, 'Tabs edge %': edge,
                          'Tabs stat projection': value, 'Tabs projection detail': detail,
                          'Tabs projection technical': technical})


def add_tabs_projection_review(edges, references, game, as_of):
    out = edges.copy()
    checks = [tabs_projection_check(row, references, game, as_of) for _, row in out.iterrows()]
    for column in ('Tabs projection flag', 'Tabs stat projection', 'Tabs edge %', 'Tabs projection detail', 'Tabs projection technical'):
        out[column] = [check[column] for check in checks]
    if 'Robust play' in out:
        passed = out['Robust play'].eq(True)
        agrees = out['Tabs projection flag'].eq('Agrees')
        demoted = passed & ~agrees
        out['Robust play'] = passed & agrees
        if 'Review group' in out:
            out.loc[demoted, 'Review group'] = 'Needs review'
            out.loc[demoted & out['Tabs projection flag'].eq('Unavailable'), 'Review group'] = 'Unchecked'
        if 'Review flags' in out:
            out.loc[demoted, 'Review flags'] = 'Review'
        if 'Review detail' in out:
            notes = {
                'Agrees': 'Tabs agrees with this pick; all robustness checks passed.',
                'OPPOSES': f'Minutes, role and injury checks passed, but Tabs favors the opposite side by more than {PLAY_THRESHOLD_PCT:g}% after accounting for the prices.',
                'Neutral': f'Minutes, role and injury checks passed, but Tabs does not clear the {PLAY_THRESHOLD_PCT:g}% price-adjusted edge cutoff.',
                'Unavailable': 'Minutes, role and injury checks passed; a fresh Tabs stat comparison is still needed.',
            }
            out.loc[passed, 'Review detail'] = out.loc[passed, 'Tabs projection flag'].map(notes)
    return out


def tabs_projection_coverage(edges):
    available = edges.get('Tabs projection flag', pd.Series('Unavailable', index=edges.index)).isin(['OPPOSES', 'Agrees', 'Neutral'])
    if not available.any():
        return 'Tabs stat comparison pending — connect a current Tabs projection with the updated connector. No disagreements are inferred from missing data.'
    return f'Tabs stat comparison: {int(available.sum())} of {len(edges)} plays checked. Green: more than {PLAY_THRESHOLD_PCT:g}% edge for this pick; yellow: more than {PLAY_THRESHOLD_PCT:g}% against; gray: within ±{PLAY_THRESHOLD_PCT:g}%. All use the quoted prices.'
