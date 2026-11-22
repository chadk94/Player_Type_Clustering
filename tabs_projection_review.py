"""Informational Tabs stat comparisons; never replace the model or its filters."""
from math import isfinite

import numpy as np
import pandas as pd

from edge_review import minute_reference

TABS_STATS = ('PTS', 'REB', 'AST', '3PM', 'STL', 'BLK', 'TOV')
TECHNICAL = ('Compares TabsNBA’s unedited stat projection with this exact sportsbook line, '
             'using Tabs’ own minutes and selected scenario. It is not our model recalculated '
             'with Tabs minutes. Equal-to-line projections are neutral. Missing, stale, OUT '
             'or zero-minute comparisons produce no disagreement flag. This is informational; '
             'our projections, probabilities, edge rankings and More robust filter are unchanged.')


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
    result = {'Tabs projection flag': 'Unavailable', 'Tabs stat projection': np.nan,
              'Tabs projection detail': 'No fresh, unedited Tabs stat projection for this player and game.',
              'Tabs projection technical': TECHNICAL}
    side, stat = row.get('Play'), row.get('Stat')
    line = finite_number(row.get('Book line'))
    if side not in ('Over', 'Under') or line is None or stat not in (*TABS_STATS, 'PRA'):
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
    opposite = value < line - 1e-9 if side == 'Over' else value > line + 1e-9
    relation = 'below' if value < line - 1e-9 else 'above' if value > line + 1e-9 else 'at'
    ours = finite_number(row.get('Projected'))
    detail = f'Our pick: {side} {line:g} {stat}.'
    if ours is not None:
        detail += f' Our projection: {ours:.2f}.'
    detail += f'\nTabs projects {value:.2f} {stat} — {relation} the {line:g} line.'
    stamp = pd.Timestamp(reference.source_updated_at).tz_convert('America/Denver')
    technical = (f'Tabs minutes: {reference.minutes:g}. Scenario: {reference.scenario}. '
                 f'Updated {stamp:%m/%d, %I:%M %p %Z}.\n' +
                 ('Tabs PRA = PTS + REB + AST from the same snapshot.\n' if stat == 'PRA' else '') + TECHNICAL)
    return dict(result, **{'Tabs projection flag': 'OPPOSES' if opposite else 'Neutral' if relation == 'at' else 'Agrees',
                          'Tabs stat projection': value, 'Tabs projection detail': detail,
                          'Tabs projection technical': technical})


def add_tabs_projection_review(edges, references, game, as_of):
    out = edges.copy()
    checks = [tabs_projection_check(row, references, game, as_of) for _, row in out.iterrows()]
    for column in ('Tabs projection flag', 'Tabs stat projection', 'Tabs projection detail', 'Tabs projection technical'):
        out[column] = [check[column] for check in checks]
    return out


def tabs_projection_coverage(edges):
    available = edges.get('Tabs projection flag', pd.Series('Unavailable', index=edges.index)).isin(['OPPOSES', 'Agrees', 'Neutral'])
    if not available.any():
        return 'Tabs stat comparison pending — connect a current Tabs projection with the updated connector. No disagreements are inferred from missing data.'
    return f'Tabs stat comparison: {int(available.sum())} of {len(edges)} plays checked. “Tabs disagrees” means their projection is on the opposite side of the line.'
