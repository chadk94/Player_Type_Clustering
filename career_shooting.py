"""Career-anchored 3P accuracy; attempt volume and other stats are unchanged."""
from functools import lru_cache
import json
from pathlib import Path
import re

import numpy as np

PRIOR_ATTEMPTS = 200.
MIN_CAREER_WEIGHT = .5
SHOOTING_POLICY_VERSION = 'career-3p-200-attempts-floor50-v1'


def parse_career_totals(payload, season):
    """ESPN NBA regular-season integer totals, excluding the target season.

    Traded-player team stints are summed, or the combined row is used once.
    Never average season percentages or include the career-total display row.
    """
    filters = {f['name']: str(f.get('value')) for f in payload.get('filters', [])}
    if filters.get('league') != 'nba' or filters.get('seasontype') != '2':
        raise ValueError('Career feed must be NBA regular season')
    totals = next((c for c in payload.get('categories', []) if c['name'] == 'totals'), None)
    if totals is None:
        raise ValueError('Missing career totals category')
    column = totals['names'].index('threePointFieldGoalsMade-threePointFieldGoalsAttempted')
    by_year = {}
    for item in totals['statistics']:
        year = int(item['season']['year'])
        if year > int(season[:4]):
            continue
        value = str(item['stats'][column]).replace(',', '')
        match = re.fullmatch(r'(\d+)-(\d+)', value)
        if not match:
            raise ValueError(f'Invalid career shooting totals: {value}')
        makes, attempts = map(int, match.groups())
        if makes > attempts:
            raise ValueError('Career makes exceed attempts')
        team = item.get('teamId')
        if team is None:
            if not str(item.get('displayName', '')).endswith('Totals'):
                raise ValueError('Unidentified season total')
            team = '0'
        by_year.setdefault(year, []).append((str(team), makes, attempts))
    seasons = []
    for year, rows in sorted(by_year.items()):
        if len({r[0] for r in rows}) != len(rows):
            raise ValueError('Duplicate season/team totals')
        combined = [r for r in rows if r[0] in ('0', '-1')]
        if len(combined) > 1:
            raise ValueError('Ambiguous combined season total')
        stints = [r for r in rows if r not in combined]
        if combined and stints and combined[0][1:] != (sum(r[1] for r in stints), sum(r[2] for r in stints)):
            raise ValueError('Combined totals disagree with team stints')
        chosen = combined or rows
        seasons.append({'season_end_year': year, 'FG3M': sum(r[1] for r in chosen),
                        'FG3A': sum(r[2] for r in chosen)})
    return {'FG3M': sum(s['FG3M'] for s in seasons),
            'FG3A': sum(s['FG3A'] for s in seasons), 'seasons': seasons}


@lru_cache(maxsize=2)
def load_career_profiles(season):
    path = Path(__file__).parent / 'data' / 'career_three_point_totals.json'
    payload = json.loads(path.read_text())
    if payload['target_season'] != season:
        raise ValueError('Career shooting baseline is for a different season')
    return {int(k): v for k, v in payload['players'].items()}


def blended_accuracy(career_makes, career_attempts, current_makes, current_attempts):
    """Current-season evidence reaches, but never exceeds, 50% at 200 3PA."""
    for makes, attempts in [(career_makes, career_attempts), (current_makes, current_attempts)]:
        if not np.isfinite([makes, attempts]).all() or not 0 <= makes <= attempts:
            raise ValueError('Invalid shooting totals')
    if career_attempts == 0:
        return None, None
    career = career_makes / career_attempts
    weight = max(MIN_CAREER_WEIGHT, PRIOR_ATTEMPTS / (PRIOR_ATTEMPTS + current_attempts))
    current = current_makes / current_attempts if current_attempts else career
    return weight * career + (1 - weight) * current, weight


def apply_accuracy(row, career, observed):
    """Update only FG3M plus transparent shooting provenance on one baseline row."""
    row.update({'Expected 3P%': np.nan, 'Career 3P%': np.nan, 'Season 3P%': np.nan,
                'Career 3P weight': np.nan, 'Season 3PA sample': np.nan,
                '3P accuracy source': 'Existing baseline — no prior NBA attempts'})
    if row.get('FG3A', 0) > 0 and np.isfinite(row.get('FG3M', np.nan)):
        row['Expected 3P%'] = row['FG3M'] / row['FG3A']
    if career is None:
        row['3P accuracy source'] = 'Existing baseline — career history unavailable'
        return
    cm, ca = career['FG3M'], career['FG3A']
    if ca <= 0:
        return
    sm, sa = 0., 0.
    source = 'Career only — no current shooting sample'
    if observed is not None:
        # Prefer exact box-score totals. Older verified snapshots contain only
        # full-precision averages; GP*average is the equivalent count.
        sm = observed.get('Season FG3M total', observed.get('FG3M', np.nan) * observed.GP)
        sa = observed.get('Season FG3A total', observed.get('FG3A', np.nan) * observed.GP)
        if np.isfinite([sm, sa]).all() and 0 <= sm <= sa:
            source = 'Career + current season' if sa > 0 else source
        else:
            sm, sa = 0., 0.
            source = 'Career only — current shooting data unavailable'
    pct, weight = blended_accuracy(cm, ca, sm, sa)
    row.update({'Expected 3P%': pct, 'Career 3P%': cm / ca,
                'Season 3P%': sm / sa if sa else np.nan, 'Career 3P weight': weight,
                'Season 3PA sample': sa, '3P accuracy source': source})
    attempts = row.get('FG3A', np.nan)
    if np.isfinite(attempts) and attempts >= 0:
        row['FG3M'] = attempts * pct
