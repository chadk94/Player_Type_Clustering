"""Player assist opportunities, applied before the existing matchup and team allocation."""
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import monotonic

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguedashptstats
from requests import RequestException

POLICY_VERSION = 'assist-passing-50-50-200-v1'
PRIOR_MINUTES = 200.
SEED_END = date(2026, 11, 9)
DATA = Path(__file__).parent / 'data'
STATS = ['AST', 'POTENTIAL_AST', 'PASSES_MADE', 'MIN']
SAMPLE = ['box_gp', 'box_ast', 'box_min', 'tracked_gp', 'tracked_ast', 'potential', 'passes']


@lru_cache(maxsize=2)
def bundled(season):
    if season != '2026-27':
        raise ValueError('No verified passing prior for this season')
    prior = pd.read_csv(DATA / 'assist_passing_prior_2025-26.csv').set_index('PLAYER_ID', verify_integrity=True)
    seed = pd.read_csv(DATA / f'assist_passing_{season}_through_{SEED_END}.csv').set_index('PLAYER_ID', verify_integrity=True)
    if not (np.isfinite(prior[STATS]).all().all() and prior[STATS].ge(0).all().all()
            and np.isfinite(seed[SAMPLE]).all().all() and seed[SAMPLE].ge(0).all().all()
            and seed.tracked_gp.le(seed.box_gp).all() and seed.tracked_ast.le(seed.box_ast).all()):
        raise ValueError('Invalid archived passing data')
    return prior, seed


def fetch_passing_day(season, day):
    """Fetch one NBA regular-season game day for all players."""
    nba_date = day.strftime('%m/%d/%Y')
    payload = leaguedashptstats.LeagueDashPtStats(
        pt_measure_type='Passing', player_or_team='Player', per_mode_simple='Totals',
        season=season, season_type_all_star='Regular Season', date_from_nullable=nba_date,
        date_to_nullable=nba_date, timeout=5).get_dict()
    expected = dict(Season=season, SeasonType='Regular Season', PtMeasureType='Passing',
                    PlayerOrTeam='Player', PerMode='Totals', DateFrom=nba_date, DateTo=nba_date)
    if any(payload['parameters'].get(k) != v for k, v in expected.items()):
        raise ValueError('Passing feed returned a different scope')
    table = payload['resultSets'][0]
    rows = pd.DataFrame(table['rowSet'], columns=table['headers'])
    rows = rows.set_index('PLAYER_ID', verify_integrity=True)[['GP', *STATS]].apply(pd.to_numeric)
    if (not np.isfinite(rows).all().all() or rows.lt(0).any().any()
            or (rows[['GP','AST','POTENTIAL_AST','PASSES_MADE']] % 1).ne(0).any().any()):
        raise ValueError('Invalid NBA passing counts')
    return rows


class DailyPassingFeed:
    """Share completed requests across sessions and keep the last successful day."""
    def __init__(self, fetch=fetch_passing_day):
        self.fetch = fetch
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.lock = Lock()
        self.pending = {}
        self.refresh_at = {}
        self.tables = {}

    def read(self, season, days, timeout=3):
        keys = [(season, day) for day in sorted(days, reverse=True)]
        with self.lock:
            for key in keys:
                if key not in self.pending and monotonic() >= self.refresh_at.get(key, 0):
                    self.pending[key] = self.pool.submit(self.fetch, *key)
            jobs = [self.pending[key] for key in keys if key in self.pending]
        if jobs:
            wait(jobs, timeout=timeout)
        with self.lock:
            for key in keys:
                job = self.pending.get(key)
                if job is None or not job.done():
                    continue
                del self.pending[key]
                try:
                    frame = job.result()
                    if frame.empty:
                        raise ValueError('No passing rows published for a completed game day')
                    self.tables[key] = frame
                    recent = key[1] >= datetime.now(timezone.utc).date() - timedelta(days=2)
                    ttl = 900 if recent else 86400
                except (RequestException, KeyError, IndexError, TypeError, ValueError, OSError):
                    ttl = 300
                self.refresh_at[key] = monotonic() + ttl
            return {day: self.tables[season, day] for day in days if (season, day) in self.tables}


def verified_days(season, logs, espn_to_nba, read_days):
    """Match daily tracking to a completed game by player, game day, assists and minutes."""
    if logs.empty or not espn_to_nba:
        return [], []
    rows = logs[['ESPN_PLAYER_ID', 'GAME_DATE', 'MIN', 'AST']].copy()
    rows['PLAYER_ID'] = rows.ESPN_PLAYER_ID.astype(str).map(espn_to_nba)
    rows['day'] = pd.to_datetime(rows.GAME_DATE, utc=True).dt.tz_convert('America/New_York').dt.date
    rows = rows[rows.PLAYER_ID.notna() & rows.day.gt(SEED_END)]
    if rows.empty:
        return [], []
    days = set(rows.day)
    tables = read_days(season, days)
    samples = []
    for day, boxes in rows.groupby('day'):
        if day not in tables:
            continue
        boxes = boxes[~boxes.PLAYER_ID.duplicated(keep=False)].set_index('PLAYER_ID')
        pair = boxes.join(tables[day], how='inner', rsuffix='_nba')
        # NBA tracking rounds to whole minutes. ESPN may also round its box.
        good = (pair.GP.eq(1) & pair.AST.eq(pair.AST_nba)
                & (pair.MIN-pair.MIN_nba).abs().le(.5+1/60))
        matched = pair.loc[good].rename(columns={'MIN':'box_min', 'AST':'box_ast',
                         'AST_nba':'tracked_ast', 'POTENTIAL_AST':'potential', 'PASSES_MADE':'passes'})
        samples.append(matched.assign(box_gp=1, tracked_gp=1)[SAMPLE])
    return samples, sorted(days-tables.keys())


def adjusted_assist_samples(season, logs, espn_to_nba, read_days):
    """Precompute the two empirical conversion estimates exactly once per snapshot."""
    prior, seed = bundled(season)
    days, unavailable = verified_days(season, logs, espn_to_nba, read_days)
    sample = pd.concat([seed, *days]).groupby(level=0).sum()
    previous = prior.reindex(sample.index).fillna(0.)
    league = prior[STATS].sum()
    delta = pd.Series(0., index=sample.index)
    for column, volume, regularizer in [('POTENTIAL_AST',sample.potential,25.),
                                        ('PASSES_MADE',sample.passes,150.)]:
        past = previous[column]
        past_yield = (previous.AST + regularizer*league.AST/league[column])/(past+regularizer)
        pace = (past/previous.MIN).where((past>0) & (previous.MIN>0), league[column]/league.MIN)
        exposure = PRIOR_MINUTES * pace
        exposure = exposure.where(past.eq(0), np.minimum(exposure, past+regularizer))
        conversion = (sample.tracked_ast+exposure*past_yield)/(volume+exposure)
        delta += .5 * (volume*conversion - sample.tracked_ast).where(volume.gt(0), 0.)
    return sample.assign(delta=delta).to_dict('index'), unavailable


def apply_assist_sample(sample, observed, actual_rate):
    """Return the corrected current rate or retain the existing box-score rate."""
    if not sample or not np.isfinite(actual_rate) or actual_rate < 0 or sample['tracked_gp'] <= 0:
        return actual_rate, False
    gp = observed.GP
    total = observed.get('Season AST total', np.nan)
    has_exact_total = pd.notna(total) and np.isfinite(total)
    if not has_exact_total:
        total = observed.AST * gp
    # Exact season counts when supplied, otherwise ESPN may round its per-game
    # average. A summary must include the archived/verified games before use.
    tolerance = 1e-6 if has_exact_total else .051*gp + 1e-6
    minutes = observed.MIN * gp
    if (gp < sample['box_gp'] or not np.isfinite(total) or total + tolerance < sample['box_ast']
            or minutes + .52*gp < sample['box_min'] or
            (gp == sample['box_gp'] and (abs(total-sample['box_ast']) > tolerance
                                        or abs(minutes-sample['box_min']) > .52*gp))):
        return actual_rate, False
    if gp == sample['box_gp']:
        total = sample['box_ast']
    # Keep credited assists for all games outside the verified tracking sample.
    total = max(total, sample['tracked_ast'])
    return (total + sample['delta']) / minutes, True
