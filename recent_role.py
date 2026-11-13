"""Optional recent-role comparison. Never writes to the production projection.

Use the ten-appearance shot-volume/assist trial after 20 positive-minute
appearances. Reweight final shares, which algebraically retains the existing
normalized archetype effects and fixes current team PTS/AST/3PA totals.
"""
import numpy as np
import pandas as pd

from live_projections import match_player_stats

HALF_LIFE = 10
MIN_APPEARANCES = 20
RECENT_STATS = {'PTS': 'PTS', 'AST': 'AST', '3PM': 'FG3M', 'PRA': 'PRA'}
DISPLAY_COLUMNS = ['Recent role', 'Recent change %', 'Recent role status']
CAPTION = ('Recent role is experimental: a 10-appearance half-life for shot volume and assists, '
           'starting after 20 positive-minute appearances. Same minutes, current team points/assist/three-attempt '
           'totals, matchup effects and career shooting blend. Threes made can change with the shooter mix. '
           'Recent change % compares the two models, not the sportsbook. Edges, rankings and points '
           'probabilities still use Projected. This trial did not demonstrate better overall accuracy.')
LOG_STATS = ['MIN', 'FGA', 'FGM', 'FG3A', 'FG3M', 'FTA', 'FTM', 'AST']


def _ratios(player, observed, logs, season, as_of):
    """Return opportunity ratios against the same current-season baseline."""
    if player.get('Season appearances', 0) < MIN_APPEARANCES:
        return np.ones(3), 'Current rates — fewer than 20 appearances'
    if observed is None or not set(['ESPN_PLAYER_ID', 'GAME_ID', 'GAME_DATE', *LOG_STATS]).issubset(logs):
        raise ValueError('Current-season game logs unavailable')
    sample = logs[logs.ESPN_PLAYER_ID.astype(str).eq(str(observed.ESPN_PLAYER_ID))].copy()
    dates = pd.to_datetime(sample.GAME_DATE, utc=True, errors='coerce')
    year = int(season[:4])
    start, end = pd.Timestamp(f'{year}-07-01', tz='UTC'), pd.Timestamp(f'{year+1}-07-01', tz='UTC')
    sample = sample.loc[dates.ge(start) & dates.lt(min(end, as_of))].copy()
    sample['_date'] = dates.loc[sample.index]
    if (sample.GAME_ID.duplicated().any() or len(sample) != observed.GP or sample.empty):
        raise ValueError('Incomplete or mismatched current-season game logs')
    values = sample[LOG_STATS].apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(values).all().all() or values.lt(0).any().any():
        raise ValueError('Invalid current-season game logs')
    if (values.FGM.gt(values.FGA).any() or values.FG3M.gt(values.FG3A).any()
            or values.FG3A.gt(values.FGA).any() or values.FG3M.gt(values.FGM).any()
            or values.FTM.gt(values.FTA).any()):
        raise ValueError('Inconsistent shooting counts')
    sample[LOG_STATS] = values
    # Summary and boxes must describe the same sample. Permit display rounding,
    # including ESPN's whole-minute box-score values, but not missing games.
    for stat in LOG_STATS:
        value = observed.get(stat, np.nan)
        tolerance = max(1., observed.GP * .51) if stat == 'MIN' else observed.GP * .051 + 1e-6
        if not np.isfinite(value) or abs(values[stat].sum() - value * observed.GP) > tolerance:
            raise ValueError('Season summary and game logs disagree')
    sample = sample[sample.MIN.gt(0)].sort_values(['_date', 'GAME_ID'])
    if len(sample) < MIN_APPEARANCES:
        return np.ones(3), 'Current rates — fewer than 20 positive-minute appearances'
    if not np.isfinite(observed.MIN) or observed.MIN <= 0:
        raise ValueError('Current-season minutes unavailable')
    weights = .5 ** (np.arange(len(sample)-1, -1, -1) / HALF_LIFE)
    recent = sample[LOG_STATS].multiply(weights, axis=0).sum() / np.dot(weights, sample.MIN)
    current = observed[LOG_STATS].astype(float) / observed.MIN
    accuracy = player.get('Expected 3P%', np.nan)
    if not np.isfinite(accuracy):
        accuracy = current.FG3M/current.FG3A if current.FG3A > 0 else 0.
    if not 0 <= accuracy <= 1:
        raise ValueError('Three-point accuracy unavailable')
    two_a, two_m = current.FGA-current.FG3A, current.FGM-current.FG3M
    two_pct = two_m/two_a if two_a > 0 else 0.
    ft_pct = current.FTM/current.FTA if current.FTA > 0 else 0.
    def opportunity(row):
        return 2*(row.FGA-row.FG3A)*two_pct + 3*row.FG3A*accuracy + row.FTA*ft_pct
    def ratio(new, old):
        if old == 0 and new == 0:
            return 1.
        if not np.isfinite([new, old]).all() or old <= 0 or new < 0:
            raise ValueError('Current-season rate unavailable')
        return new/old
    return np.array([ratio(opportunity(recent), opportunity(current)),
                     ratio(recent.AST, current.AST), ratio(recent.FG3A, current.FG3A)]), '10-appearance half-life'


def recent_role_comparison(final, roster, current, logs, season, as_of=None, *, matched_players=None):
    """Full-roster, copy-only comparison; unavailable comparisons stay explicit.

    All active teammates participate before any player/stat filters. A missing
    eligible teammate sample disables that team's comparison, rather than
    allocating a partial roster. No network, persistence or main model mutation.
    """
    result = final.copy(deep=True)
    for stat in ['PTS', 'AST', 'FG3A', 'FG3M', 'PRA']:
        result[f'Recent {stat}'] = np.nan
    result['Recent role status'] = 'No projected minutes'
    result['Recent 3PM status'] = ''
    if final.empty:
        return result
    as_of = pd.Timestamp(as_of or pd.Timestamp.now(tz='UTC'))
    if as_of.tzinfo is None:
        raise ValueError('Recency snapshot time must have a timezone')
    matches = match_player_stats(roster, current) if matched_players is None else matched_players
    for team, group in final.groupby('Team', sort=False):
        mask = final.Team.eq(team) & final['Projected minutes'].gt(0)
        active = final.loc[mask]
        if active.empty:
            continue
        try:
            if not np.isfinite(group['Projected minutes']).all() or group['Projected minutes'].lt(0).any():
                raise ValueError('Incomplete team minutes')
            needed = active[['PTS', 'AST', 'FG3A', 'FG3M', 'REB']]
            if not np.isfinite(needed).all().all() or needed.lt(0).any().any():
                raise ValueError('Incomplete team projections')
            ratios, statuses = [], []
            for _, player in active.iterrows():
                try:
                    ratio, status = _ratios(player, matches.get(player.PLAYER_ID), logs, season, as_of)
                except (ValueError, KeyError, TypeError, AttributeError) as exc:
                    raise ValueError(f'{player.Player}: {exc}') from exc
                ratios.append(ratio)
                statuses.append(status)
            if all(status.startswith('Current rates') for status in statuses):
                result.loc[mask, 'Recent role status'] = 'Not active yet — requires 20 player appearances'
                continue
            ratios = np.stack(ratios)
            for i, stat in enumerate(['PTS', 'AST', 'FG3A']):
                original = active[stat].to_numpy()
                changed = original * ratios[:, i]
                factor = original.sum()/changed.sum() if changed.sum() > 0 else 1.
                if changed.sum() == 0 and original.sum() > 0:
                    raise ValueError('Recent rates cannot allocate the team total')
                result.loc[mask, f'Recent {stat}'] = changed * factor
                if stat == 'FG3A':
                    result.loc[mask, 'Recent FG3M'] = active.FG3M.to_numpy() * ratios[:, i] * factor
            result.loc[mask, 'Recent PRA'] = result.loc[mask, 'Recent PTS'] + active.REB + result.loc[mask, 'Recent AST']
            result.loc[mask, 'Recent role status'] = [s + ('; team shares rebalanced' if s.startswith('Current rates') else '') for s in statuses]
            if (3*result.loc[mask, 'Recent FG3M'] > result.loc[mask, 'Recent PTS'] + 1e-9).any():
                result.loc[mask, 'Recent FG3M'] = np.nan
                result.loc[mask, 'Recent 3PM status'] = 'Unavailable — recent threes would exceed projected points'
        except (ValueError, KeyError, TypeError) as exc:
            result.loc[mask, [f'Recent {s}' for s in ['PTS', 'AST', 'FG3A', 'FG3M', 'PRA']]] = np.nan
            result.loc[mask, 'Recent role status'] = f'Unavailable — {exc}'
    return result


def add_recent_columns(table, recent, player_id=None):
    """Append display-only fields AFTER existing edge/probability calculations."""
    result = table.copy(deep=True)
    result['Recent role'] = np.nan
    result['Recent change %'] = np.nan
    result['Recent role status'] = 'Unchanged category'
    if player_id is not None:
        lookup = recent[recent.PLAYER_ID.eq(player_id)]
    else:
        lookup = recent
    for index, row in result.iterrows():
        stat = RECENT_STATS.get(row.Stat)
        if stat is None:
            continue
        match = lookup if player_id is not None else lookup[lookup.Player.eq(row.Player) & lookup.Team.eq(row.Team)]
        if len(match) != 1:
            result.loc[index, 'Recent role status'] = 'Unavailable — player match is ambiguous'
            continue
        player = match.iloc[0]
        value = player.get(f'Recent {stat}', np.nan)
        result.loc[index, 'Recent role'] = value
        result.loc[index, 'Recent role status'] = player.get('Recent 3PM status') if stat == 'FG3M' and player.get('Recent 3PM status') else player['Recent role status']
        if np.isfinite(value) and np.isfinite(row.Projected) and row.Projected > 0:
            result.loc[index, 'Recent change %'] = 100*(value/row.Projected-1)
    return result
