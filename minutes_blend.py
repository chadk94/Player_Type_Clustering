"""Session-only 50/50 rotation plans; never overwrite either source."""
import numpy as np
import pandas as pd

from edge_review import minute_reference


def rotation_240(values):
    """Scale active weights to 240, retaining zeros and the 48-minute cap."""
    values = values.astype(float).copy()
    if values.index.duplicated().any() or not np.isfinite(values).all() or not values.between(0, 48).all():
        raise ValueError('Invalid rotation minutes.')
    weights = values[values.gt(0)].copy()
    if len(weights) < 5:
        raise ValueError('A full active rotation is required.')
    result = values * 0
    remaining = 240.
    while len(weights):
        allocation = remaining * weights / weights.sum()
        capped = allocation.ge(48)
        if not capped.any():
            result.loc[weights.index] = allocation
            remaining = 0.
            break
        result.loc[weights.index[capped]] = 48.
        remaining -= 48 * int(capped.sum())
        weights = weights.loc[~capped]
    if abs(remaining) > 1e-6 or abs(result.sum() - 240) > 1e-6:
        raise ValueError('No feasible full rotation.')
    return result


def minutes_plan(base, references, game, now):
    """Require complete team coverage; retain explicitly saved browser snapshots.

    Only an explicit Tabs OUT status overrides the arithmetic mean. An omitted
    player has a zero Tabs allocation only in a fully matched browser rotation
    totaling 240, fresh when captured; omissions are not injury designations.
    Source-only scenarios retain the two complete source rotations separately.
    """
    metadata = base[['PLAYER_ID', 'Team', 'Projected minutes']].copy()
    metadata = metadata.rename(columns={'Projected minutes': 'Your minutes'})
    metadata['Tabs minutes'] = np.nan
    metadata['Tabs minutes basis'] = 'Unavailable'
    metadata['Minutes blend status'] = 'Pending — fresh TabsNBA rotation unavailable'
    metadata['Tabs source time'] = ''
    metadata['Tabs minutes saved'] = False
    plans = dict(blended={}, yours={}, tabs={})
    for team, rows in base.groupby('Team', sort=False):
        positions = metadata.Team.eq(team)
        own = rows.set_index('PLAYER_ID')['Projected minutes'].astype(float)
        if own.index.duplicated().any() or not np.isfinite(own).all() or not own.between(0, 48).all() or abs(own.sum()-240) > 1e-6:
            metadata.loc[positions, 'Minutes blend status'] = 'Pending — your full 240-minute rotation is required'
            continue
        opponent = game['Away'] if team == game['Home'] else game['Home']
        ref_values, stamps, statuses, saved = {}, [], {}, False
        for player in rows.itertuples():
            ref = minute_reference(references, pd.Series(dict(PLAYER_ID=player.PLAYER_ID, Team=team, Opponent=opponent)), game, now, allow_saved=True)
            if ref is not None and ref.source == 'TabsNBA':
                ref_values[int(player.PLAYER_ID)] = float(ref.minutes)
                statuses[int(player.PLAYER_ID)] = ref.get('source_status','Unknown')
                stamps.append(str(ref.get('source_updated_at', ref.updated_at)))
                saved = saved or bool(ref.get('source_saved',False))
        metadata.loc[positions, 'Tabs minutes'] = metadata.loc[positions, 'PLAYER_ID'].map(ref_values)
        metadata.loc[positions, 'Tabs minutes basis'] = metadata.loc[positions, 'PLAYER_ID'].map(
            {pid:'Explicit OUT' if status=='Out' else 'Published minutes' for pid,status in statuses.items()}).fillna('Unavailable')
        if references is None or team in references.attrs.get('incomplete_teams', []):
            gaps = [] if references is None else [
                f"{row['Player']} ({row['Minutes']:g} min)"
                for row in references.attrs.get('unmatched_players', [])
                if row['Team'] == team and row['Opponent'] == opponent and row['Minutes'] > 0]
            metadata.loc[positions, 'Minutes blend status'] = (
                'Using your minutes — TabsNBA roster profile missing: ' + ', '.join(gaps)
                if gaps else 'Pending — TabsNBA roster coverage is incomplete')
            continue
        missing = set(own[own.gt(0)].index) - set(ref_values)
        inferred = set()
        if missing:
            day = pd.Timestamp(game['GameTimeUTC']).tz_convert('America/New_York').date().isoformat()
            supplied = references.loc[references.Team.eq(team) & references.Opponent.eq(opponent) & references.game_date.eq(day)]
            complete_browser = (not supplied.empty and
                supplied.get('source_mode', pd.Series(index=supplied.index,dtype=str)).eq('Browser comparison').all() and
                supplied.get('source_status', pd.Series(index=supplied.index,dtype=str)).isin(['Projected','Out']).all() and
                set(supplied.PLAYER_ID).issubset(ref_values) and
                abs(sum(ref_values.values())-240) <= 1e-6)
            # A stale or rejected row is not an omitted row. Never fill it with zero.
            if not complete_browser or missing.intersection(supplied.PLAYER_ID):
                names = rows.loc[rows.PLAYER_ID.isin(missing), 'Player'] if 'Player' in rows else sorted(missing)
                metadata.loc[positions, 'Minutes blend status'] = 'Pending — Tabs minutes unavailable for ' + ', '.join(map(str,names))
                continue
            inferred = missing
            ref_values.update({pid:0. for pid in inferred})
            selected = positions & metadata.PLAYER_ID.isin(inferred)
            metadata.loc[selected, 'Tabs minutes'] = 0.
            metadata.loc[selected, 'Tabs minutes basis'] = 'Assumed 0 — not listed in complete 240-minute Tabs rotation'
        # Missing source rows are allowed only for players already at zero.
        tabs = pd.Series(ref_values, dtype=float).reindex(own.index, fill_value=0.)
        if abs(tabs.sum()-240) > 2.:
            metadata.loc[positions, 'Minutes blend status'] = f'Pending — TabsNBA rotation totals {tabs.sum():g}, expected 240'
            continue
        try:
            tab_rotation = rotation_240(tabs)
            average = (own + tabs) / 2
            explicit_out = [pid for pid,status in statuses.items() if status=='Out']
            average.loc[explicit_out] = 0.
            blended = rotation_240(average)
        except ValueError:
            continue
        for key, values in [('blended', blended), ('yours', own), ('tabs', tab_rotation)]:
            plans[key].update(values.to_dict())
        metadata.loc[positions, 'Minutes blend status'] = '50/50 your + TabsNBA; explicit OUT stays out; 240-minute rotation'
        metadata.loc[positions, 'Tabs minutes saved'] = saved
        if saved:
            metadata.loc[positions, 'Minutes blend status'] += '; using saved Tabs minutes — live check pending'
        metadata.loc[positions, 'Tabs source time'] = min(stamps) if stamps else ''
    return plans, metadata
