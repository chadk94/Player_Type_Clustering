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
    """Require complete, fresh team coverage before changing that team's minutes.

    Tabs controls explicit availability conflicts. Other active players start at
    the arithmetic mean; the rotation is scaled to 240 after availability changes.
    Source-only scenarios retain the two complete source rotations separately.
    """
    metadata = base[['PLAYER_ID', 'Team', 'Projected minutes']].copy()
    metadata = metadata.rename(columns={'Projected minutes': 'Your minutes'})
    metadata['Tabs minutes'] = np.nan
    metadata['Minutes blend status'] = 'Pending — fresh TabsNBA rotation unavailable'
    metadata['Tabs source time'] = ''
    plans = dict(blended={}, yours={}, tabs={})
    for team, rows in base.groupby('Team', sort=False):
        positions = metadata.Team.eq(team)
        own = rows.set_index('PLAYER_ID')['Projected minutes'].astype(float)
        if own.index.duplicated().any() or not np.isfinite(own).all() or not own.between(0, 48).all() or abs(own.sum()-240) > 1e-6:
            metadata.loc[positions, 'Minutes blend status'] = 'Pending — your full 240-minute rotation is required'
            continue
        opponent = game['Away'] if team == game['Home'] else game['Home']
        ref_values, stamps = {}, []
        for player in rows.itertuples():
            ref = minute_reference(references, pd.Series(dict(PLAYER_ID=player.PLAYER_ID, Team=team, Opponent=opponent)), game, now)
            if ref is not None and ref.source == 'TabsNBA':
                ref_values[int(player.PLAYER_ID)] = float(ref.minutes)
                stamps.append(str(ref.get('source_updated_at', ref.updated_at)))
        metadata.loc[positions, 'Tabs minutes'] = metadata.loc[positions, 'PLAYER_ID'].map(ref_values)
        if not set(own[own.gt(0)].index).issubset(ref_values):
            continue  # Missing players are not implicit OUT decisions.
        if references is None or team in references.attrs.get('incomplete_teams', []):
            metadata.loc[positions, 'Minutes blend status'] = 'Pending — TabsNBA roster coverage is incomplete'
            continue
        # Missing source rows are allowed only for players already at zero.
        tabs = pd.Series(ref_values, dtype=float).reindex(own.index, fill_value=0.)
        if abs(tabs.sum()-240) > 2.:
            metadata.loc[positions, 'Minutes blend status'] = f'Pending — TabsNBA rotation totals {tabs.sum():g}, expected 240'
            continue
        try:
            tab_rotation = rotation_240(tabs)
            average = (own + tabs) / 2
            conflict = own.eq(0) | tabs.eq(0)
            average.loc[conflict] = tabs.loc[conflict]
            blended = rotation_240(average)
        except ValueError:
            continue
        for key, values in [('blended', blended), ('yours', own), ('tabs', tab_rotation)]:
            plans[key].update(values.to_dict())
        metadata.loc[positions, 'Minutes blend status'] = '50/50 your + TabsNBA; Tabs availability; 240-minute rotation'
        metadata.loc[positions, 'Tabs source time'] = min(stamps) if stamps else ''
    return plans, metadata
