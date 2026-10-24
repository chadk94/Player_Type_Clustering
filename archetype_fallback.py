"""Explicit temporary stat estimates for players without an individual profile."""
import numpy as np
import pandas as pd

ARCHETYPE_BASELINE = 'Temporary archetype-average profile'
OFF_STATS = ('PTS', 'AST', 'OREB', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'TOV')
DEF_STATS = ('DREB', 'STL', 'BLK', 'PF')


def build_archetype_priors(roster, history, min_donors=3):
    """Equal-player mean per-minute rates from observed, matching-role profiles.

    Build once with the full league roster, then reuse for game/player slices.
    Require a complete nonnegative vector and at least three historical donors
    on each side. Never use forecasts, synthetic profiles, or the recipient as
    donors. The returned estimates are separate from saved historical inputs.
    """
    if min_donors < 3:
        raise ValueError('Archetype estimates require at least three historical donors')
    profiles = history.copy()
    if profiles.PLAYER_ID.duplicated().any() or roster.PLAYER_ID.duplicated().any():
        raise ValueError('Archetype estimates require unique player IDs')
    minutes = pd.to_numeric(profiles.MIN, errors='coerce')
    usable = np.isfinite(minutes) & minutes.gt(0)
    own_profile_ids = set(profiles.loc[usable, 'PLAYER_ID'])
    # Forecasts never become evidence for another forecast.
    labels = profiles.get('BaselineLabel', pd.Series('', index=profiles.index)).fillna('')
    observed = ~labels.str.contains('rotowire|archetype|projection|forecast', case=False, regex=True)
    donors = profiles.loc[usable & observed].copy()
    donors['MIN'] = minutes.loc[donors.index]
    donors = donors.merge(roster[['PLAYER_ID', 'OffArchetype', 'DefArchetype']], on='PLAYER_ID', validate='one_to_one')
    groups = {}
    for side, stats in [('Off', OFF_STATS), ('Def', DEF_STATS)]:
        rates = donors.reindex(columns=stats).apply(pd.to_numeric, errors='coerce').div(donors.MIN, axis=0)
        valid = np.isfinite(rates).all(axis=1) & rates.ge(0).all(axis=1)
        rates = rates.loc[valid].assign(role=donors.loc[valid, f'{side}Archetype'])
        groups[side] = {
            role: (frame[list(stats)].mean().to_dict(), len(frame))
            for role, frame in rates.groupby('role')
            if len(frame) >= min_donors and str(role).strip().lower() not in ('', 'unclassified', 'uncategorized')
        }
    output = {}
    for player in roster.itertuples():
        if player.PLAYER_ID in own_profile_ids:
            continue
        offense = groups['Off'].get(player.OffArchetype)
        defense = groups['Def'].get(player.DefArchetype)
        if offense is None or defense is None:
            continue
        off, off_count = offense
        deff, def_count = defense
        output[player.PLAYER_ID] = {
            'MIN': 1., **off, **deff, 'REB': off['OREB'] + deff['DREB'],
            'BaselineLabel': ARCHETYPE_BASELINE,
            'Fallback details': (f'{player.OffArchetype}: {off_count} offensive donors; '
                                 f'{player.DefArchetype}: {def_count} defensive donors. '
                                 'Equal-player mean historical per-minute rates.'),
        }
    return output
