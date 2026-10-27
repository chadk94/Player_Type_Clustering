"""Stable named archetype assignment, with reviewed roles below user overrides.

KMeans numbers describe a particular fit, not basketball roles. Never force a new
fit onto old numbers by a one-to-one permutation. Use fixed historical references
for preseason and fixed named centroids for genuinely new feature profiles.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent


def named_roles(rules, side):
    return {int(key):name for key,name in rules[f'{side.upper()}_CLUSTER_NAMES'].items()
            if name and not any(word in name.lower() for word in ['uncategorized','unclassified','unnamed'])}


def load_reviews():
    reviews = pd.concat([pd.read_csv(ROOT / filename) for filename in ['archetype_reviews.csv','archetype_reference_reviews.csv','archetype_rotation_reviews.csv','archetype_league_reviews.csv']],ignore_index=True)
    required = ['PLAYER_ID','PLAYER_NAME','OffCluster','DefCluster','OffReason','DefReason']
    if not set(required).issubset(reviews) or reviews.PLAYER_ID.duplicated().any() or reviews.PLAYER_NAME.duplicated().any():
        raise ValueError('Role reviews require unique player IDs/names and per-side reasons')
    if reviews[required].isna().any().any():
        raise ValueError('Incomplete role review')
    return reviews


def apply_reviews(frame, side, rules):
    frame = frame.copy()
    reviews = load_reviews()
    column = f'{side.title()}Cluster'
    if not reviews[column].isin(named_roles(rules,side)).all():
        raise ValueError(f'Review contains unnamed {side} roles')
    key = 'PLAYER_ID' if 'PLAYER_ID' in frame else 'PLAYER_NAME'
    values = frame[key].map(reviews.set_index(key)[column])
    mask = values.notna()
    if 'AssignmentSource' not in frame:
        frame['AssignmentSource'] = 'auto_model'
    frame.loc[mask,'Cluster'] = values[mask]
    frame.loc[mask,'AssignmentSource'] = 'analyst_review'
    return frame


def validate_named_roles(frame, side, rules):
    assigned = frame.Cluster.dropna()
    if not assigned.isin(named_roles(rules,side)).all():
        raise ValueError(f'Unclassified/unnamed {side} archetypes cannot be published')


def physically_eligible(frame, side, roles):
    """Automatic big-role guardrail; explicit human rules are applied afterwards."""
    allowed = np.ones((len(frame),len(roles)),dtype=bool)
    heights = frame.Height_IN.to_numpy(dtype=float) if 'Height_IN' in frame else None
    if heights is not None:
        bigs = {1,7,8,9} if side == 'off' else {1,3,7}
        for i,role in enumerate(roles):
            if role in bigs:
                allowed[:,i] = heights >= 80  # 6'8"; humans may override exceptional roles.
    return allowed


def assign_named_roles(frame, reference, features, side, rules, apply_rules, prior_mode=False):
    valid = named_roles(rules,side)
    anchors = apply_rules(reference.copy(),side,rules)
    # Tiny/zero-minute samples should not define an archetype centroid.
    training = anchors[anchors.Cluster.isin(valid) & anchors.MIN.ge(10)].copy()
    if 'games_played' in training:
        training = training[training.games_played.ge(5)]
    if training.empty or not np.isfinite(training[features].to_numpy(dtype=float)).all():
        raise ValueError(f'No finite named {side} reference profiles')
    roles = sorted(int(x) for x in training.Cluster.unique())
    if set(roles) != set(valid):
        raise ValueError(f'Missing named {side} centroid anchors: {set(valid)-set(roles)}')
    scaler = StandardScaler().fit(training[features])
    prototypes = training.groupby('Cluster')[features].mean().reindex(roles)
    transformed = scaler.transform(frame[features])
    centers = scaler.transform(prototypes)
    distances = ((transformed[:,None,:]-centers[None,:,:])**2).sum(axis=2)
    eligible = physically_eligible(frame,side,roles)
    distances[~eligible] = np.inf
    if not np.isfinite(distances).any(axis=1).all():
        raise ValueError('No eligible named role for a feature profile')
    chosen = np.argmin(distances,axis=1)
    result = frame.copy()
    result['Cluster'] = np.array(roles)[chosen]
    result['AssignmentSource'] = 'named_reference_centroid'
    if prior_mode:
        # Same prior features should retain their historical semantic role. A new
        # roster/minutes cutoff cannot move a player into an unrelated category.
        historical = frame.PLAYER_ID.map(anchors.set_index('PLAYER_ID').Cluster)
        for i,role in enumerate(roles):
            mask = historical.eq(role).to_numpy() & eligible[:,i]
            result.loc[mask,'Cluster'] = role
            result.loc[mask,'AssignmentSource'] = 'historical_named_role'
    result = apply_rules(result,side,rules)
    validate_named_roles(result,side,rules)
    metadata = {'method':'historical_named_roles' if prior_mode else 'fixed_named_reference_centroids',
                'reference_players':len(training), 'role_counts':{str(int(k)):int(v) for k,v in training.groupby('Cluster').size().items()},
                'eligible_role_ids':roles,'forced_cluster_permutation':False,
                'precedence':['user_manual_override','analyst_review','historical_named_role' if prior_mode else 'named_reference_centroid']}
    return result, metadata
