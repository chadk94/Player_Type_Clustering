"""Archetype-versus-opponent per-minute adjustments, with explicit sample counts."""
import json
import numpy as np
import pandas as pd
from live_projections import STAT_NAMES, name_key
from season_pipeline import ROOT, apply_overrides, load_rules
from archetype_model import assign_named_roles, validate_named_roles

OFF_STATS = ['PTS','AST','OREB','FGM','FGA','FG3M','FG3A','FTM','FTA','TOV']
DEF_STATS = ['DREB','STL','BLK','PF']
HEAT_STATS = ['PTS','AST','DREB','OREB','FG3M','FG3A','STL','BLK']
START_GAME = 5


def assign_game_archetypes(logs, roster):
    """Current approved labels win; archived labels cover former NBA players."""
    rules = load_rules()
    features = json.loads((ROOT / 'clustering_features.json').read_text())
    # Former/low-minute players also need semantic roles before entering matchup
    # samples. Raw archived unnamed IDs must not form separate hidden groups.
    archives = []
    for side, filename in [('off','player_clusters_detailed.csv'),('def','def_player_clusters_detailed.csv')]:
        reference = pd.read_csv(ROOT / 'references' / filename)
        assigned, _ = assign_named_roles(reference, reference, features[f'cluster_players_{side}'],
                                         side, rules, apply_overrides, prior_mode=True)
        archives.append(assigned[['PLAYER_ID','PLAYER_NAME','Cluster']].rename(columns={'Cluster':f'{side.title()}Cluster'}))
    archived = archives[0].merge(archives[1][['PLAYER_ID','DefCluster']],on='PLAYER_ID',validate='one_to_one')
    # A low-minute player may lack a roster assignment but have real historical
    # features. Use that player's reviewed/named historical role for samples.
    labels = roster[['PLAYER_ID','PLAYER_NAME','OffCluster','DefCluster']].copy()
    historical = archived.set_index('PLAYER_ID')
    for kind in ['Off','Def']:
        labels[f'{kind}Cluster'] = labels[f'{kind}Cluster'].fillna(labels.PLAYER_ID.map(historical[f'{kind}Cluster']))
    archived = archived[~archived.PLAYER_ID.isin(roster.PLAYER_ID)]
    labels = pd.concat([labels,archived],ignore_index=True)
    for kind in ['Off','Def']:
        final = apply_overrides(labels.rename(columns={f'{kind}Cluster':'Cluster'}),kind.lower(),rules)
        validate_named_roles(final,kind.lower(),rules)
        labels[f'{kind}Cluster'] = final.Cluster
    keys = []
    for row in labels.itertuples():
        keys.append({'_key':name_key(row.PLAYER_NAME),'OffCluster':row.OffCluster,'DefCluster':row.DefCluster})
    for row in roster.itertuples():
        if hasattr(row,'WorkbookPlayerName'):
            finalized = labels.loc[labels.PLAYER_ID.eq(row.PLAYER_ID)].iloc[0]
            keys.append({'_key':name_key(row.WorkbookPlayerName),'OffCluster':finalized.OffCluster,'DefCluster':finalized.DefCluster})
    lookup = pd.DataFrame(keys).drop_duplicates()
    lookup = lookup[~lookup._key.duplicated(keep=False)]
    frame = logs.copy()
    frame['_key'] = frame.PLAYER_NAME.map(name_key)
    return frame.merge(lookup,on='_key',how='left',validate='many_to_one').drop(columns='_key')


def matchup_effects(logs):
    """Mean player-game per-36 deviations, shrunk toward neutral by three samples.

    Reproduces the prior app's half-life policy. Counts are player-games, not games.
    Missing or zero cluster baselines produce unavailable effects, never invented zeroes.
    """
    effects = {}
    valid = logs[logs.MIN.ge(5)].sort_values(['GAME_DATE','GAME_ID','ESPN_PLAYER_ID'])
    for kind,stats in [('Off',OFF_STATS),('Def',DEF_STATS)]:
        for cluster, group in valid.dropna(subset=[f'{kind}Cluster']).groupby(f'{kind}Cluster'):
            rates = group.reindex(columns=stats).div(group.MIN,axis=0)*36
            means = rates.mean()
            for opponent, opponent_rows in group.groupby('Opponent'):
                result = {'samples':len(opponent_rows)}
                for stat in stats:
                    values = rates.loc[opponent_rows.index,stat]
                    if not np.isfinite(means[stat]) or means[stat] <= 0:
                        result[stat] = np.nan
                        continue
                    differences = ((values / means[stat] - 1)*100).dropna().to_numpy()
                    if not len(differences):
                        result[stat] = np.nan
                        continue
                    half_life = 15 if stat == 'AST' else 10 if stat in ['OREB','DREB','STL','BLK','PF'] else 999
                    weights = 0.5 ** (np.arange(len(differences)-1,-1,-1)/half_life)
                    result[stat] = float(np.dot(differences,weights)/(3+weights.sum()))
                if kind == 'Off':
                    result['FG3M'] = result['FG3A']  # Preserve player's own 3P conversion rate.
                effects[(kind,int(cluster),opponent)] = result
    return effects


def apply_matchup(baselines, roster, teams, prior_effects, current_effects, opponent=None):
    """Opponent defensive sample switches after four completed opponent games."""
    clusters = roster.set_index('PLAYER_ID')
    counts = teams.set_index('TEAM_ABBREVIATION').GAMES_PLAYED.to_dict() if 'TEAM_ABBREVIATION' in teams else {}
    output = []
    for _,base in baselines.iterrows():
        row = base.to_dict()
        target = opponent or row.get('Opponent','')
        row['Opponent'] = target
        current = counts.get(target,0) >= START_GAME-1
        effects = current_effects if current else prior_effects
        row['Matchup data'] = '2026–27' if current else '2025–26'
        player = clusters.loc[row['PLAYER_ID']]
        missing_role = False
        for kind,stats in [('Off',OFF_STATS),('Def',DEF_STATS)]:
            cluster = player[f'{kind}Cluster']
            unclassified = pd.isna(cluster)
            missing_role = missing_role or unclassified
            effect = effects.get((kind,int(cluster),target),{}) if pd.notna(cluster) else {}
            row[f'{kind} samples'] = effect.get('samples',0)
            for stat in stats:
                pct = effect.get(stat,np.nan)
                row[f'{stat} %'] = pct
                row[f'Baseline {stat}'] = base.get(stat,np.nan)
                base_value = base.get(stat,np.nan)
                if unclassified:
                    # Keep real baseline production in team shares without
                    # inventing an archetype, matchup effect or sample count.
                    row[stat] = max(0,base_value) if np.isfinite(base_value) else np.nan
                else:
                    row[stat] = max(0,base_value*(1+pct/100)) if np.isfinite(pct) and np.isfinite(base_value) else np.nan
        row['Baseline REB'] = base.get('REB',np.nan)
        row['REB'] = row['OREB'] + row['DREB']
        row['PRA'] = row['PTS'] + row['REB'] + row['AST']
        row['Matchup status'] = ('Baseline fallback — missing archetype' if missing_role else
            'Available' if row['Off samples'] and row['Def samples'] else 'Insufficient archetype matchup history')
        output.append(row)
    return pd.DataFrame(output)


def prior_season_profiles(logs, roster):
    """Full previous regular-season per-game averages, matched without fuzzy guesses."""
    from live_projections import match_player_stats
    grouped = logs.groupby(['ESPN_PLAYER_ID','PLAYER_NAME'],as_index=False)
    profiles = grouped[['MIN',*STAT_NAMES]].mean()
    profiles['GP'] = grouped.size()['size']
    matched = match_player_stats(roster,profiles)
    return pd.DataFrame([{'PLAYER_ID':player_id,'MIN':row.MIN,
                          **{s:row[s] for s in STAT_NAMES},'BaselineLabel':'2025–26 full regular season'}
                         for player_id,row in matched.items()],columns=['PLAYER_ID','MIN',*STAT_NAMES,'BaselineLabel'])
