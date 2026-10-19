"""Reproducible audit for the 100 players flagged before the semantic-role repair."""
import hashlib
import json
from pathlib import Path
import pandas as pd
from archetype_model import load_reviews, named_roles
from season_pipeline import ROOT, load_rules


def build_audit():
    before = pd.read_csv(ROOT / 'data/audit/affected_players_before.csv').set_index('PLAYER_ID')
    reviews = load_reviews().set_index('PLAYER_ID')
    roster = pd.read_csv(ROOT / 'preseason_archetypes.csv').set_index('PLAYER_ID')
    features = pd.read_csv(ROOT / 'references/player_clusters_detailed.csv').set_index('PLAYER_ID')
    rules = load_rules()
    if len(before) != 100 or not set(before.index).issubset(reviews.index):
        raise ValueError('Review does not cover the original 100-player scope exactly')
    records = []
    for player_id,old in before.iterrows():
        new, review, profile = roster.loc[player_id],reviews.loc[player_id],features.loc[player_id]
        record = {'PLAYER_ID':player_id,'Player':new.PLAYER_NAME,'Team':new.TEAM_ABBREVIATION,
                  'Confidence':review.Confidence,'Evidence season':'2025–26 archived partial-season tracking profile',
                  'Archived games':profile.games_played,'Archived MPG':profile.MIN,
                  'Height':profile.HEIGHT,'PTS per 36':profile.PTS_per36,'AST per 36':profile.AST_per36,
                  'OREB per 36':profile.OREB_per36,'DREB per 36':profile.DREB_per36,
                  'STL per 36':profile.STL_per36,'BLK per 36':profile.BLK_per36,
                  'Review date':review.ReviewDate,'Evidence source':review.ReviewSource}
        for kind in ['Off','Def']:
            side=kind.lower();override=rules[f'{side.upper()}_CLUSTER_OVERRIDES'].get(new.PLAYER_NAME)
            expected = override if override is not None else review[f'{kind}Cluster']
            if new[f'{kind}Cluster'] != expected or expected not in named_roles(rules,side):
                raise ValueError(f'Lost reviewed role or override for {new.PLAYER_NAME}')
            label = 'Offense' if side=='off' else 'Defense'
            record[f'Previous {label.lower()}'] = old[f'{kind}Archetype']
            record[label] = new[f'{kind}Archetype']
            record[f'{label} reason'] = review[f'{kind}Reason']
            record[f'{label} source'] = new[f'{kind}AssignmentSource']
        records.append(record)
    report = pd.DataFrame(records)
    first = ['Player','Team','Previous offense','Offense','Offense reason','Offense source',
             'Previous defense','Defense','Defense reason','Defense source','Confidence']
    report = report[first+[c for c in report if c not in first]]
    destination = ROOT / 'data/audit'
    report.to_csv(destination / 'reviewed_100_players.csv',index=False)
    # Keep the full repair reproducible without depending on Git availability.
    original = pd.read_csv(destination / 'roster_before.csv').set_index('PLAYER_ID')
    changed = roster.OffCluster.fillna(-1).ne(original.OffCluster.fillna(-1)) | roster.DefCluster.fillna(-1).ne(original.DefCluster.fillna(-1))
    changes = []
    for player_id in roster.index[changed]:
        old,new = original.loc[player_id],roster.loc[player_id]
        changes.append({'PLAYER_ID':player_id,'Player':new.PLAYER_NAME,'Team':new.TEAM_ABBREVIATION,
            'In original 100':player_id in before.index,'Previous offense':old.OffArchetype,'Offense':new.OffArchetype,
            'Previous defense':old.DefArchetype,'Defense':new.DefArchetype,
            'Offense source':new.OffAssignmentSource,'Defense source':new.DefAssignmentSource})
    pd.DataFrame(changes).to_csv(destination / 'all_assignment_changes.csv',index=False)
    extra = pd.read_csv(ROOT/'archetype_reference_reviews.csv').set_index('PLAYER_ID').join(roster[['TEAM_ABBREVIATION','OffArchetype','DefArchetype','OffAssignmentSource','DefAssignmentSource']]).join(features[['games_played','MIN','PTS_per36','AST_per36','STL_per36','BLK_per36']])
    extra.to_csv(destination / 'additional_90_players.csv')
    rotation = pd.read_csv(ROOT/'archetype_rotation_reviews.csv').set_index('PLAYER_ID')
    rotation = rotation.join(roster[['TEAM_ABBREVIATION','PROJECTED_MIN','OffArchetype','DefArchetype','OffAssignmentSource','DefAssignmentSource','HasHistoricalProfile']])
    rotation.to_csv(destination/'low_minute_33_players.csv')
    meta = {'reviewed_players':len(report),'missing_named_roles':0,
            'manual_override_counts':{side:len(rules[f'{side}_CLUSTER_OVERRIDES']) for side in ['OFF','DEF']},
            'manual_override_file_sha256':hashlib.sha256((ROOT/'archetype_overrides.json').read_bytes()).hexdigest(),
            'additional_reference_reviews':len(pd.read_csv(ROOT/'archetype_reference_reviews.csv')),
            'low_minute_reviews':len(pd.read_csv(ROOT/'archetype_rotation_reviews.csv')),
            'reference_review_file_sha256':hashlib.sha256((ROOT/'archetype_reference_reviews.csv').read_bytes()).hexdigest(),
            'review_file_sha256':hashlib.sha256((ROOT/'archetype_reviews.csv').read_bytes()).hexdigest(),
            'role_policy':'User manual override > analyst review > fixed named historical role/centroid',
            'notes':'Analyst judgments within existing role vocabulary, not new observed 2026–27 data. Developing/small-sample roles remain provisional. Offense Spot-Up Role Player is the legacy low-usage complementary-role bucket, including cutters.'}
    meta['total_players_with_changed_roles'] = len(changes)
    meta['additional_players_with_changed_roles'] = sum(not row['In original 100'] for row in changes)
    meta['unchanged_workbook_minutes'] = roster.PROJECTED_MIN.equals(original.PROJECTED_MIN)
    meta['named_eligible_players'] = int((roster.ModelEligibleMinutes & roster.OffArchetype.notna() & roster.DefArchetype.notna()).sum())
    meta['role_policy_version'] = hashlib.sha256(b''.join((ROOT/p).read_bytes() for p in ['archetype_reviews.csv','archetype_reference_reviews.csv','archetype_rotation_reviews.csv','archetype_overrides.json','archetype_model.py'])).hexdigest()
    (destination/'review_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    return report


if __name__ == '__main__':
    print(f'Audited {len(build_audit())} players')
