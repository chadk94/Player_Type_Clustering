"""Read-only fit screen: candidates for human review, never new assignments."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from archetype_model import named_roles, physically_eligible
from season_pipeline import ROOT, apply_overrides, load_rules


def audit_role_fit():
    rules = load_rules()
    roster = pd.read_csv(ROOT/'preseason_archetypes.csv')
    roster = roster[roster.PROJECTED_MIN.gt(0)]
    feature_sets = json.loads((ROOT/'clustering_features.json').read_text())
    rows = []
    for side, filename in [('off','player_clusters_detailed.csv'),('def','def_player_clusters_detailed.csv')]:
        prefix = side.title()
        names = named_roles(rules,side)
        features = feature_sets[f'cluster_players_{side}']
        reference = pd.read_csv(ROOT/'references'/filename)
        anchors = apply_overrides(reference,side,rules)
        training = anchors[anchors.Cluster.isin(names) & anchors.MIN.ge(10) & anchors.games_played.ge(5)]
        roles = sorted(int(x) for x in training.Cluster.unique())
        scaler = StandardScaler().fit(training[features])
        centers = scaler.transform(training.groupby('Cluster')[features].mean().reindex(roles))
        profiles = reference.set_index('PLAYER_ID')
        # Outlier cutoff is the 95th percentile of reference profiles' nearest RMS distance.
        train_dist = np.sqrt(((scaler.transform(training[features])[:,None,:]-centers[None,:,:])**2).mean(axis=2))
        train_dist[~physically_eligible(training,side,roles)] = np.inf
        outlier_cutoff = np.quantile(train_dist.min(axis=1),.95)
        for player in roster.itertuples():
            current_id = getattr(player,f'{prefix}Cluster')
            current = getattr(player,f'{prefix}Archetype')
            manual = player.PLAYER_NAME in rules[f'{side.upper()}_CLUSTER_OVERRIDES']
            row = {'PLAYER_ID':player.PLAYER_ID,'Player':player.PLAYER_NAME,'Team':player.TEAM_ABBREVIATION,
                   'Side':'Offense' if side=='off' else 'Defense','Current':current,
                   'Assignment source':getattr(player,f'{prefix}AssignmentSource'),'Protected manual override':manual,
                   'Proposed':current,'Action':'Keep manual override' if manual else 'Keep current; review only',
                   'Reason':'','Nearest candidate':None,'Runner-up':None,'Margin %':np.nan,
                   'Assigned / nearest distance':np.nan,'Nearest RMS distance':np.nan,
                   'Outlier cutoff':round(outlier_cutoff,3),'Archived GP':np.nan,'Archived MPG':np.nan}
            if player.PLAYER_ID not in profiles.index:
                row['Reason']='No archived tracking profile; style-based assignment needs human review'
                rows.append(row)
                continue
            profile = profiles.loc[player.PLAYER_ID]
            row.update({'Archived GP':profile.games_played,'Archived MPG':round(profile.MIN,2)})
            if not np.isfinite(profile[features].to_numpy(dtype=float)).all():
                row['Reason']='Incomplete archived tracking profile'
                rows.append(row)
                continue
            distances = np.sqrt(((scaler.transform(profile[features].to_frame().T)[0]-centers)**2).mean(axis=1))
            allowed = physically_eligible(profile.to_frame().T,side,roles)[0]
            distances[~allowed] = np.inf
            order = np.argsort(distances)
            best,second = order[:2]
            nearest,runner = roles[best],roles[second]
            margin = (distances[second]-distances[best])/max(distances[second],1e-9)
            ratio = distances[roles.index(int(current_id))]/max(distances[best],1e-9) if pd.notna(current_id) and int(current_id) in roles else np.inf
            reasons = []
            if margin < .10:
                reasons.append('Ambiguous: top two roles within 10%')
            if nearest != current_id and ratio >= 1.35:
                reasons.append('Assigned role at least 35% farther than closest centroid')
            if distances[best] > outlier_cutoff:
                reasons.append('Outlier beyond 95th-percentile reference distance')
            if profile.games_played < 10 or profile.MIN < 10:
                reasons.append('Small sample: fewer than 10 archived games or MPG')
            row.update({'Reason':'; '.join(reasons),'Nearest candidate':names[nearest],
                'Runner-up':names[runner],'Margin %':round(100*margin,1),
                'Assigned / nearest distance':round(ratio,2),'Nearest RMS distance':round(distances[best],3)})
            for field in ['AST_per36','PTS_per36','DRIVESPER36','PASSESMADEPER36','OFF_PRBALLHANDLER_FREQ',
                          'OFF_SPOTUP_FREQ','OFF_CUT_FREQ','CATCHANDSHOOT3PAPER36','STL_per36','BLK_per36','DREB_per36']:
                row[field] = round(float(profile[field]),3)
            rows.append(row)
    report = pd.DataFrame(rows)
    proposals = pd.read_csv(ROOT/'data/audit/role_fit_proposals.csv')
    for proposal in proposals.itertuples(index=False):
        mask = report.PLAYER_ID.eq(proposal.PLAYER_ID) & report.Side.eq(proposal.Side)
        if mask.sum() != 1:
            raise ValueError('Proposal must match exactly one positive-minute role')
        if report.loc[mask,'Protected manual override'].any():
            raise ValueError('Audit proposals cannot alter a protected user override')
        if report.loc[mask,'Current'].iloc[0] != proposal.Current:
            raise ValueError('Stale proposal: current assignment changed; review again')
        valid = named_roles(rules,'off' if proposal.Side == 'Offense' else 'def')
        if proposal.Proposed not in valid.values():
            raise ValueError('Proposal must use a named role')
        report.loc[mask,'Proposed'] = proposal.Proposed
        report.loc[mask,'Action'] = proposal.Decision
        report.loc[mask,'Review rationale'] = proposal.Rationale
        report.loc[mask,'Review confidence'] = proposal.Confidence
    return report


def write_audit(destination):
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    report=audit_role_fit()
    report.to_csv(destination/'all_role_fit_checks.csv',index=False)
    flagged=report[report.Reason.ne('')].copy()
    flagged.to_csv(destination/'flagged_role_fit.csv',index=False)
    manifest={'scope':'Every positive-workbook-minute player; both offense and defense',
        'players_screened':report.PLAYER_ID.nunique(),'sides_screened':len(report),
        'flagged_players':flagged.PLAYER_ID.nunique(),'flagged_sides':len(flagged),
        'protected_flagged_sides':int(flagged['Protected manual override'].sum()),
        'applied_assignment_changes':0,
        'proposed_changes':int(report.Proposed.ne(report.Current).sum()),
        'evidence':'Immutable partial-season 2025–26 tracking features; no new 2026–27 observations',
        'method':'Existing feature sets; standardized fixed named centroids; height guardrail; RMS distances. Distance margins are not calibrated probabilities. Closest centroid is a review candidate, not proof of basketball role.',
        'thresholds':{'ambiguous_margin_below':.10,'assigned_distance_ratio_at_least':1.35,'outlier_reference_quantile':.95,'small_sample_games_or_MPG_below':10},
        'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['archetype_overrides.json','preseason_archetypes.csv','clustering_features.json','references/player_clusters_detailed.csv','references/def_player_clusters_detailed.csv','data/audit/role_fit_proposals.csv']}}
    (destination/'role_fit_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k.endswith(('screened','players','sides','changes'))},indent=2))
    return flagged


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(ROOT/'data/audit/role_fit'))
    write_audit(parser.parse_args().output)
