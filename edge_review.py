"""Read-only robustness checks. Never replace the model mean, price or probability.

Scenarios retain team targets and normalized archetype effects. Minute stress
redistributes a full 240-minute rotation among existing active teammates. That
redistribution is a sensitivity test, not a prediction of the coach's rotation.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sportsbook_lines import PLAY_THRESHOLD_PCT
from team_threes import exceeds_projected_points
from sportsbook_injuries import pending_key_players

CONFIG = json.loads(Path(__file__).with_suffix('.json').read_text())
FLAG_COLUMNS = ['Minutes flag', 'Role flag', 'Injury flag']
REVIEW_VIEWS = ['All plays','More robust plays','Waiting on news','Needs review / unchecked']
SCENARIO_STATS = {'3PM':'FG3M', **{s:s for s in ['PTS','AST','REB','STL','BLK','PRA']}}


def unavailable_review(edges):
    """Review failures may not take down the existing quote/projection table."""
    out=edges.copy(deep=True)
    for flag in FLAG_COLUMNS:out[flag]='Unchecked'
    for kind in ('Minutes','Role','Injury','Review'):out[kind+' detail']='Review inputs unavailable; current model values retained.'
    for kind in ('Minutes','Recent','Combined'):
        out[kind+' test projection']=np.nan;out[kind+' test edge %']=np.nan
    out['Robust play']=False;out['Review group']='Unchecked';out['Review flags']='? Unchecked'
    out['Reference minutes']=np.nan;out['Minutes difference']=np.nan;out['Reference source']='Unavailable'
    out['Minutes test minutes']=np.nan
    out['Rebound trend %']=np.nan
    return out


def side_edge(projection, implied, side):
    if side not in ('Over','Under') or not np.isfinite([projection,implied]).all() or implied <= 0:
        return np.nan
    return (projection/implied-1)*100*(1 if side=='Over' else -1)


def rebalance_minutes(team, player_id, target):
    """Keep exclusions, preserve 240, and cap every active player at 48."""
    values = team.set_index('PLAYER_ID')['Projected minutes'].astype(float).copy()
    if (values.index.duplicated().any() or not np.isfinite(values).all() or values.lt(0).any()
            or values.gt(48).any() or abs(values.sum()-240)>1e-6):
        raise ValueError('A complete 240-minute rotation is required.')
    if player_id not in values or values[player_id] <= 0 or not np.isfinite(target) or not 0<=target<=48:
        raise ValueError('The changed player needs an active, valid minute allocation.')
    active=values.index[(values>0)&(values.index!=player_id)]
    weights=values.loc[active].copy()
    remaining=240-float(target)
    values.loc[active]=0.
    while len(weights):
        allocation=remaining*weights/weights.sum()
        capped=allocation.ge(48)
        if not capped.any():
            values.loc[weights.index]=allocation
            remaining=0.
            break
        values.loc[weights.index[capped]]=48.
        remaining-=48*int(capped.sum())
        weights=weights.loc[~capped]
    values[player_id]=target
    if abs(remaining)>1e-6 or abs(values.sum()-240)>1e-6:
        raise ValueError('No feasible 240-minute rotation at these minutes.')
    return values.to_dict()


def scenario_projection(team, minutes=None, role=None):
    """Reallocate the existing team's targets with changed opportunity weights.

An incoming zero-minute player requires an actual reprojected baseline; do not
invent one by dividing a zero projection. Injury branches can supply fully
recomputed frames instead (see assess_edges).
"""
    out=team.copy(deep=True)
    original=out['Projected minutes'].to_numpy(float)
    updated=np.array([minutes[int(p)] for p in out.PLAYER_ID],float) if minutes is not None else original.copy()
    if not np.isfinite(updated).all() or (updated<0).any() or (updated>48).any() or abs(updated.sum()-240)>1e-6:
        raise ValueError('Scenario minutes must total 240.')
    if ((original==0)&(updated>0)).any():
        raise ValueError('An incoming player needs a verified per-minute baseline.')
    scale=np.divide(updated,original,out=np.ones_like(updated),where=original>0)
    role=role or {}
    def weights(stat):
        return np.array([role.get(int(p),{}).get(stat,1.) for p in out.PLAYER_ID])
    def allocate(stat,before,target,weight=None):
        target_values=out.get(target,pd.Series(dtype=float)).dropna()
        multiplier=scale*weights(weight or stat)
        if len(target_values):
            if target_values.max()-target_values.min()>1e-6 or before not in out:
                raise ValueError('Team target metadata is inconsistent.')
            raw=out[before].to_numpy(float)*multiplier
            total=float(target_values.iloc[0])
            if not np.isfinite(raw).all() or (raw<0).any() or (raw.sum()==0 and total>0):
                raise ValueError('Incomplete scenario production.')
            return raw*(total/raw.sum()) if raw.sum()>0 else raw
        return out[stat].to_numpy(float)*multiplier
    points=allocate('PTS','Pre-market PTS','Market team total')
    assists=allocate('AST','Pre-team AST','Team AST target')
    if 'Team REB target' in out and out['Team REB target'].notna().any():
        rebounds=allocate('OREB','Pre-team OREB','Team OREB target')+allocate('DREB','Pre-team DREB','Team DREB target')
    else:rebounds=out.REB.to_numpy(float)*scale
    threes=out.FG3M.to_numpy(float)*scale*weights('FG3A')
    targets=out.get('Candidate team 3PA',pd.Series(dtype=float)).dropna()
    if len(targets):
        attempts=out['Pre-team 3PA'].to_numpy(float)*scale*weights('FG3A')
        raw=out['Pre-team 3PM'].to_numpy(float)*scale*weights('FG3A')
        if not np.isfinite(attempts).all() or attempts.sum()<=0:
            raise ValueError('Incomplete three-point attempt allocation.')
        threes=raw*float(targets.iloc[0])/attempts.sum()
        if exceeds_projected_points(threes,points).any():threes=raw
    out['Projected minutes']=updated
    for stat,values in dict(PTS=points,AST=assists,REB=rebounds,FG3M=threes,
                           PRA=points+assists+rebounds,STL=out.STL.to_numpy(float)*scale,
                           BLK=out.BLK.to_numpy(float)*scale).items():out[stat]=values
    return out


def passing_role_evidence(sample, player_id, n, passing):
    """Keep the existing 50/50 opportunity check and report its exact coverage."""
    prior_start=max(0,len(sample)-n-CONFIG['previous_games'])
    previous=sample.iloc[prior_start:-n]
    result=dict(ratios={},trends={})
    track=(passing.loc[passing.PLAYER_ID.eq(player_id)].reindex(columns=['GAME_ID','POTENTIAL_AST','PASSES_MADE']).copy()
           if passing is not None and not passing.empty else pd.DataFrame(columns=['GAME_ID','POTENTIAL_AST','PASSES_MADE']))
    track['GAME_ID']=track.GAME_ID.astype(str)
    if track.GAME_ID.duplicated().any():
        return dict(result,passing_detail='Duplicate passing records; the passing-role check is pending.')
    merged=sample.assign(GAME_ID=sample.GAME_ID.astype(str)).merge(track,on='GAME_ID',how='left',validate='one_to_one')
    counts=merged[['POTENTIAL_AST','PASSES_MADE']].apply(pd.to_numeric,errors='coerce')
    verified=np.isfinite(counts).all(axis=1)&counts.ge(0).all(axis=1)
    coverage=(f'Potential assists and passes made are verified for {int(verified.iloc[-n:].sum())} of {n} recent games '
              f'and {int(verified.iloc[prior_start:-n].sum())} of {len(previous)} preceding games '
              f'({int(verified.sum())} of {len(sample)} season games).')
    if not verified.all():
        return dict(result,passing_detail=coverage+' The passing-role check is pending until the missing records are available.')
    detail=[coverage]
    ratios=[];trends=[]
    for stat, label in [('POTENTIAL_AST','potential assists'),('PASSES_MADE','passes made')]:
        rates=[x[stat].sum()/x.MIN.sum() for x in (merged,merged.iloc[-n:],merged.iloc[prior_start:-n])]
        if min(rates[0],rates[2])<=0:
            return dict(result,passing_detail=coverage+' The comparison period has zero passing opportunities; a percentage trend cannot be calculated.')
        ratios.append(rates[1]/rates[0]);trends.append(rates[1]/rates[2]-1)
        detail.append(f'{label.title()} per 36 minutes: {rates[2]*36:.2f} preceding → {rates[1]*36:.2f} recent ({trends[-1]:+.1%}).')
    result['ratios']['AST']=sum(ratios)/2
    result['trends']['AST']=sum(trends)/2
    detail.append(f'50/50 opportunity blend: {result["trends"]["AST"]:+.1%} recent vs preceding; '
                  f'{result["ratios"]["AST"]:.4f} × current-season opportunity rate. Conversion rates stay fixed; raw assist counts do not drive this trend.')
    return dict(result,passing_detail='\n'.join(detail))


def role_evidence(player, observed, logs, as_of, passing=None):
    """Compare nonoverlapping windows; retain conversion/accuracy, change volume.

Passing is optional verified player/game tracking: PLAYER_ID, GAME_ID,
POTENTIAL_AST, PASSES_MADE. Missing tracking is not replaced by noisy assists.
"""
    result=dict(ratios={},trends={},detail='Insufficient recent opportunity history.',
                passing_detail='The passing-role check needs at least three recent and five preceding appearances, with 45/60 minutes in the two periods.',
                rebound_detail='Needs five recent and at least five preceding appearances on this team, with 45/60 minutes and complete rebound records.')
    if observed is None or logs.empty:return result
    sample=logs.loc[logs.ESPN_PLAYER_ID.astype(str).eq(str(observed.ESPN_PLAYER_ID))].copy()
    dates=pd.to_datetime(sample.GAME_DATE,utc=True,errors='coerce')
    year=pd.Timestamp(as_of).year-(pd.Timestamp(as_of).month<7)
    sample=sample.loc[dates.lt(pd.Timestamp(as_of)) & dates.ge(pd.Timestamp(f'{year}-07-01',tz='UTC'))]
    if sample.GAME_ID.duplicated().any():return dict(result,detail='Duplicate player game logs; trend unchecked.',passing_detail='Duplicate player game logs prevent a reliable passing comparison.')
    if pd.notna(observed.get('GP')) and len(sample)!=int(observed.GP):
        return dict(result,detail='Season summary and available game logs disagree; trend unchecked.',passing_detail='The season summary and game-by-game records disagree. The passing-role check is pending.')
    sample=sample.loc[sample.Team.eq(player.Team)&sample.MIN.gt(0)].sort_values(['GAME_DATE','GAME_ID'])
    n=min(CONFIG['recent_games'],len(sample)-CONFIG['minimum_previous_games'])
    if n<CONFIG['minimum_recent_games']:return result
    recent=sample.iloc[-n:]
    previous=sample.iloc[max(0,len(sample)-n-CONFIG['previous_games']):-n]
    if recent.MIN.sum()<CONFIG['minimum_recent_minutes'] or previous.MIN.sum()<CONFIG['minimum_previous_minutes']:
        return result
    # Rebounds need no shooting or passing tracking. Keep this simple observed
    # rate comparison separate from the normalized opportunity scenarios below.
    if len(recent) == CONFIG['recent_games'] and 'REB' in sample:
        window = pd.concat([previous, recent])
        if np.isfinite(window[['REB','MIN']].to_numpy(float)).all() and window.REB.ge(0).all():
            recent_rate = float(recent.REB.sum()/recent.MIN.sum())
            previous_rate = float(previous.REB.sum()/previous.MIN.sum())
            if previous_rate > 0:
                change = recent_rate/previous_rate-1
                result['rebound'] = dict(rate=recent_rate, previous_rate=previous_rate, change=change)
                result['rebound_detail'] = (
                    f'{len(recent)} recent vs {len(previous)} preceding appearances on {player.Team}. '
                    f'Recent: {recent.REB.sum():g} rebounds / {recent.MIN.sum():g} minutes = {recent_rate:.4f} REB/min. '
                    f'Previous: {previous.REB.sum():g} rebounds / {previous.MIN.sum():g} minutes = {previous_rate:.4f} REB/min. '
                    f'Change: {change:+.1%}. A decline of at least {CONFIG["rebound_change_fraction"]:.0%} flags Overs; '
                    f'an increase of at least {CONFIG["rebound_change_fraction"]:.0%} flags Unders. '
                    'This is an observed rebounding trend, not proof of a role change; missed-shot opportunities can affect it.')
            else:
                result['rebound_detail'] = 'The preceding window has zero rebounds, so a percentage trend cannot be calculated.'
    # Passing tracking does not depend on complete shooting fields.
    passing_result=passing_role_evidence(sample,player.PLAYER_ID,n,passing)
    result['ratios'].update(passing_result['ratios'])
    result['trends'].update(passing_result['trends'])
    result['passing_detail']=passing_result['passing_detail']
    fields=['FGA','FG3A','FTA','FGM','FG3M','FTM','MIN']
    if not set(fields).issubset(sample):return result
    if not np.isfinite(sample[fields].to_numpy(float)).all() or sample[fields].lt(0).any().any():return result
    season=sample[fields].sum()
    if ((sample.FGM>sample.FGA)|(sample.FG3M>sample.FG3A)|(sample.FG3A>sample.FGA)|(sample.FTM>sample.FTA)).any():return result
    accuracy=player.get('Expected 3P%',np.nan)
    if not np.isfinite(accuracy):accuracy=season.FG3M/season.FG3A if season.FG3A>0 else 0.
    two=(season.FGM-season.FG3M)/(season.FGA-season.FG3A) if season.FGA>season.FG3A else 0.
    ft=season.FTM/season.FTA if season.FTA>0 else 0.
    def rate(frame,stat):
        x=frame[fields].sum()
        volume=x.FG3A if stat=='FG3A' else 2*(x.FGA-x.FG3A)*two+3*x.FG3A*accuracy+x.FTA*ft
        return volume/x.MIN
    for stat in ('PTS','FG3A'):
        old,new,all_rate=rate(previous,stat),rate(recent,stat),rate(sample,stat)
        if all_rate>0 and old>0:
            result['ratios'][stat]=new/all_rate
            result['trends'][stat]=new/old-1
    result['detail']=f'{len(recent)} recent vs {len(previous)} preceding appearances on {player.Team}; {recent.MIN.sum():.0f}/{previous.MIN.sum():.0f} minutes. Attempts per minute; shooting conversion held fixed.'
    if 'AST' not in result['ratios']:result['detail']+=' Passing-opportunity trend unavailable; assists not inferred from raw assist counts.'
    return result


def required_roles(stat):
    return {'PTS':('PTS',),'3PM':('FG3A',),'AST':('AST',),'REB':('REB',),'PRA':('PTS','AST')}.get(stat,())


def minute_reference(references, row, game, as_of):
    if references is None or references.empty:return None
    day=pd.Timestamp(game['GameTimeUTC']).tz_convert('America/New_York').date().isoformat()
    found=references.loc[references.PLAYER_ID.eq(row.PLAYER_ID)&references.Team.eq(row.Team)&
        references.Opponent.eq(row.Opponent)&references.game_date.eq(day)]
    if len(found)!=1:return None
    ref=found.iloc[0]
    try:
        stamp=pd.Timestamp(ref.updated_at)
        age=(pd.Timestamp(as_of)-stamp).total_seconds()/3600
        if stamp.tzinfo is None or not 0<=age<=CONFIG['minutes_source_max_hours'] or not isinstance(ref.independent,(bool,np.bool_)) or not ref.independent:return None
        if ref.get('requires_source_timestamp',False):
            published=pd.Timestamp(ref.get('source_updated_at'))
            if pd.isna(published) or published.tzinfo is None:return None
            source_age=(pd.Timestamp(as_of)-published).total_seconds()/3600
            if not 0<=source_age<=float(ref.get('source_max_age_hours',CONFIG['minutes_source_max_hours'])):return None
        if not np.isfinite(ref.minutes) or not 0<=ref.minutes<=48:return None
    except (ValueError,TypeError):return None
    return ref


def directional_injury_review(row, side, pending, team_frames, injury_info):
    """Screen availability by bet direction without inventing in/out projections."""
    affected, informational = {}, {}
    def add(groups, team, name, status, reason):
        names = groups.setdefault((team, reason), [])
        if (name, status) not in names:
            names.append((name, status))

    def join_names(names):
        return names[0] if len(names) == 1 else ' and '.join(names) if len(names) == 2 else ', '.join(names[:-1]) + ', and ' + names[-1]

    def describe(groups):
        messages = []
        for (team, reason), players in groups.items():
            statuses = {}
            for name, status in players:
                statuses.setdefault(status, []).append(name)
            clauses = [f'{join_names(names)} {"is" if len(names) == 1 else "are"} {status.lower()}'
                       for status, names in sorted(statuses.items(), key=lambda item: {'Out':0,'Doubtful':1,'Questionable':2}.get(item[0],3))]
            status_text = clauses[0] if len(clauses) == 1 else ', '.join(clauses[:-1]) + ', and ' + clauses[-1]
            explanation = {
                'own': 'The player’s own availability affects both Overs and Unders.',
                'opponent': f'Opponent availability could affect this {side}; its direction has not been verified.',
                'under': f'If {"this teammate sits" if len(players) == 1 else "any of these teammates sits"}, {row.Player} could gain minutes or production, hurting this Under.',
                'included': 'They are already included in your rotation. No teammate-Q warning on this Over; an absence could free up opportunity. This is a screening rule, not a calculated injury boost.',
                'return': f'{"He is" if len(players) == 1 else "They are"} currently excluded (0 projected minutes). A return could reduce {row.Player}’s opportunity and hurt this Over.',
                'unknown': 'Their minutes in this rotation could not be verified, so this Over still needs review.',
                'out': f'{"He still has" if len(players) == 1 else "They still have"} projected minutes. The rotation needs review for both directions.',
            }[reason]
            messages.append(f'{team}: {status_text}. {explanation}')
        return messages

    personal = injury_info.get(row.Team)
    own_status = personal.statuses.get(row.Player) if personal is not None else None
    if own_status in ('Questionable', 'Doubtful'):
        add(affected, row.Team, row.Player, own_status, 'own')
    for item in pending:
        team, name, status = item['Team'], item['Player'], item['Status']
        if team == row.Team and name == row.Player and own_status in ('Questionable', 'Doubtful'):
            continue
        if status == 'Out':
            add(affected, team, name, status, 'out')
        elif team != row.Team:
            add(affected, team, name, status, 'opponent')
        elif name == row.Player:
            add(affected, team, name, status, 'own')
        elif side == 'Under':
            add(affected, team, name, status, 'under')
        else:
            minutes = item['Minutes']
            players = team_frames.get(team)
            matches = players.loc[players.Player.eq(name)] if players is not None else pd.DataFrame()
            if len(matches) == 1:
                minutes = matches.iloc[0]['Projected minutes']
            try:
                minutes = float(minutes)
            except (TypeError, ValueError):
                minutes = np.nan
            if np.isfinite(minutes) and 0 < minutes <= 48:
                add(informational, team, name, status, 'included')
            elif minutes == 0:
                add(affected, team, name, status, 'return')
            else:
                add(affected, team, name, status, 'unknown')
    return describe(affected), describe(informational)


def assess_edges(edges, final, game, logs, matches, references=None, injury_info=None,
                 injury_stamp=None, as_of=None, passing=None):
    """Append explanations, scenario means, directional gaps and eligibility only.

No data source is interpreted as clear when missing. Direction-aware injury
warnings are a screening rule, not a numerical in/out forecast.
"""
    out=edges.copy(deep=True)
    now=pd.Timestamp(as_of or pd.Timestamp.now(tz='UTC'))
    deadline=min(now,pd.Timestamp(game['GameTimeUTC']))
    team_frames={team:rows.reset_index(drop=True) for team,rows in final.groupby('Team')}
    evidence={}
    for _,p in final.loc[final['Projected minutes'].gt(0)].iterrows():
        evidence[int(p.PLAYER_ID)]=role_evidence(p,matches.get(int(p.PLAYER_ID)),logs,deadline,passing)
    stale=True
    if injury_stamp is not None:
        stamp=pd.Timestamp(injury_stamp)
        stale=stamp.tzinfo is None or not 0<=(now-stamp).total_seconds()<=60*CONFIG['injury_max_age_minutes']
    injury_info=injury_info or {}
    complete=not stale and all(t in injury_info and injury_info[t].complete for t in (game['Away'],game['Home']))
    injury_info={team:info for team,info in injury_info.items() if team in (game['Away'],game['Home'])}
    pending=pending_key_players(injury_info)
    for team,info in injury_info.items():
        if team not in team_frames:continue
        active_names=set(team_frames[team].loc[team_frames[team]['Projected minutes'].gt(0),'Player'])
        listed={(item['Team'],item['Player']) for item in pending}
        pending.extend(dict(Team=team,Player=name,Status=status,Minutes=None)
                       for name,status in info.statuses.items()
                       if name in active_names and status in ('Out','Doubtful') and (team,name) not in listed)
    entries=[]
    for _,row in out.iterrows():
        side=row.Play if row.Play in ('Over','Under') else ('Over' if row.get('Signed edge %',0)>0 else 'Under')
        pid=int(row.PLAYER_ID); team=team_frames[row.Team]
        stat=SCENARIO_STATS[row.Stat]
        record={key:'Unchecked' for key in FLAG_COLUMNS}
        record.update({'Minutes detail':'Independent current-game minutes unavailable.',
            'Role detail':evidence.get(pid,{}).get('detail','Recent role history unavailable.'),
            'Injury detail':'Current, complete injury report unavailable.',
            'Minutes test minutes':np.nan,'Minutes test projection':np.nan,'Minutes test edge %':np.nan,
            'Recent test projection':np.nan,'Recent test edge %':np.nan,
            'Rebound trend %':np.nan,
            'Combined test projection':np.nan,'Combined test edge %':np.nan,
            'Reference minutes':np.nan,'Minutes difference':np.nan,'Reference source':'Unavailable',
            'Robust play':False,'Review group':'Unchecked'})
        passing_detail=evidence.get(pid,{}).get('passing_detail',record['Role detail'])
        if row.Stat == 'AST':
            record['Role detail']=passing_detail
        elif row.Stat == 'PRA':
            record['Role detail']+='\n'+passing_detail
        reference=minute_reference(references,row,game,now)
        target=max(0.,row['Projected minutes']-CONFIG['minutes_stress']) if side=='Over' else min(48.,row['Projected minutes']+CONFIG['minutes_stress'])
        if reference is not None:
            target=min(target,float(reference.minutes)) if side=='Over' else max(target,float(reference.minutes))
            record['Reference minutes']=float(reference.minutes)
            record['Minutes difference']=float(row['Projected minutes'])-float(reference.minutes)
            record['Reference source']=reference.source
            record['Minutes detail']=f"Your rotation: {row['Projected minutes']:g} min. {reference.source}: {reference.minutes:g} min (retrieved {reference.updated_at})."
            if pd.notna(reference.get('scenario')):record['Minutes detail']+=f" Scenario: {reference.scenario}."
            if pd.notna(reference.get('source_updated_at')):record['Minutes detail']+=f" Source last updated: {reference.source_updated_at}."
        minute_case=None
        try:
            minute_case=rebalance_minutes(team,pid,target)
            tested=scenario_projection(team,minute_case)
            value=float(tested.loc[tested.PLAYER_ID.eq(pid),stat].iloc[0])
            edge=side_edge(value,row['Book implied total'],side)
            record.update({'Minutes test minutes':target,'Minutes test projection':value,'Minutes test edge %':edge})
            adverse_diff=(row['Projected minutes']-float(reference.minutes))*(1 if side=='Over' else -1) if reference is not None else 0.
            record['Minutes flag']='MIN' if edge<=PLAY_THRESHOLD_PCT or adverse_diff>=CONFIG['minutes_disagreement'] else ('Clear' if reference is not None else 'Unchecked')
            record['Minutes detail']+=f'\nTest at {target:g} min: {value:.2f} {row.Stat}, {edge:+.1f}% {side} edge. Other active minutes redistribute proportionally to total 240, capped at 48; excluded players stay out.'
            if reference is None:record['Minutes detail']+=' Independent comparison still unchecked.'
        except (ValueError,KeyError,IndexError):record['Minutes detail']+=' Complete rotation test unavailable.'
        needed=required_roles(row.Stat)
        signal=evidence.get(pid,dict(ratios={},trends={}))
        covered=bool(needed) and all(s in signal['ratios'] for s in needed)
        if not needed:record['Role detail']='No verified recent opportunity measure for this category; not treated as a clean role check.'
        if row.Stat == 'REB':
            rebound=signal.get('rebound')
            covered=bool(rebound)
            record['Role detail']=signal.get('rebound_detail',signal.get('detail','Recent rebound history unavailable.'))
            if covered:
                # Diagnostic only: recent REB / recent MIN × selected minutes.
                # Do not replace model rebounds, PRA, team targets or probabilities.
                value=rebound['rate']*float(row['Projected minutes'])
                edge=side_edge(value,row['Book implied total'],side)
                adverse=rebound['change']*(1 if side=='Over' else -1)
                flag=adverse<=-CONFIG['rebound_change_fraction']+1e-12
                record.update({'Recent test projection':value,'Recent test edge %':edge,
                               'Rebound trend %':100*rebound['change'], 'Role flag':'ROLE' if flag else 'Clear'})
                record['Role detail']+=(f'\nRecent-rate projection: {rebound["rate"]:.4f} REB/min × '
                    f'{row["Projected minutes"]:g} projected minutes = {value:.2f} REB. '
                    f'Remaining {side} edge: {edge:+.1f}% versus the book-implied total. '
                    'This comparison does not apply another matchup adjustment or rebalance team totals; the main model is unchanged.')
                if minute_case:
                    value=rebound['rate']*minute_case[pid]
                    record.update({'Combined test projection':value,'Combined test edge %':side_edge(value,row['Book implied total'],side)})
        elif covered:
            role={pid:signal['ratios']}
            try:
                tested=scenario_projection(team,role=role)
                value=float(tested.loc[tested.PLAYER_ID.eq(pid),stat].iloc[0])
                edge=side_edge(value,row['Book implied total'],side)
                trend=any(signal['trends'][s]*(1 if side=='Over' else -1)<=-CONFIG['role_change_fraction'] for s in needed)
                hurts=edge<side_edge(row.Projected,row['Book implied total'],side)-1e-9
                record.update({'Recent test projection':value,'Recent test edge %':edge,
                    'Role flag':'ROLE' if hurts and (trend or edge<=PLAY_THRESHOLD_PCT) else 'Clear'})
                record['Role detail']+=f'\nAt unchanged minutes: {value:.2f} {row.Stat}, {edge:+.1f}% {side} edge. Stress factor = recent opportunity rate / current-season opportunity rate, applied to this player’s weight; teammate weights and team targets remain fixed before normalization.'
                if minute_case:
                    joint=scenario_projection(team,minute_case,role)
                    value=float(joint.loc[joint.PLAYER_ID.eq(pid),stat].iloc[0])
                    record.update({'Combined test projection':value,'Combined test edge %':side_edge(value,row['Book implied total'],side)})
            except (ValueError,KeyError,IndexError):covered=False
        row_pending, informational=directional_injury_review(row,side,pending,team_frames,injury_info)
        injury_ok=complete and not row_pending
        if row_pending:
            record['Injury flag']='Q'
            record['Injury detail']='\n'.join(row_pending)
            if stale:record['Injury detail']+=' Report is stale; refresh required.'
            record['Injury detail']+='\nNo verified in/out rotations saved. Needs review; no guessed conditional projection.'
        elif injury_ok:
            record['Injury flag']='Clear';record['Injury detail']='Both teams have complete, recent reports; no adverse availability warning under the direction-aware rules.'
        if informational:record['Injury detail']+='\n'+'\n'.join(informational)
        gaps=[side_edge(row.Projected,row['Book implied total'],side),record['Minutes test edge %'],record['Recent test edge %'],record['Combined test edge %']]
        robust=(row.Play in ('Over','Under') and reference is not None and covered and injury_ok
                and np.isfinite(gaps).all() and min(round(g,2) for g in gaps)>PLAY_THRESHOLD_PCT
                and (row.Stat!='REB' or record['Role flag']=='Clear'))
        record['Robust play']=bool(robust)
        record['Review group']='More robust' if robust else 'Waiting on news' if row_pending else 'Unchecked' if reference is None or not covered or not complete else 'Needs review'
        record['Review detail']='All checks covered; each scenario retains more than 5% directional edge.' if robust else 'Not cleared by all review checks.'
        if np.isfinite(record['Combined test edge %']):
            record['Review detail']+=f" Combined minutes + role test: {record['Combined test projection']:.2f} {row.Stat}, {record['Combined test edge %']:+.1f}% {side} edge."
        record['Review flags']=' · '.join(record[k] for k in FLAG_COLUMNS if record[k] in ('MIN','ROLE','Q')) or ('✓' if robust else 'Review')
        if any(record[k]=='Unchecked' for k in FLAG_COLUMNS):record['Review flags']+=' · ?'
        record['Review version']=CONFIG['version'];record['Review captured UTC']=now.isoformat()
        entries.append(record)
    for key in {k for r in entries for k in r}:out[key]=[r.get(key,np.nan) for r in entries]
    if not entries:
        for key in ['Robust play','Review group','Review flags',*FLAG_COLUMNS]:out[key]=pd.Series(index=out.index,dtype=object)
    return out


def filter_review(rows, view):
    if view=='All plays':return rows.copy()
    if view=='More robust plays':return rows.loc[rows['Robust play'].eq(True)].copy()
    if view=='Waiting on news':return rows.loc[rows['Review group'].eq('Waiting on news')].copy()
    if view=='Needs review / unchecked':return rows.loc[rows['Review group'].isin(['Needs review','Unchecked'])].copy()
    raise ValueError('Unknown review filter.')
