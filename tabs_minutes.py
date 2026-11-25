"""Private TabsNBA minutes source, with a browser-local last-valid snapshot."""
from datetime import datetime
from hashlib import sha256
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import json
import re
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

from live_projections import name_key
from tabs_minutes_status import minutes_metadata
from tabs_projection_review import TABS_STATS, tabs_stat_values

# Tabs uses the short name; resolve only this known alias on the same team.
ALIASES = {('POR','robert williams'):'robert williams iii',
           ('PHI','labaron philon jr'):'labaron philon'}
SOURCE_MAX_AGE_HOURS = 2


def parse_tabs_browser_snapshot(snapshot, roster, games, as_of=None, *, saved=False):
    """Validate a displayed-page snapshot, reusing the strict same-game name map."""
    now=pd.Timestamp(as_of if as_of is not None else pd.Timestamp.now(tz='UTC'))
    if not isinstance(snapshot,dict) or len(json.dumps(snapshot))>300000:
        raise ValueError('Invalid Tabs browser comparison.')
    if (snapshot.get('schema')!=1 or snapshot.get('source')!='TabsNBA' or
        snapshot.get('source_url')!='https://app.tabsnba.com/projections' or snapshot.get('manual_edits') is not False):
        raise ValueError('Use an unedited TabsNBA projections view for an independent comparison.')
    observed=pd.Timestamp(snapshot.get('observed_at'));published=pd.Timestamp(snapshot.get('source_updated_at'))
    for stamp in (observed,published):
        if (pd.isna(stamp) or stamp.tzinfo is None or (now-stamp).total_seconds()<0 or
                (not saved and (now-stamp).total_seconds()>SOURCE_MAX_AGE_HOURS*3600)):
            raise ValueError('Tabs comparison is stale or its timestamp cannot be verified.')
    # A retained snapshot must have been current when captured. Preserve its real
    # timestamps; allowing older saved minutes never manufactures a fresh check.
    if not 0<=(observed-published).total_seconds()<=SOURCE_MAX_AGE_HOURS*3600:
        raise ValueError('Tabs minutes were not current when captured.')
    rows=snapshot.get('rows')
    if not isinstance(rows,list) or not 1<=len(rows)<=600:raise ValueError('No usable players in the Tabs view.')
    frame=pd.DataFrame(rows)
    if not {'Player','Team','Opp','minutes','status'}.issubset(frame):raise ValueError('Incomplete Tabs minutes rows.')
    if not frame.status.isin(['Projected','Out']).all():raise ValueError('Unknown Tabs row status.')
    if frame.loc[frame.status.eq('Out'),'minutes'].ne(0).any():raise ValueError('An OUT row must explicitly have zero minutes.')
    frame=frame.rename(columns={'minutes':'Tabs Min'})
    refs,details=parse_tabs_minutes(frame.to_csv(index=False).encode(),f"Tabs_DK_{snapshot.get('slate_date')}.csv",
        roster,games,observed,snapshot.get('scenario'),published, unedited_projection_stats=True)
    refs['source_mode']='Browser comparison'
    refs['source_saved']=saved
    details['scope']=str(snapshot.get('scope','Displayed players'))[:120]
    return refs,details


@lru_cache(maxsize=1)
def _browser_component(runtime):
    return st.components.v2.component('tabs_minutes_browser_saved_v2',
        html='<div><button type="button">Check Tabs connection</button> <span role="status">Connecting to Tabs…</span></div>',
        css='button {font:inherit;padding:.35rem .7rem;border:1px solid #8888;border-radius:.4rem;color:var(--st-text-color);background:var(--st-background-color);cursor:pointer} span {font-size:.9rem}',
        js=Path(__file__).with_name('tabs_browser_bridge.mjs').read_text())


def _reference_signature(refs, as_of=None):
    """Projection inputs, excluding heartbeat/publication times and row order."""
    if refs is None:
        return None
    columns=sorted(set(refs.columns)-{'updated_at','source_updated_at'})
    inputs=refs[columns].copy()
    now=pd.Timestamp(as_of if as_of is not None else pd.Timestamp.now(tz='UTC'))
    # Crossing the freshness boundary changes eligibility even when allocations
    # are identical, so restored/stale comparisons must refresh every page.
    for clock in ('updated_at','source_updated_at'):
        age=(now-pd.to_datetime(refs[clock],utc=True,errors='coerce')).dt.total_seconds()
        inputs[clock+'_fresh']=age.between(0,SOURCE_MAX_AGE_HOURS*3600)
    rows=inputs.sort_values(['Team','Opponent','PLAYER_ID']).to_json(orient='split',index=False)
    gaps=tuple(sorted((r['Team'],r['Opponent'],r['Player'],r['Minutes'])
                      for r in refs.attrs.get('unmatched_players',[])))
    return rows, tuple(sorted(refs.attrs.get('incomplete_teams',[]))), gaps


def _refresh_dashboard():
    # A named callback rerun replaces the small panel's normal rerun. The fixed
    # Streamlit scheduler coalesces any overlapping child-widget events.
    st.rerun(scope='projection_dashboard')


def _saved_parts(cache):
    """Migrate the original single snapshot; never let one filtered team erase others."""
    if not isinstance(cache,dict) or len(json.dumps(cache))>900000:return []
    if cache.get('schema')==1:return [cache]
    parts=cache.get('snapshots',[])
    if cache.get('schema')==2 and isinstance(parts,list) and len(parts)<=90:
        return parts
    return []


def _team_parts(snapshot):
    grouped={}
    for row in snapshot['rows']:
        grouped.setdefault((snapshot['slate_date'],row['Team'],row['Opp']),[]).append(row)
    return [(key,snapshot|{'rows':rows,'scope':f'{key[1]} vs {key[2]}'}) for key,rows in grouped.items()]


def _complete_rotation(refs):
    return (not refs.empty and not refs.attrs.get('incomplete_teams') and
            refs.minutes.gt(0).sum()>=5 and abs(refs.minutes.sum()-240)<=2)


def resolve_browser_result(value, saved_snapshot, roster, games, as_of=None):
    """Merge exact-game team snapshots, preserving complete backups under filters."""
    entries={};error='Tabs is reconnecting.'
    # Include the browser backup even with a successful fresh reply: the first
    # reply can beat the restore callback after a page reload.
    candidates=[value.get('saved_snapshot') if isinstance(value,dict) else None,saved_snapshot]
    for candidate in candidates:
        for snapshot in _saved_parts(candidate):
            try:
                parse_tabs_browser_snapshot(snapshot,roster,games,as_of,saved=True)
                for key,part in _team_parts(snapshot):
                    refs,details=parse_tabs_browser_snapshot(part,roster,games,as_of,saved=True)
                    if refs.empty:continue
                    previous=entries.get(key)
                    if previous and _complete_rotation(previous[1]) and not _complete_rotation(refs):continue
                    if previous and pd.Timestamp(previous[0]['source_updated_at'])>pd.Timestamp(part['source_updated_at']):continue
                    entries[key]=(part,refs,details)
            except (ValueError,TypeError,KeyError,OverflowError):
                continue
    if isinstance(value,dict) and value.get('ok'):
        try:
            snapshot=value['snapshot']
            parsed,_=parse_tabs_browser_snapshot(snapshot,roster,games,as_of)
            if parsed.empty:raise ValueError('Tabs has no matching players for the loaded schedule.')
            error=''
            for key,part in _team_parts(snapshot):
                refs,details=parse_tabs_browser_snapshot(part,roster,games,as_of)
                if refs.empty:continue
                previous=entries.get(key)
                if previous and ((_complete_rotation(previous[1]) and not _complete_rotation(refs)) or
                        pd.Timestamp(previous[0]['source_updated_at'])>pd.Timestamp(part['source_updated_at'])):
                    # A search/position filter or delayed response is not a new rotation.
                    continue
                entries[key]=(part,refs,details)
        except (ValueError,TypeError,KeyError,OverflowError) as exc:
            error=str(exc)
    elif isinstance(value,dict):
        error=str(value.get('error') or error)[:250]
    if not entries:return None,None,saved_snapshot,error
    ordered=[entries[key] for key in sorted(entries)]
    refs=pd.concat([item[1] for item in ordered],ignore_index=True)
    refs.attrs['incomplete_teams']=sorted({team for _,r,_ in ordered for team in r.attrs.get('incomplete_teams',[])})
    refs.attrs['unmatched_players']=[row for _,r,_ in ordered for row in r.attrs.get('unmatched_players',[])]
    stored={'schema':2,'snapshots':[item[0] for item in ordered]}
    details=dict(export_date=', '.join(sorted({k[0] for k in entries})),
        scope=f"{len(entries)} team rotations", scenario=' / '.join(sorted({r['scenario'] for _,_,r in ordered})),
        unmatched=[row for _,_,d in ordered for row in d['unmatched']])
    return refs,details,stored,error


def _apply_browser_result(value,roster,games):
    refs,details,saved,error=resolve_browser_result(value,st.session_state.get('tabs_browser_saved_snapshot'),roster,games)
    st.session_state['tabs_browser_last_result']=value
    st.session_state['tabs_browser_saved_snapshot']=saved
    return refs,details,error


def _remember_browser_result(roster, games):
    state=st.session_state.get('tabs_minutes_browser',{})
    value=state.get('result') if isinstance(state,dict) else None
    refs,_,_=_apply_browser_result(value,roster,games)
    changed=_reference_signature(refs)!=_reference_signature(st.session_state.get('tabs_minutes_reference'))
    st.session_state['tabs_minutes_reference']=refs
    # Renew freshness without rebuilding the league for a publication-time-only
    # update. Real minutes/status/scenario/coverage changes still update all pages.
    if changed:
        _refresh_dashboard()


@st.fragment
def render_tabs_comparison(ui,roster,games):
    with ui.expander('TabsNBA minutes comparison',expanded=False):
        mode=ui.radio('Tabs connection',['Browser connection','CSV upload'],horizontal=True,key='tabs_minutes_mode',on_change=_refresh_dashboard)
        ui.caption('Fresh, complete rotations blend 50/50 with your minutes, including healthy-player zeroes. Explicit Tabs OUT statuses stay at zero. Players omitted from a complete 240-minute browser rotation count as zero for that source. Team minutes stay at 240; your saved rotation is preserved.')
        if mode=='CSV upload':return render_tabs_upload(ui,roster,games)
        previous=st.session_state.get('tabs_browser_last_result')
        result=_browser_component(st.runtime.get_instance())(data={'previous':previous,
            'metadata':minutes_metadata(st.session_state.get('tabs_minutes_reference')),
            'saved_snapshot':st.session_state.get('tabs_browser_saved_snapshot')},
            key='tabs_minutes_browser',on_result_change=lambda:_remember_browser_result(roster,games))
        value=result.result
        refs,details,error=_apply_browser_result(value,roster,games)
        ui.caption('Keep TabsNBA and this app open in Chrome. Connector 0.3.0 compares minutes and stat projections in either Projections or Props view and checks every 30 seconds. Timestamp-only updates stay in this panel. Select All games and clear Tabs filters for full coverage.')
        ui.caption('Each team’s last valid Tabs rotation is saved privately in this browser. Filtering to one team or player cannot erase the other saved rotations. Saved minutes stay in projections until that team has a valid fresh update; the live minutes check remains pending.')
        if refs is None:
            ui.info('Waiting for a current Tabs comparison. Missing data remains unchecked.')
            if error:ui.caption(error)
            return None
        if refs.source_saved.any():
            saved_refs=refs.loc[refs.source_saved]
            names=', '.join(sorted(saved_refs.Team.unique()))
            ui.warning(f'Using saved Tabs minutes for {names}. Blended projections are preserved; their live minutes check is pending. Select all teams in Tabs to refresh the whole slate.')
            if error:ui.caption(error)
        coverage=refs.groupby(['Team','Opponent','source_saved','source_updated_at','scenario'],dropna=False).agg(Players=('PLAYER_ID','size'),Minutes=('minutes','sum')).reset_index()
        coverage['Status']=coverage.source_saved.map({True:'Saved Tabs minutes',False:'Live Tabs minutes'})
        coverage['Tabs updated']=pd.to_datetime(coverage.source_updated_at,utc=True).dt.tz_convert('America/Denver').dt.strftime('%b %d %I:%M %p %Z')
        ui.dataframe(coverage[['Team','Opponent','Status','Tabs updated','Players','Minutes','scenario']].rename(columns={'scenario':'Scenario'}),hide_index=True,width='stretch')
        stat_columns=[f'Tabs projection {stat}' for stat in TABS_STATS]
        stat_players=int(refs.reindex(columns=stat_columns).notna().any(axis=1).sum())
        ui.caption(f'Tabs stat projections received for {stat_players} of {len(refs)} matched players. '
                   'Only fresh, active-player values can be compared with sportsbook lines.')
        if isinstance(value,dict) and value.get('ok') and stat_players == 0:
            browser_rows=value.get('snapshot',{}).get('rows',[])
            supplied=sum(any(stat in row for stat in TABS_STATS) for row in browser_rows if isinstance(row,dict))
            ui.caption(f'Connection diagnostic: the browser supplied stat fields for {supplied} of {len(browser_rows)} player rows.')
        with ui.expander('View saved player minutes and projections',expanded=False):
            saved_rows=refs.merge(roster[['PLAYER_ID','TEAM_ABBREVIATION','PLAYER_NAME']].drop_duplicates(),
                left_on=['PLAYER_ID','Team'],right_on=['PLAYER_ID','TEAM_ABBREVIATION'],how='left')
            saved_rows['Status']=saved_rows.source_saved.map({True:'Saved',False:'Live'})
            saved_rows['Tabs updated']=pd.to_datetime(saved_rows.source_updated_at,utc=True).dt.tz_convert('America/Denver').dt.strftime('%b %d %I:%M %p %Z')
            ui.dataframe(saved_rows[['PLAYER_NAME','Team','Opponent','minutes','source_status','Status','Tabs updated',*stat_columns]].rename(
                columns={'PLAYER_NAME':'Player','minutes':'Tabs minutes','source_status':'Availability'}),hide_index=True,width='stretch')
        ui.caption(f"{details['export_date']} · {details['scope']} · {details['scenario']} · {len(refs)} matched players")
        if details['unmatched']:
            ui.warning(f"{len(details['unmatched'])} displayed players have no matching roster profile. Their comparisons remain unchecked.")
            ui.dataframe(pd.DataFrame(details['unmatched']),hide_index=True,width='stretch')
        return refs


def export_day(filename):
    found=re.fullmatch(r'Tabs_(?:DK|FD)_(\d{4}-\d{2}-\d{2})(?: \(\d+\))?\.csv',str(filename))
    if not found:raise ValueError('Use the original Tabs_DK_YYYY-MM-DD.csv or Tabs_FD_YYYY-MM-DD.csv export name.')
    return pd.Timestamp(found.group(1)).date()


def parse_tabs_minutes(content, filename, roster, games, captured_at, scenario, source_updated_at=None, *, unedited_projection_stats=False):
    """Map Tabs Min, never My Min or its fantasy-platform ID, to our roster.

    CSV omits out players. Missing rows therefore stay missing; they never become
    zero-minute assumptions. Date + team + opponent must match a verified game.
    The export contains no publication time or scenario label; these must be
    supplied from the visible site. Unknown publication time cannot pass freshness.
    """
    day=export_day(filename);capture=pd.Timestamp(captured_at)
    if capture.tzinfo is None:raise ValueError('The export capture needs a timezone.')
    if not isinstance(scenario,str) or not scenario.strip():raise ValueError('Specify Default or the selected Tabs If/Then scenario.')
    if len(scenario)>200:raise ValueError('Keep the scenario label under 200 characters.')
    published=pd.Timestamp(source_updated_at) if source_updated_at is not None else None
    if published is not None and (published.tzinfo is None or published>capture):
        raise ValueError('Tabs publication time must have a timezone and cannot be in the future.')
    frame=pd.read_csv(BytesIO(content))
    required=['Player','Team','Opp','Tabs Min']
    if not set(required).issubset(frame):raise ValueError('This is not a TabsNBA projections export: Player, Team, Opp and Tabs Min are required.')
    if frame[required].isna().any().any():raise ValueError('Tabs export contains missing names, teams or minutes.')
    frame['Tabs Min']=pd.to_numeric(frame['Tabs Min'],errors='coerce')
    if not np.isfinite(frame['Tabs Min']).all() or not frame['Tabs Min'].between(0,48).all():
        raise ValueError('Tabs minutes must be finite numbers from 0 to 48.')
    frame['match_name']=frame.Player.map(name_key)
    if frame.duplicated(['Team','Opp','match_name']).any():raise ValueError('Duplicate players in the Tabs export; use one complete export.')
    pairs=set()
    for game in games:
        if pd.Timestamp(game['GameTimeUTC']).tz_convert('America/New_York').date()==day:
            pairs.update([(game['Away'],game['Home']),(game['Home'],game['Away'])])
    names={}
    for player in roster.itertuples():
        for alias in (player.PLAYER_NAME,getattr(player,'WorkbookPlayerName',None)):
            if pd.notna(alias):names.setdefault((player.TEAM_ABBREVIATION,name_key(alias)),set()).add(int(player.PLAYER_ID))
    rows=[];unmatched=[];other_games=0
    if 'status' in frame:
        if not frame.status.isin(['Projected','Out']).all():raise ValueError('Unknown Tabs row status.')
        if frame.loc[frame.status.eq('Out'),'Tabs Min'].ne(0).any():raise ValueError('An OUT row must explicitly have zero minutes.')
    for _,r in frame.iterrows():
        team,opponent=str(r.Team).strip(),str(r.Opp).strip()
        if (team,opponent) not in pairs:other_games+=1;continue
        key=ALIASES.get((team,r.match_name),r.match_name)
        ids=names.get((team,key),set())
        if len(ids)!=1:
            unmatched.append(dict(Player=r.Player,Team=team,Opponent=opponent,Minutes=float(r['Tabs Min']),Reason='No unique matching profile on this team'))
            continue
        rows.append(dict(PLAYER_ID=next(iter(ids)),Team=team,Opponent=opponent,game_date=day.isoformat(),
            minutes=float(r['Tabs Min']),source='TabsNBA',updated_at=capture.isoformat(),independent=True,
            scenario=scenario.strip(),source_updated_at=published.isoformat() if published is not None else None,
            source_max_age_hours=SOURCE_MAX_AGE_HOURS,requires_source_timestamp=True,source_mode='CSV snapshot',
            source_status=r.get('status','Unknown'), **tabs_stat_values(r, unedited=unedited_projection_stats)))
    columns=['PLAYER_ID','Team','Opponent','game_date','minutes','source','updated_at','independent','scenario',
             'source_updated_at','source_max_age_hours','requires_source_timestamp','source_mode','source_status',
             *tabs_stat_values({})]
    result=pd.DataFrame(rows,columns=columns)
    # Unmatched reserves with explicit zero minutes contribute no allocation.
    # Missing players can be resolved only from a complete browser rotation.
    result.attrs['incomplete_teams'] = sorted({row['Team'] for row in unmatched if row['Minutes']>0})
    result.attrs['unmatched_players'] = [row for row in unmatched if row['Minutes']>0]
    if result.duplicated(['PLAYER_ID','Team','Opponent']).any():raise ValueError('Multiple Tabs names map to the same player.')
    return result,dict(export_date=day.isoformat(),export_rows=len(frame),matched=len(result),
                       other_games=other_games,unmatched=unmatched,scenario=scenario.strip())


def render_tabs_upload(ui, roster, games):
    """Session-only import; paid data is not bundled, cached globally or published."""
    with ui.container():
        ui.caption('Fresh, complete Tabs rotations blend with your minutes. This CSV is a snapshot; it does not automatically refresh the Tabs website. Missing or stale coverage retains your minutes and stays unchecked.')
        upload=ui.file_uploader('TabsNBA projections CSV',type=['csv'],key='tabs_minutes_csv',on_change=_refresh_dashboard)
        if upload is None:
            ui.info('Export your selected scenario from TabsNBA, then upload it here. Missing Tabs comparisons remain unchecked.')
            return None
        try:day=export_day(upload.name)
        except ValueError as exc:ui.warning(str(exc));return None
        scenario=ui.text_input('Tabs scenario used for this export',placeholder='Default, or POR: Avdija OUT',key='tabs_minutes_scenario',on_change=_refresh_dashboard)
        left,right=ui.columns(2)
        with left:published_day=ui.date_input('Tabs update date',value=day,key='tabs_minutes_updated_day',on_change=_refresh_dashboard)
        with right:published_time=ui.time_input('Tabs updated time (Denver)',value=None,key='tabs_minutes_updated_time',on_change=_refresh_dashboard)
        ui.caption('Use the Last updated time shown on Tabs, not the download time. Without it, freshness is unchecked; snapshots older than two hours cannot clear the minutes check. Out players omitted from the export are not assumed to have zero minutes.')
        if not scenario.strip():
            ui.info('Enter the selected If/Then scenario, or Default if none was selected. The CSV does not record it.')
            return None
        content=upload.getvalue();fingerprint=sha256(content).hexdigest()
        captures=st.session_state.setdefault('tabs_minutes_captured',{})
        capture=captures.setdefault(fingerprint,pd.Timestamp.now(tz='UTC').isoformat())
        published=(datetime.combine(published_day,published_time,tzinfo=ZoneInfo('America/Denver'))
                   if published_time is not None else None)
        try:
            references,details=parse_tabs_minutes(content,upload.name,roster,games,capture,scenario,published)
        except (ValueError,KeyError,TypeError,UnicodeError) as exc:
            ui.warning(str(exc));return None
        ui.caption(f"TabsNBA · {details['export_date']} · {details['scenario']} · {len(references)} matching players. Uses Tabs Min; ignores My Min and Boost edits.")
        if details['unmatched']:
            ui.warning(f"{len(details['unmatched'])} Tabs players could not be matched to this game's roster. Their comparisons stay unchecked.")
            ui.dataframe(pd.DataFrame(details['unmatched']),hide_index=True,width='stretch')
        if published is None:ui.info('Add the Tabs update time to verify freshness.')
        elif (pd.Timestamp.now(tz='UTC')-pd.Timestamp(published)).total_seconds()>SOURCE_MAX_AGE_HOURS*3600:
            ui.warning('This Tabs update is over two hours old. Upload a current export before treating these comparisons as checked.')
        return references
