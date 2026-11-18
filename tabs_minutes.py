"""Private, session-only TabsNBA minutes source for blending and review."""
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

# Tabs uses the short name; resolve only this known alias on the same team.
ALIASES = {('POR','robert williams'):'robert williams iii',
           ('PHI','labaron philon jr'):'labaron philon'}
SOURCE_MAX_AGE_HOURS = 2


def parse_tabs_browser_snapshot(snapshot, roster, games, as_of=None):
    """Validate a displayed-page snapshot, reusing the strict same-game name map."""
    now=pd.Timestamp(as_of if as_of is not None else pd.Timestamp.now(tz='UTC'))
    if not isinstance(snapshot,dict) or len(json.dumps(snapshot))>150000:
        raise ValueError('Invalid Tabs browser comparison.')
    if (snapshot.get('schema')!=1 or snapshot.get('source')!='TabsNBA' or
        snapshot.get('source_url')!='https://app.tabsnba.com/projections' or snapshot.get('manual_edits') is not False):
        raise ValueError('Use an unedited TabsNBA projections view for an independent comparison.')
    observed=pd.Timestamp(snapshot.get('observed_at'));published=pd.Timestamp(snapshot.get('source_updated_at'))
    for stamp in (observed,published):
        if pd.isna(stamp) or stamp.tzinfo is None or not 0<=(now-stamp).total_seconds()<=SOURCE_MAX_AGE_HOURS*3600:
            raise ValueError('Tabs comparison is stale or its timestamp cannot be verified.')
    rows=snapshot.get('rows')
    if not isinstance(rows,list) or not 1<=len(rows)<=600:raise ValueError('No usable players in the Tabs view.')
    frame=pd.DataFrame(rows)
    if not {'Player','Team','Opp','minutes','status'}.issubset(frame):raise ValueError('Incomplete Tabs minutes rows.')
    if not frame.status.isin(['Projected','Out']).all():raise ValueError('Unknown Tabs row status.')
    if frame.loc[frame.status.eq('Out'),'minutes'].ne(0).any():raise ValueError('An OUT row must explicitly have zero minutes.')
    frame=frame.rename(columns={'minutes':'Tabs Min'})
    refs,details=parse_tabs_minutes(frame.to_csv(index=False).encode(),f"Tabs_DK_{snapshot.get('slate_date')}.csv",
        roster,games,observed,snapshot.get('scenario'),published)
    refs['source_mode']='Browser comparison'
    details['scope']=str(snapshot.get('scope','Displayed players'))[:120]
    return refs,details


@lru_cache(maxsize=1)
def _browser_component(runtime):
    return st.components.v2.component('tabs_minutes_browser',
        html='<div><button type="button">Check Tabs connection</button> <span role="status">Connecting to Tabs…</span></div>',
        css='button {font:inherit;padding:.35rem .7rem;border:1px solid #8888;border-radius:.4rem;color:var(--st-text-color);background:var(--st-background-color);cursor:pointer} span {font-size:.9rem}',
        js=Path(__file__).with_name('tabs_browser_bridge.mjs').read_text())


def _remember_browser_result():
    state=st.session_state.get('tabs_minutes_browser',{})
    st.session_state['tabs_browser_last_result']=state.get('result') if isinstance(state,dict) else None


def render_tabs_comparison(ui,roster,games):
    with ui.expander('TabsNBA minutes comparison',expanded=False):
        mode=ui.radio('Tabs connection',['Browser connection','CSV upload'],horizontal=True,key='tabs_minutes_mode')
        ui.caption('Fresh, complete rotations blend 50/50 with your minutes, including healthy-player zeroes. Explicit Tabs OUT statuses stay at zero. Players omitted from a complete 240-minute browser rotation count as zero for that source. Team minutes stay at 240; your saved rotation is preserved.')
        if mode=='CSV upload':return render_tabs_upload(ui,roster,games)
        previous=st.session_state.get('tabs_browser_last_result')
        result=_browser_component(st.runtime.get_instance())(data={'previous':previous},
            key='tabs_minutes_browser',on_result_change=_remember_browser_result)
        value=result.result
        st.session_state['tabs_browser_last_result']=value
        ui.caption('Connect one TabsNBA tab using the Chrome extension, and keep it open alongside this app in Chrome. Reads the displayed players and selected If/Then scenario every 30 seconds; select All games and clear filters in Tabs for full coverage. Only changed comparisons refresh the flags. Tabs controls when its projections update.')
        if not value or not isinstance(value,dict) or not value.get('ok'):
            ui.info('Waiting for a current Tabs comparison. Missing data remains unchecked.')
            return None
        try:refs,details=parse_tabs_browser_snapshot(value['snapshot'],roster,games)
        except (ValueError,TypeError,KeyError,OverflowError) as exc:
            ui.warning(str(exc));return None
        ui.caption(f"{details['export_date']} · {details['scope']} · {details['scenario']} · {len(refs)} matched players")
        if details['unmatched']:
            ui.warning(f"{len(details['unmatched'])} displayed players have no matching roster profile. Their comparisons remain unchecked.")
            ui.dataframe(pd.DataFrame(details['unmatched']),hide_index=True,width='stretch')
        return refs


def export_day(filename):
    found=re.fullmatch(r'Tabs_(?:DK|FD)_(\d{4}-\d{2}-\d{2})(?: \(\d+\))?\.csv',str(filename))
    if not found:raise ValueError('Use the original Tabs_DK_YYYY-MM-DD.csv or Tabs_FD_YYYY-MM-DD.csv export name.')
    return pd.Timestamp(found.group(1)).date()


def parse_tabs_minutes(content, filename, roster, games, captured_at, scenario, source_updated_at=None):
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
            source_status=r.get('status','Unknown')))
    columns=['PLAYER_ID','Team','Opponent','game_date','minutes','source','updated_at','independent','scenario',
             'source_updated_at','source_max_age_hours','requires_source_timestamp','source_mode','source_status']
    result=pd.DataFrame(rows,columns=columns)
    # Unmatched reserves with explicit zero minutes contribute no allocation.
    # Missing players can be resolved only from a complete browser rotation.
    result.attrs['incomplete_teams'] = sorted({row['Team'] for row in unmatched if row['Minutes']>0})
    if result.duplicated(['PLAYER_ID','Team','Opponent']).any():raise ValueError('Multiple Tabs names map to the same player.')
    return result,dict(export_date=day.isoformat(),export_rows=len(frame),matched=len(result),
                       other_games=other_games,unmatched=unmatched,scenario=scenario.strip())


def render_tabs_upload(ui, roster, games):
    """Session-only import; paid data is not bundled, cached globally or published."""
    with ui.container():
        ui.caption('Fresh, complete Tabs rotations blend with your minutes. This CSV is a snapshot; it does not automatically refresh the Tabs website. Missing or stale coverage retains your minutes and stays unchecked.')
        upload=ui.file_uploader('TabsNBA projections CSV',type=['csv'],key='tabs_minutes_csv')
        if upload is None:
            ui.info('Export your selected scenario from TabsNBA, then upload it here. Missing Tabs comparisons remain unchecked.')
            return None
        try:day=export_day(upload.name)
        except ValueError as exc:ui.warning(str(exc));return None
        scenario=ui.text_input('Tabs scenario used for this export',placeholder='Default, or POR: Avdija OUT',key='tabs_minutes_scenario')
        left,right=ui.columns(2)
        with left:published_day=ui.date_input('Tabs update date',value=day,key='tabs_minutes_updated_day')
        with right:published_time=ui.time_input('Tabs updated time (Denver)',value=None,key='tabs_minutes_updated_time')
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
