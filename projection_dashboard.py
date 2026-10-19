"""Streamlit player-stat projection view; no betting or odds dependency."""
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

from live_projections import (ESPN, build_projections, fetch_json, load_live_state, parse_schedule)
from season_pipeline import ROOT, SEASON, OPENING_GAMES
from rotowire_fallback import normalize_rotowire, read_rotowire_csv


@st.cache_data(ttl=300, show_spinner=False)
def live_state(season):
    return load_live_state(season)


@st.cache_data(ttl=300, show_spinner=False)
def today_schedule(season, date):
    payload = fetch_json(f'{ESPN}/site/v2/sports/basketball/nba/scoreboard?dates={date:%Y%m%d}')
    return parse_schedule(payload, int(season[:4]) + 1)


@st.fragment(run_every=300)
def render_live_dashboard(ui, roster):
    ui.title(f'🏀 Player Stat Projections ({SEASON})')
    ui.caption(f'Your workbook minutes for each team’s first {OPENING_GAMES} games; season-average minutes from game {OPENING_GAMES + 1}. Counts include completed regular-season team games, not player appearances.')
    if ui.button('Refresh live data'):
        live_state.clear()
        today_schedule.clear()
    with ui.spinner('Refreshing team records and season averages…'):
        try:
            state = live_state(SEASON)
            teams, players = state['teams'], state['players']
            ui.caption(f"Live data checked: {state['as_of']} · ESPN regular-season data · refreshes every 5 minutes while open")
            if state['errors']:
                ui.warning('Some current-season player summaries could not be loaded. Affected players may lack season-average minutes; unavailable values stay blank.')
                with ui.expander('Data refresh details'):
                    ui.write(state['errors'])
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            ui.warning('Live team records are unavailable. Numerical projections are paused because the correct minutes window cannot be verified. Your archetypes remain available below.')
            teams = pd.DataFrame(columns=['TEAM_ID','GAMES_PLAYED'])
            players = pd.DataFrame()
            with ui.expander('Data refresh details'):
                ui.write(f'{type(exc).__name__}: {exc}')
    history = pd.read_csv(ROOT / 'historical_player_data.csv')
    supplement_path = ROOT / 'data' / 'prior_stat_supplements.csv'
    if supplement_path.exists():
        supplement = pd.read_csv(supplement_path)
        if not supplement.SEASON_ID.eq(int(SEASON[:4]) + 19999).all():
            raise ValueError('Prior-stat supplement has the wrong source season')
        history = pd.concat([history,supplement[~supplement.PLAYER_ID.isin(history.PLAYER_ID)]],ignore_index=True)
    fallback = None
    stored_fallback = ROOT / 'data' / 'rotowire_fallback.csv'
    if stored_fallback.exists():
        fallback = pd.read_csv(stored_fallback)
    with ui.expander('RotoWire baseline for players without historical stats'):
        ui.write('RotoWire supplies statistical rates only for missing historical profiles. Your workbook minutes still apply for games 1–3, then season-average minutes from game 4. An uploaded CSV applies to this session and is not saved publicly.')
        upload = ui.file_uploader('RotoWire 2026–27 season projections CSV',type=['csv'],key='rotowire_csv')
        if upload is not None:
            units = ui.selectbox('Export stat units',['Per game','Season totals','Per 36 minutes'],key='rotowire_units')
            try:
                fallback, unmatched = normalize_rotowire(read_rotowire_csv(upload),roster,
                    {'Per game':'per_game','Season totals':'season_totals','Per 36 minutes':'per_36'}[units])
                ui.caption(f'{len(fallback)} player rows matched; {len(unmatched)} unmatched rows ignored. Only missing historical profiles use this fallback.')
            except (ValueError, KeyError) as exc:
                ui.error(str(exc))
                fallback = None
        elif fallback is not None:
            ui.caption(f'{len(fallback)} RotoWire baseline rows loaded from the October 16, 2026 export; only missing historical profiles use them.')
        elif fallback is None:
            ui.caption('No RotoWire export has been loaded yet. Missing-profile projections stay unavailable until an authorized export is supplied.')
    projections = build_projections(roster, history, teams, players, opening_games=OPENING_GAMES, fallback=fallback)
    date = datetime.now(ZoneInfo('America/Denver')).date()
    try:
        schedule = today_schedule(SEASON, date)
    except (requests.RequestException, ValueError, KeyError, TypeError):
        schedule = pd.DataFrame(columns=['TEAM_ID','Opponent','Venue','GameTimeUTC','GameStatus'])
        ui.caption('Today’s schedule is unavailable; the table still shows per-game baseline estimates.')
    projections = projections.merge(schedule, on='TEAM_ID', how='left', validate='many_to_one') if not schedule.empty else projections.assign(Opponent='', Venue='', GameTimeUTC='', GameStatus='')
    projection_tab, archetype_tab = ui.tabs(['Player Stat Projections', 'Archetypes'])
    with projection_tab:
        ui.info('Per-game baseline estimates: last year’s archived rates, or RotoWire rates for missing profiles, scale to the selected minutes. Current-season rates gradually replace that prior using a 200-minute prior weight. No opponent, pace, injury or starting-lineup adjustments are applied.')
        team = ui.selectbox('Projection team', ['All'] + sorted(roster.TEAM_ABBREVIATION.unique()), key='projection_team')
        selected = projections if team == 'All' else projections[projections.Team.eq(team)]
        if ui.checkbox('Only players with positive projected minutes', value=True, key='projection_rotation'):
            selected = selected[selected['Projected minutes'].gt(0) | selected['Projected minutes'].isna()]
        if ui.checkbox('Only today’s regular-season games', value=False, key='projection_today'):
            selected = selected[selected.Opponent.ne('') & selected.Opponent.notna()]
        if schedule.empty:
            ui.caption(f'No verified regular-season games on the loaded schedule for {date:%b %d}. These are per-game previews, not confirmed appearances.')
        display = ['Player','Team','Opponent','Team games completed','Projected minutes','Minutes source',
                   'PTS','REB','AST','FG3M','STL','BLK','TOV','PRA','Stat baseline','Status']
        ui.dataframe(selected[display].rename(columns={'FG3M':'3PM'}).round(2), hide_index=True, width='stretch')
        pending = selected[~selected.Status.eq('Available')]
        if not pending.empty:
            ui.caption(f'{len(pending)} displayed players have missing minutes or stat rates. Blank means unavailable, not zero. Manually assigning an archetype does not create a statistical profile.')
        ui.download_button('Download player projections CSV', selected.to_csv(index=False), 'player_stat_projections.csv', 'text/csv')
        with ui.expander('How these projections work'):
            ui.write('Projected stat = estimated stat per minute × projected minutes. Prior-season rates use observed historical minutes, never your projected minutes as their denominator. Current season rates receive weight equal to current minutes / (current minutes + 200). Players without a historical profile use their observed current-season rates once available.')
            ui.write('After a team completes its third game, minutes equal the player’s current regular-season minutes divided by appearances. DNP games do not count as appearances. A player with no matched appearances has unavailable minutes after the opening window. Workbook zeroes remain zero during the first three team games.')
            ui.write('ESPN team records are authoritative for the window. Name matching is exact after punctuation and diacritic normalization; ambiguous player matches are left unavailable. Traded players use the largest available season-summary sample rather than adding potentially overlapping team summaries.')
    with archetype_tab:
        ui.caption('Approved 2026–27 archetypes and the original workbook minutes. Manual overrides remain authoritative.')
        team = ui.selectbox('Team', ['All'] + sorted(roster.TEAM_ABBREVIATION.unique()), key='archetype_team')
        selected = roster if team == 'All' else roster[roster.TEAM_ABBREVIATION.eq(team)]
        if ui.checkbox('Only players with projected minutes', value=True, key='archetype_rotation'):
            selected = selected[selected.PROJECTED_MIN.gt(0)]
        columns = ['PLAYER_NAME','TEAM_ABBREVIATION','PROJECTED_MIN','OffArchetype','DefArchetype',
                   'OffAssignmentSource','DefAssignmentSource','HasHistoricalProfile']
        ui.dataframe(selected[columns], hide_index=True, width='stretch')
        ui.download_button('Download archetypes CSV', selected.to_csv(index=False), 'preseason_archetypes.csv', 'text/csv')
