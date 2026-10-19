"""Streamlit player-stat projection view; no betting or odds dependency."""
from datetime import datetime
from pathlib import Path
import tempfile
import hashlib
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

from live_projections import (ESPN, build_projections, fetch_json, load_live_state, parse_schedule)
from season_pipeline import ROOT, SEASON, OPENING_GAMES, apply_roster_rules
from matchup_data import load_game_logs
from matchup_projections import assign_game_archetypes, matchup_effects, apply_matchup, HEAT_STATS, prior_season_profiles
from rotowire_fallback import normalize_rotowire, read_rotowire_csv
from FanduelScrape import get_odds_api_key
from sportsbook_lines import (BOOKS, MARKETS, OddsFeedError, read_feed, next_matchup,
                              player_quotes, compare_with_book, comparison_style)


@st.cache_data(ttl=300, show_spinner=False)
def live_state(season):
    return load_live_state(season)


@st.cache_data(ttl=300, show_spinner=False)
def today_schedule(season, date):
    payload = fetch_json(f'{ESPN}/site/v2/sports/basketball/nba/scoreboard?dates={date:%Y%m%d}')
    return parse_schedule(payload, int(season[:4]) + 1)


@st.cache_data(ttl=300, show_spinner=False)
def current_matchup_logs(year, team_ids):
    cache = Path(tempfile.gettempdir()) / f'nba-matchup-boxes-{year}'
    cache.mkdir(exist_ok=True)
    return load_game_logs(year, team_ids, cache)


@st.cache_data(show_spinner=False)
def prior_matchup_effects(roster, role_policy_version):
    logs = pd.read_csv(ROOT / 'data' / 'game_logs_2026.csv.gz', dtype={'GAME_ID':str,'ESPN_PLAYER_ID':str})
    return matchup_effects(assign_game_archetypes(logs, roster))


def heat_color(value):
    if pd.isna(value):
        return ''
    color = '#1f9d45' if value >= 10 else '#b9efbc' if value >= 0 else '#d92f45' if value <= -10 else '#f8d7da'
    foreground = 'white' if abs(value) >= 10 else '#202124'
    return f'background-color: {color}; color: {foreground}'


@st.cache_data(ttl=300, show_spinner=False)
def sportsbook_events(_api_key):
    events = read_feed('/events',_api_key)
    if not isinstance(events,list):
        raise OddsFeedError('Sportsbook event list is unavailable.')
    return events


@st.cache_data(ttl=300, show_spinner=False)
def sportsbook_props(event_id, bookmaker, _api_key):
    payload = read_feed(f'/events/{event_id}/odds',_api_key,
                        bookmakers=bookmaker,markets=','.join(MARKETS),oddsFormat='american')
    if not isinstance(payload,dict):
        raise OddsFeedError('Sportsbook prices are unavailable.')
    return payload


def mark_manual_minutes(key, slider_key):
    st.session_state[key] = st.session_state[slider_key]


def reset_manual_minutes(flag_key, slider_key):
    st.session_state.pop(flag_key, None)
    st.session_state.pop(slider_key, None)


@st.fragment(run_every=300)
def render_live_dashboard(ui, roster):
    roster = apply_roster_rules(roster)
    ui.title(f'🏀 Player Stat Projections ({SEASON})')
    ui.caption(f'Your workbook minutes for each team’s first {OPENING_GAMES} games; season-average minutes from game {OPENING_GAMES + 1}. Counts include completed regular-season team games, not player appearances.')
    if ui.button('Refresh live data'):
        live_state.clear()
        today_schedule.clear()
        current_matchup_logs.clear()
        sportsbook_events.clear()
        sportsbook_props.clear()
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
    full_prior = prior_season_profiles(pd.read_csv(ROOT / 'data' / 'game_logs_2026.csv.gz',dtype={'ESPN_PLAYER_ID':str}),roster)
    history = pd.concat([full_prior,history[~history.PLAYER_ID.isin(full_prior.PLAYER_ID)]],ignore_index=True)
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
    role_policy_version = hashlib.sha256(b''.join((ROOT / p).read_bytes() for p in ['archetype_reviews.csv','archetype_reference_reviews.csv','archetype_rotation_reviews.csv','archetype_overrides.json','archetype_model.py'])).hexdigest()
    prior_effects = prior_matchup_effects(roster, role_policy_version)
    current_effects = {}
    if not teams.empty and teams.GAMES_PLAYED.ge(4).any():
        with ui.spinner('Refreshing current-season matchup history…'):
            try:
                logs, details = current_matchup_logs(int(SEASON[:4])+1, tuple(teams.ESPN_TEAM_ID))
                current_effects = matchup_effects(assign_game_archetypes(logs,roster))
                if details['errors']:
                    ui.warning('Some current-season matchup games could not be loaded. Sample counts may be incomplete.')
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                ui.warning(f'Current-season matchup data unavailable: {exc}')
    projection_tab, best_tab, archetype_tab = ui.tabs(['Player vs Opponent', "🔥 Today's Best Matchups", 'Archetypes'])
    ui.caption('Last season’s stat rates for team games 1–4; current-season rates blend in from game 5. Opponent tendencies switch to this season after that opponent completes four games. Matchup samples are player-games with at least 5 minutes.')
    with projection_tab:
        team_col, player_col, opponent_col = ui.columns(3)
        with team_col:
            team = ui.selectbox('Player team', sorted(roster.TEAM_ABBREVIATION.unique()), key='projection_team')
        team_players = projections[projections.Team.eq(team)]
        names = team_players.sort_values('Projected minutes',ascending=False).Player.tolist()
        with player_col:
            player_name = ui.selectbox('Player',names,key='projection_player')
        with opponent_col:
            opponent = ui.selectbox('Opponent',[t for t in sorted(roster.TEAM_ABBREVIATION.unique()) if t != team],key='projection_opponent')
        adjusted_on = ui.toggle('Apply opponent matchup adjustments',value=True,key='opponent_adjustments')
        baseline = team_players[team_players.Player.eq(player_name)]
        player_id = int(baseline.iloc[0].PLAYER_ID)
        automatic_minutes = baseline.iloc[0]['Projected minutes']
        default_minutes = min(48.,max(0.,float(automatic_minutes))) if pd.notna(automatic_minutes) else 0.
        slider_key = f'player_minutes_{SEASON}_{player_id}'
        flag_key = f'custom_minutes_{SEASON}_{player_id}'
        st.session_state[slider_key] = st.session_state.get(flag_key,default_minutes)
        custom_minutes = ui.slider('Projected minutes',min_value=0.,max_value=48.,step=0.5,
                                   key=slider_key,format='%.1f min',on_change=mark_manual_minutes,args=(flag_key,slider_key))
        ui.button('Use automatic minutes',key=f'reset_minutes_{player_id}',on_click=reset_manual_minutes,
                  args=(flag_key,slider_key))
        if flag_key in st.session_state:
            baseline = build_projections(roster[roster.PLAYER_ID.eq(player_id)],history,teams,players,
                opening_games=OPENING_GAMES,fallback=fallback,minutes_overrides={player_id:custom_minutes})
            ui.caption('Custom minutes apply to this player’s projection in your current session. Use automatic minutes to return to the workbook/season-average setting.')
        if pd.isna(automatic_minutes):
            ui.caption('Automatic minutes are unavailable. Adjust the slider to set your own minutes.')
        else:
            ui.caption(f'Automatic setting: {automatic_minutes:.1f} minutes ({team_players.loc[team_players.Player.eq(player_name),"Minutes source"].iloc[0]}).')
        adjusted = apply_matchup(baseline,roster,teams,prior_effects,current_effects,opponent)
        row = adjusted.iloc[0]
        ui.subheader(f'{player_name} vs {opponent}')
        ui.caption(f"{row['Offensive archetype']} · {row['Defensive archetype']} · {row['Projected minutes']:.1f} projected minutes ({row['Minutes source']})")
        ui.caption(f"Stat baseline: {row['Stat baseline']} · Opponent history: {row['Matchup data']} · Offensive samples: {row['Off samples']} · Defensive samples: {row['Def samples']}")
        if row['Status'] != 'Available':
            ui.warning(row['Status'])
        if row['Matchup status'] != 'Available':
            ui.warning(row['Matchup status'] + '. Unavailable adjustments stay blank; turn off matchup adjustments to view the baseline.')
        stats = ['PTS','REB','AST','FG3M','STL','BLK','TOV','PRA']
        preview = pd.DataFrame({'Stat':['3PM' if x=='FG3M' else x for x in stats],
            'Baseline':[baseline.iloc[0][x] for x in stats],
            'Projected':[row[x] if adjusted_on else baseline.iloc[0][x] for x in stats]})
        preview['Change'] = preview.Projected - preview.Baseline
        book_name = ui.selectbox('Sportsbook',list(BOOKS),key='projection_sportsbook')
        book_key = BOOKS[book_name]
        quotes = player_quotes({},player_name,book_key)
        try:
            api_key = get_odds_api_key()
            if not api_key:
                raise OddsFeedError('Sportsbook feed access is not configured.')
            event = next_matchup(sportsbook_events(api_key),team,opponent)
            if event is None:
                ui.caption(f'No upcoming {team} vs {opponent} sportsbook event was found. Choose an opponent with a scheduled game to compare current lines.')
            else:
                payload = sportsbook_props(event['id'],book_key,api_key)
                if payload.get('id') != event['id'] or {payload.get('home_team'),payload.get('away_team')} != {event['home_team'],event['away_team']}:
                    raise OddsFeedError('Sportsbook matchup could not be verified.')
                quotes = player_quotes(payload,player_name,book_key)
                game_time = pd.Timestamp(event['commence_time']).tz_convert('America/Denver')
                ui.caption(f'{book_name} via The Odds API · {team} vs {opponent} · {game_time:%b %d, %Y at %I:%M %p %Z}. Quotes refresh every 5 minutes; quotes older than 15 minutes are excluded from comparison.')
        except OddsFeedError as exc:
            ui.caption(str(exc))
        comparison = compare_with_book(preview,quotes)
        display_columns = ['Stat','Baseline','Projected','Book line','Book implied total','Edge %','Play','Change']
        display = comparison[display_columns]
        ui.caption('Edge % = (projection ÷ book-implied total − 1) × 100. Over plays require more than +5%; Under plays require less than −5%. Only qualifying edges are shaded, from light to dark as the edge grows. The 5% buffer is your extra projection margin; the implied total already removes bookmaker margin.')
        ui.dataframe(display.style.apply(comparison_style,axis=1).format(
            {**{column:'{:.2f}' for column in display_columns if column not in ['Stat','Play','Edge %']},'Edge %':'{:+.2f}%'},na_rep='—'),hide_index=True,width='stretch')
        unavailable = comparison.loc[comparison['Book implied total'].isna(),'Stat'].tolist()
        if unavailable:
            ui.caption('No usable current two-sided line for: '+', '.join(unavailable)+'. Unavailable comparisons stay blank.')
        with ui.expander('Sportsbook prices and implied-total calculation'):
            ui.dataframe(comparison[['Stat','Book line','Over odds','Under odds','Fair over probability','Book implied total','Updated UTC','Market status']],hide_index=True,width='stretch')
            ui.write('American odds convert to implied probabilities. Fair P(over) = raw P(over) / [raw P(over) + raw P(under)]. The implied total is the Poisson mean whose over probability matches that fair probability; integer lines account for a push. One line and its prices do not uniquely identify the real stat distribution, so this is an explicitly modeled mean, not a sportsbook-published forecast. Minutes changes update your projection; the market estimate remains tied to the book’s game and prices.')
            ui.write('For illustration only: over 2.5 threes at -150 with under +120 implies a mean above 2.5. Both prices are needed to remove margin; the over price alone is insufficient.')
            ui.download_button('Download stat and sportsbook comparison',comparison.to_csv(index=False),'player_sportsbook_comparison.csv','text/csv')
        effect_cols = [f'{stat} %' for stat in HEAT_STATS]
        ui.dataframe(adjusted[effect_cols].style.map(heat_color).format('{:+.1f}%',na_rep='—'),hide_index=True,width='stretch')
        ui.download_button('Download this projection CSV',(adjusted if adjusted_on else baseline).to_csv(index=False),'player_opponent_projection.csv','text/csv')
        with ui.expander('How matchup projections work'):
            ui.write('Each archetype’s per-36 production against the selected opponent is compared with its league-wide per-36 average. Percent changes are weighted toward recent player-games and shrunk toward zero with three neutral samples. Offensive archetypes drive points, assists, offensive rebounds and shooting; defensive archetypes drive defensive rebounds, steals and blocks. Projected rebounds equal offensive plus defensive rebounds. Three-point makes use the attempts adjustment to preserve the player’s own shooting rate.')
            ui.write('Minutes use your workbook for team games 1–3, then observed season minutes per appearance from game 4. Stat rates stay on last season (or the supplied RotoWire fallback) through game 4; from game 5, current rates blend with the prior using a 200-minute prior weight. Opponent matchup history uses last season until that opponent has completed four games, then only this season. Missing history stays unavailable. Samples count player-games, not unique team games.')
            ui.write('These are matchup estimates, not confirmed lineups or availability. The model does not adjust for pace, injuries, starters or home court. Last season’s box scores are assigned the approved current archetypes, preserving manual overrides. Only players with matched archetype labels contribute to the matchup samples; sample counts show that coverage.')
    with best_tab:
        ui.subheader("🔥 Today's Best Matchup Opportunities")
        date_col, rank_col, sample_col = ui.columns(3)
        with date_col:
            date = ui.date_input('Slate date',value=datetime.now(ZoneInfo('America/Denver')).date(),key='matchup_date')
        try:
            schedule = today_schedule(SEASON,date)
        except (requests.RequestException, ValueError, KeyError, TypeError):
            schedule = pd.DataFrame()
            ui.warning('The selected day’s schedule could not be loaded.')
        if schedule.empty:
            ui.info(f'No upcoming regular-season games were found for {date:%b %d, %Y}. Choose another date to preview its matchups.')
        else:
            slate = projections.merge(schedule,on='TEAM_ID',how='inner',validate='many_to_one')
            slate = slate[slate['Projected minutes'].gt(0)]
            matchups = apply_matchup(slate,roster,teams,prior_effects,current_effects)
            with rank_col:
                ranked_stat = ui.selectbox('Rank matchups by',HEAT_STATS,key='matchup_rank')
            with sample_col:
                minimum = ui.number_input('Minimum samples per archetype',min_value=1,max_value=500,value=1,step=1,key='matchup_samples')
            include_negative = ui.checkbox('Show negative matchups too',value=True,key='negative_matchups')
            eligible = matchups[matchups['Off samples'].ge(minimum) & matchups['Def samples'].ge(minimum)]
            eligible = eligible[eligible[f'{ranked_stat} %'].notna()]
            if not include_negative:
                eligible = eligible[eligible[f'{ranked_stat} %'].gt(0)]
            eligible = eligible.sort_values(f'{ranked_stat} %',ascending=False)
            ui.success(f'Found {len(eligible)} players with matchup history')
            ui.caption('Home/away and opponent come from the selected day’s schedule. Percentages compare archetype production against that opponent with the archetype’s usual production. An attractive matchup is not an availability confirmation.')
            columns = ['Player','Team','Venue','Opponent','Offensive archetype','Defensive archetype',*[f'{x} %' for x in HEAT_STATS],'Off samples','Def samples','Matchup data']
            display = eligible[columns]
            formats = {f'{x} %':'{:+.1f}%' for x in HEAT_STATS}
            ui.dataframe(display.style.map(heat_color,subset=list(formats)).format(formats,na_rep='—'),hide_index=True,width='stretch')
            with ui.expander('Projected stat lines for these matchups',expanded=True):
                ui.dataframe(eligible[['Player','Team','Opponent','Projected minutes','PTS','REB','AST','FG3M','STL','BLK','TOV','PRA','Stat baseline','Status']].rename(columns={'FG3M':'3PM'}).round(2),hide_index=True,width='stretch')
            ui.download_button('Download matchup opportunities CSV',eligible.to_csv(index=False),'matchup_opportunities.csv','text/csv')
    with archetype_tab:
        with ui.expander('Review the 100 corrected players'):
            audit = pd.read_csv(ROOT / 'data/audit/reviewed_100_players.csv')
            ui.write('All 100 previously uncategorized or unnamed assignments have been reviewed. Your manual overrides take precedence. Developing roles remain provisional.')
            ui.dataframe(audit[['Player','Team','Offense','Defense','Confidence']],hide_index=True,width='stretch')
            ui.download_button('Download the full 100-player audit',audit.to_csv(index=False),'archetype_audit_100_players.csv','text/csv')
        with ui.expander('Review the 33 additional rotation players'):
            rotation_audit = pd.read_csv(ROOT / 'data/audit/low_minute_33_players.csv')
            ui.write('Roles are available even at 10 minutes or fewer. Existing manual assignments take precedence; developing roles remain provisional.')
            ui.dataframe(rotation_audit[['PLAYER_NAME','TEAM_ABBREVIATION','OffArchetype','DefArchetype','Confidence']],hide_index=True,width='stretch')
            ui.download_button('Download the 33-player assignments',rotation_audit.to_csv(index=False),'additional_33_archetypes.csv','text/csv')
        ui.caption('Approved 2026–27 archetypes and the original workbook minutes. Manual overrides remain authoritative.')
        team = ui.selectbox('Team', ['All'] + sorted(roster.TEAM_ABBREVIATION.unique()), key='archetype_team')
        selected = roster if team == 'All' else roster[roster.TEAM_ABBREVIATION.eq(team)]
        if ui.checkbox('Only players with projected minutes', value=True, key='archetype_rotation'):
            selected = selected[selected.PROJECTED_MIN.gt(0)]
        columns = ['PLAYER_NAME','TEAM_ABBREVIATION','PROJECTED_MIN','OffArchetype','DefArchetype',
                   'OffAssignmentSource','DefAssignmentSource','HasHistoricalProfile']
        ui.dataframe(selected[columns], hide_index=True, width='stretch')
        ui.download_button('Download archetypes CSV', selected.to_csv(index=False), 'preseason_archetypes.csv', 'text/csv')
