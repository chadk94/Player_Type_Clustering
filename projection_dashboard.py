"""Streamlit player-stat projections, opponent previews and sportsbook comparison."""
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import hashlib
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

from live_projections import (ESPN, build_projections, fetch_json, load_live_state, parse_schedule, next_regular_season_game, round_projected_minutes)
from season_pipeline import ROOT, SEASON, OPENING_GAMES, apply_roster_rules
from matchup_data import load_game_logs
from matchup_projections import assign_game_archetypes, matchup_effects, apply_matchup, HEAT_STATS, prior_season_profiles
from rotowire_fallback import normalize_rotowire, read_rotowire_csv
from FanduelScrape import get_odds_api_key
from market_points import market_targets, apply_market_points
from team_assists import historical_assist_rates, apply_team_assists
from team_threes import historical_three_rates, apply_team_threes
from team_rebounds import historical_rebound_rates, apply_team_rebounds, REBOUND_METADATA
from team_projection_view import upcoming_team_games, team_projection_tables, TEAM_STATS, TEAM_GAME_RETENTION
from sportsbook_edges import scheduled_book_games, game_edges, EDGE_STATS, EVENT_TIME_TOLERANCE
from projection_presentation import over_line_label, matchup_rating
from sportsbook_lines import (BOOKS, MARKETS, OddsFeedError, read_feed, next_matchup,
                              player_quotes, compare_with_book, comparison_style, TEAM_NAMES)


@st.cache_data(ttl=300, show_spinner=False)
def live_state(season):
    return load_live_state(season)


@st.cache_data(ttl=300, show_spinner=False)
def today_schedule(season, date):
    payload = fetch_json(f'{ESPN}/site/v2/sports/basketball/nba/scoreboard?dates={date:%Y%m%d}')
    return parse_schedule(payload, int(season[:4]) + 1)


@st.cache_data(ttl=300, show_spinner=False)
def next_team_game(season, espn_team_id, abbreviation):
    year = int(season[:4])+1
    payload = fetch_json(f'{ESPN}/site/v2/sports/basketball/nba/teams/{espn_team_id}/schedule?season={year}&seasontype=2')
    if str(payload.get('team',{}).get('id')) != str(espn_team_id):
        raise ValueError('Schedule returned the wrong team')
    return next_regular_season_game(payload,year,abbreviation)


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


@st.cache_data(ttl=300, show_spinner=False)
def sportsbook_game_lines(event_id, bookmaker, _api_key):
    return read_feed(f'/events/{event_id}/odds',_api_key,
                     bookmakers=bookmaker,markets='spreads,totals',oddsFormat='american')


@st.cache_data(show_spinner=False)
def prior_team_assist_rates():
    logs = pd.read_csv(ROOT / 'data' / 'game_logs_2026.csv.gz', dtype={'GAME_ID':str})
    return historical_assist_rates(logs, expected_games=82)


@st.cache_data(show_spinner=False)
def prior_team_three_rates():
    logs = pd.read_csv(ROOT / 'data' / 'game_logs_2026.csv.gz', dtype={'GAME_ID':str})
    return historical_three_rates(logs, expected_games=82)


@st.cache_data(show_spinner=False)
def prior_team_rebound_rates():
    logs = pd.read_csv(ROOT / 'data' / 'game_logs_2026.csv.gz', dtype={'GAME_ID':str})
    return historical_rebound_rates(logs, expected_games=82)


def team_projection_preview(ui, frame, event, book, api_key):
    try:
        targets = market_targets(sportsbook_game_lines(event['id'],book,api_key),event,book)
    except (OddsFeedError,ValueError,KeyError,TypeError,AttributeError):
        ui.caption('Team points, assist, three-point and rebound adjustments unavailable: no usable current spread/total for this sportsbook and game. Existing model estimates are retained.')
        return frame
    adjusted,summary,issues = apply_market_points(frame,targets)
    if not summary.empty:
        details = [f"{r.Team}: {r['Original PTS']:.2f} → {r['Market team total']:.2f} points ({(r.Factor-1)*100:+.2f}%)" for _,r in summary.iterrows()]
        ui.caption('Full market points adjustment · '+'; '.join(details)+'.')
    if issues:
        ui.caption('Points kept at model estimates for incomplete coverage: '+'; '.join(issues))
    try:
        home,away = TEAM_NAMES[event['home_team']],TEAM_NAMES[event['away_team']]
        adjusted,assist_summary,assist_issues = apply_team_assists(adjusted,targets,prior_team_assist_rates(),{home:away,away:home})
        if not assist_summary.empty:
            details = [f"{r.Team}: {r['Original AST']:.2f} → {r['Team AST target']:.2f} assists ({(r.Factor-1)*100:+.2f}%)" for _,r in assist_summary.iterrows()]
            ui.caption('Team assist adjustment · '+'; '.join(details)+'. Uses a 50/50 blend of last season’s team assist rate and opponent assist rate allowed. Player shares preserve relative archetype effects and sum to the team target.')
        if assist_issues:
            ui.caption('Assists kept at model estimates for incomplete coverage: '+'; '.join(assist_issues))
    except (ValueError,KeyError,TypeError,OSError):
        ui.caption('Team assist adjustment unavailable: historical team rates could not be verified. Existing assist estimates are retained.')
    try:
        home,away = TEAM_NAMES[event['home_team']],TEAM_NAMES[event['away_team']]
        adjusted,three_summary,three_issues = apply_team_threes(adjusted,targets,prior_team_three_rates(),{home:away,away:home})
        if not three_summary.empty:
            details = [f"{r.Team}: {r['Original 3PM']:.2f} → {r['Team 3PM target']:.2f} threes ({(r.Factor-1)*100:+.2f}%)" for _,r in three_summary.iterrows()]
            ui.caption('Team three-point adjustment · '+'; '.join(details)+'. Uses last season’s team and opponent three-point scoring mix. Player shares recalculate from minutes and normalized archetype effects; the team target comes from the game total and matchup rates.')
        if three_issues:
            ui.caption('Threes kept at model estimates: '+'; '.join(three_issues))
    except (ValueError,KeyError,TypeError,OSError):
        ui.caption('Team three-point adjustment unavailable: historical team rates could not be verified. Existing three-point estimates are retained.')
    try:
        home,away = TEAM_NAMES[event['home_team']],TEAM_NAMES[event['away_team']]
        own,allowed = prior_team_rebound_rates()
        adjusted,rebound_summary,rebound_issues = apply_team_rebounds(adjusted,targets,own,allowed,{home:away,away:home})
        if not rebound_summary.empty:
            details = [f"{r.Team}: {r['Original REB']:.2f} → {r['Team REB target']:.2f} rebounds ({(r['Team REB target']/r['Original REB']-1)*100:+.2f}%)" for _,r in rebound_summary.iterrows()]
            ui.caption('Team rebound adjustment · '+'; '.join(details)+'. Expected missed shots and historical rebound rates set the team targets. Offensive and defensive rebounds use separate normalized archetype shares.')
        if rebound_issues:
            ui.caption('Rebounds kept at model estimates: '+'; '.join(rebound_issues))
    except (ValueError,KeyError,TypeError,OSError):
        ui.caption('Team rebound adjustment unavailable: historical team rates could not be verified. Existing rebound estimates are retained.')
    ui.caption('PRA uses updated points + updated rebounds + updated assists. Threes adjust the scoring mix within those points. Matchup ratings show the original opponent effect before team adjustments.')
    return adjusted


def mark_manual_minutes(key, slider_key):
    st.session_state[key] = st.session_state[slider_key]


def reset_manual_minutes(flag_key, slider_key):
    st.session_state.pop(flag_key, None)
    st.session_state.pop(slider_key, None)


def set_edge_stats(selected):
    for stat in EDGE_STATS:
        st.session_state[f'edges_stat_{stat}'] = selected


def render_team_projections(ui, projections, roster, teams, prior_effects, current_effects):
    ui.subheader('🏀 Team Projections')
    ui.caption('Both teams’ projected stat totals, using the same saved minutes and adjustments as Top Sportsbook Edges. Player sliders affect individual previews only.')
    ui.caption('Games stay available until one hour after their scheduled tip-off. The list updates when you use a control or refresh.')
    now = pd.Timestamp(datetime.now(timezone.utc))
    snapshots = st.session_state.setdefault('team_projection_snapshots', {})
    for key in list(snapshots):
        if snapshots[key]['expires'] <= now:
            del snapshots[key]
    date_col, book_col = ui.columns(2)
    with date_col:
        date = ui.date_input('Team projection date', value=datetime.now(ZoneInfo('America/Denver')).date(), key='team_projection_date')
    with book_col:
        book_name = ui.selectbox('Sportsbook for team projections', list(BOOKS), key='team_projection_book')
    try:
        schedule = today_schedule(SEASON, date)
        games = upcoming_team_games(schedule, now=now)
    except (requests.RequestException, ValueError, KeyError, TypeError):
        ui.warning('The game schedule could not be loaded. Use Refresh live data to retry.')
        return
    if not games:
        ui.info('No upcoming games or games within one hour of scheduled tip-off were found for this date. Choose another date.')
        return
    by_key = {game['key']: game for game in games}
    if st.session_state.get('team_projection_game') not in by_key:
        st.session_state['team_projection_game'] = games[0]['key']
    selected = ui.pills('Team matchup', list(by_key), selection_mode='single', key='team_projection_game',
                        format_func=lambda key: f"{by_key[key]['Away']} @ {by_key[key]['Home']} · {by_key[key]['GameTimeUTC'].tz_convert('America/Denver'):%I:%M %p %Z}")
    if selected is None:
        ui.info('Select a game to view its team projections.')
        return
    game = by_key[selected]
    snapshot_key = (selected, BOOKS[book_name])
    order = [game['Away'], game['Home']]
    base = projections[projections.Team.isin(order)].copy()
    rows = [apply_matchup(base[base.Team.eq(side)], roster, teams, prior_effects, current_effects, opponent)
            for side, opponent in [(game['Away'], game['Home']), (game['Home'], game['Away'])]
            if not base[base.Team.eq(side)].empty]
    if len(rows) != 2:
        ui.warning('Both teams need roster projections before a game total can be shown.')
        return
    before = pd.concat(rows, ignore_index=True)
    final = before.copy()
    ui.subheader(f"{game['Away']} @ {game['Home']} — {book_name}")
    with ui.expander('Adjustment details'):
        try:
            api_key = get_odds_api_key()
            if not api_key:
                raise OddsFeedError('Sportsbook feed access is not configured.')
            verified = scheduled_book_games(sportsbook_events(api_key), schedule)
            matches = [event for event in verified if event['Home'] == game['Home'] and event['Away'] == game['Away']
                       and abs(event['GameTimeUTC'] - game['GameTimeUTC']) <= EVENT_TIME_TOLERANCE]
            if len(matches) != 1:
                raise OddsFeedError('No unique sportsbook event matches this scheduled game.')
            event = {**matches[0], 'commence_time': matches[0]['GameTimeUTC'].isoformat()}
            final = team_projection_preview(ui, before, event, BOOKS[book_name], api_key)
            if 'Market team total' not in final or not final['Market team total'].notna().any():
                raise OddsFeedError('Pregame team targets are unavailable.')
            snapshots[snapshot_key] = {'base': base.copy(deep=True), 'before': before.copy(deep=True),
                                       'final': final.copy(deep=True), 'saved_at': now,
                                       'expires': game['GameTimeUTC'] + TEAM_GAME_RETENTION}
        except (OddsFeedError, requests.RequestException, ValueError, KeyError, TypeError):
            saved = snapshots.get(snapshot_key) if game['GameTimeUTC'] <= now else None
            if saved is not None:
                base, before, final = (saved[name].copy(deep=True) for name in ['base', 'before', 'final'])
                ui.caption(f"Showing the pregame projection saved at {saved['saved_at'].tz_convert('America/Denver'):%I:%M %p %Z}. In-game odds are not used.")
            else:
                final = before.copy()
                ui.caption('Pregame lines are unavailable and no pregame projection is saved in this session. Showing model projections without team-target adjustments.')
    totals, breakdown, issues = team_projection_tables(base, before, final, order)
    for issue in issues:
        ui.warning(issue)
    calibrated = breakdown['Team target'].notna()
    if not calibrated.all():
        ui.info('Some team targets are unavailable. Those rows retain the existing model projection; missing player data stays blank. See Adjustment details.')
    ui.dataframe(totals.style.format({'Minutes': '{:.1f}', **{stat: '{:.2f}' for stat in TEAM_STATS}}, na_rep='—'),
                 hide_index=True, width='stretch')
    if totals.PTS.notna().all():
        ui.caption(f"Combined projected points: {totals.PTS.sum():.2f}. Team totals sum the unrounded player projections.")
    ui.subheader('How the team adjustments change projections')
    ui.caption('Baseline uses player stat rates and saved minutes. After archetypes adds opponent effects. Team adjustment % compares the final projection with After archetypes.')
    ui.dataframe(breakdown.style.format({'Baseline': '{:.2f}', 'After archetypes': '{:.2f}',
                                        'Team target': '{:.2f}', 'Final projection': '{:.2f}',
                                        'Team adjustment %': '{:+.2f}%'}, na_rep='—'),
                 hide_index=True, width='stretch')
    ui.caption('PTS targets come from the selected sportsbook’s spread and total. AST, 3PM and REB targets are model estimates using last season’s team and opponent rates. Steals, blocks and turnovers retain their player matchup projections. PRA = final PTS + REB + AST.')
    with ui.expander('Player contributions and minutes'):
        active = final['Projected minutes'].gt(0) | final['Projected minutes'].isna()
        columns = ['Player', 'Team', 'Projected minutes', *TEAM_STATS.values(), 'OREB', 'DREB', 'Minutes source', 'Stat baseline', 'Status', 'Matchup status']
        display = final.loc[active, columns].sort_values(['Team', 'Projected minutes'], ascending=[True, False]).rename(columns={'FG3M': '3PM'})
        ui.dataframe(display.round(2), hide_index=True, width='stretch')
        ui.caption('Zero-minute players are excluded from totals. Minutes are rounded individually to 0.5, so roster totals can differ slightly from 240. Uploaded and published minutes changes recalculate these projections; the local workbook does not sync automatically.')
    with ui.expander('Where the assist and three-point targets come from'):
        inputs = ['Market team total', 'Expected team FGM', 'Team assist rate', 'Team three rate']
        if set(inputs).issubset(final):
            drivers = final.loc[final['Projected minutes'].gt(0), ['Team', *inputs]].drop_duplicates()
            drivers = drivers.rename(columns={'Market team total': 'Market points', 'Expected team FGM': 'Expected FGM',
                                              'Team assist rate': 'Blended assist rate', 'Team three rate': 'Blended three rate'})
            ui.dataframe(drivers.style.format({'Market points': '{:.2f}', 'Expected FGM': '{:.2f}',
                                               'Blended assist rate': '{:.1%}', 'Blended three rate': '{:.1%}'}, na_rep='—'),
                         hide_index=True, width='stretch')
        ui.write('Expected made field goals = sportsbook team points ÷ the average of last season’s team points per made field goal and the opponent’s points allowed per made field goal. Points include free throws.')
        ui.write('Assist target = expected field goals × the average of team AST/FGM and opponent AST allowed/FGM allowed. Three-point target uses the same method with 3PM/FGM. The three-point rate is the share of made baskets that are threes, not shooting percentage.')
        ui.write('Each target is distributed using the roster’s minutes and normalized archetype effects. The final player totals add up to the team target. Threes change the scoring mix within the points projection; they do not add points again.')
    with ui.expander('Rebound targets and opportunity pools'):
        fields = ['Team OREB target', 'Team DREB target', 'Team REB target', 'Expected FG misses',
                  'Expected FT misses', 'Player credited fraction', 'Player rebound pool', 'Offensive rebound share']
        if set(fields).issubset(final):
            rebound_inputs = final.loc[final['Projected minutes'].gt(0), ['Team', *fields]].drop_duplicates()
            ui.dataframe(rebound_inputs.style.format({field: '{:.1%}' if field in ['Player credited fraction', 'Offensive rebound share'] else '{:.2f}' for field in fields}, na_rep='—'),
                         hide_index=True, width='stretch')
        ui.write('Each row’s misses belong to that shooting team. Its credited rebound pool splits into its own offensive rebounds and the opponent’s defensive rebounds. A team’s defensive target therefore comes from the other team’s missed-shot pool.')
        ui.write('Misses per made basket, the fraction producing credited player rebounds, and offensive capture shares blend last season’s team and opponent rates equally. The credited fraction accounts approximately for non-reboundable free throws and dead-ball/team rebounds. Offensive and defensive targets are allocated separately using normalized player archetype shares. REB is their sum; PRA uses the updated rebounds.')
    ui.download_button('Download team projections', totals.to_csv(index=False),
                       f"{game['Away']}_{game['Home']}_team_projections.csv", 'text/csv', key='team_projection_download')
    ui.download_button('Download team adjustment breakdown', breakdown.to_csv(index=False),
                       f"{game['Away']}_{game['Home']}_team_adjustments.csv", 'text/csv', key='team_adjustment_download')


@st.fragment
def render_live_dashboard(ui, roster):
    roster = apply_roster_rules(roster)
    roster['PROJECTED_MIN'] = roster.PROJECTED_MIN.map(round_projected_minutes)
    ui.title(f'🏀 Player Stat Projections ({SEASON})')
    ui.caption(f'Your workbook minutes for each team’s first {OPENING_GAMES} games; season-average minutes from game {OPENING_GAMES + 1}. Counts include completed regular-season team games, not player appearances.')
    if ui.button('Refresh live data'):
        live_state.clear()
        today_schedule.clear()
        next_team_game.clear()
        current_matchup_logs.clear()
        sportsbook_events.clear()
        sportsbook_props.clear()
        sportsbook_game_lines.clear()
    with ui.spinner('Refreshing team records and season averages…'):
        try:
            state = live_state(SEASON)
            teams, players = state['teams'], state['players']
            ui.caption(f"Live data checked: {state['as_of']} · ESPN regular-season data · Snapshot stays fixed while you browse. Use Refresh live data to update.")
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
    role_policy_version = hashlib.sha256(b''.join((ROOT / p).read_bytes() for p in ['archetype_reviews.csv','archetype_reference_reviews.csv','archetype_rotation_reviews.csv','archetype_overrides.json','archetype_model.py','matchup_projections.py','matchup_data.py'])).hexdigest()
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
    projection_tab, best_tab, edges_tab, team_tab, archetype_tab = ui.tabs(['Player vs Opponent', "🔥 Today's Best Matchups", '💰 Top Sportsbook Edges', '🏀 Team Projections', 'Archetypes'])
    ui.caption('Last season’s stat rates for team games 1–4; current-season rates blend in from game 5. Opponent tendencies switch to this season after that opponent completes four games. Matchup samples are player-games with at least 5 minutes.')
    with projection_tab:
        team_col, player_col, opponent_col = ui.columns(3)
        with team_col:
            team = ui.selectbox('Player team', sorted(roster.TEAM_ABBREVIATION.unique()), key='projection_team')
        team_players = projections[projections.Team.eq(team)]
        names = team_players.sort_values('Projected minutes',ascending=False).Player.tolist()
        with player_col:
            player_name = ui.selectbox('Player',names,key='projection_player')
        upcoming = None
        try:
            team_record = teams[teams.TEAM_ABBREVIATION.eq(team)] if 'TEAM_ABBREVIATION' in teams else teams.iloc[:0]
            if not team_record.empty:
                upcoming = next_team_game(SEASON,str(team_record.iloc[0].ESPN_TEAM_ID),team)
        except (requests.RequestException,ValueError,KeyError,TypeError):
            pass
        opponents = [t for t in sorted(roster.TEAM_ABBREVIATION.unique()) if t != team]
        # Initialize only when the team changes. Player/minutes/book reruns must
        # preserve an opponent the user deliberately selected.
        if st.session_state.get('projection_opponent_team') != team:
            st.session_state['projection_opponent'] = upcoming['Opponent'] if upcoming else None
            st.session_state['projection_opponent_team'] = team
        elif st.session_state.get('projection_opponent') not in opponents:
            st.session_state['projection_opponent'] = upcoming['Opponent'] if upcoming else None
        with opponent_col:
            opponent = ui.selectbox('Opponent',opponents,index=None,key='projection_opponent',placeholder='Choose opponent')
        if upcoming:
            next_time = pd.Timestamp(upcoming['GameTimeUTC']).tz_convert('America/Denver')
            ui.caption(f'Next scheduled game: {team} vs {upcoming["Opponent"]} · {next_time:%b %d at %I:%M %p %Z}. You can change the opponent for any matchup preview.')
        else:
            ui.caption('The next regular-season opponent could not be verified. Choose an opponent manually.')
        if opponent is None:
            ui.info('Choose an opponent to view this player’s projection.')
        else:
            adjusted_on = ui.toggle('Apply opponent matchup adjustments',value=True,key='opponent_adjustments')
            baseline = team_players[team_players.Player.eq(player_name)]
            player_id = int(baseline.iloc[0].PLAYER_ID)
            automatic_minutes = baseline.iloc[0]['Projected minutes']
            default_minutes = min(48.,max(0.,float(automatic_minutes))) if pd.notna(automatic_minutes) else 0.
            slider_key = f'player_minutes_{SEASON}_{player_id}'
            flag_key = f'custom_minutes_{SEASON}_{player_id}'
            st.session_state[slider_key] = round_projected_minutes(st.session_state.get(flag_key,default_minutes))
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
            stats = ['PTS','REB','AST','FG3M','STL','BLK','TOV','PF','PRA']
            preview = pd.DataFrame({'Stat':['3PM' if x=='FG3M' else x for x in stats],
                'Baseline':[baseline.iloc[0][x] for x in stats],
                'Projected':[row[x] if adjusted_on else baseline.iloc[0][x] for x in stats]})
            preview['Matchup rating'] = [matchup_rating(base,projected,stat,adjusted_on) for base,projected,stat in zip(preview.Baseline,preview.Projected,preview.Stat)]
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
                    team_baselines = team_players.copy()
                    if flag_key in st.session_state:
                        for column in baseline.columns:
                            if column in team_baselines:
                                team_baselines.loc[team_baselines.PLAYER_ID.eq(player_id),column] = baseline.iloc[0][column]
                    full_team = apply_matchup(team_baselines,roster,teams,prior_effects,current_effects,opponent) if adjusted_on else team_baselines
                    market_team = team_projection_preview(ui,full_team,event,book_key,api_key)
                    market_player = market_team[market_team.PLAYER_ID.eq(player_id)].iloc[0]
                    for stat in ['PTS','AST','FG3M','FG3A','OREB','DREB','REB','PRA']:
                        display_stat = '3PM' if stat == 'FG3M' else stat
                        preview.loc[preview.Stat.eq(display_stat),'Projected'] = market_player[stat]
                        if adjusted_on:
                            adjusted.loc[adjusted.PLAYER_ID.eq(player_id),stat] = market_player[stat]
                        else:
                            baseline.loc[baseline.PLAYER_ID.eq(player_id),stat] = market_player[stat]
                    destination = adjusted if adjusted_on else baseline
                    for column in ['Pre-market PTS','Pre-market PRA','Market PTS factor','Market team total',
                                   'Pre-team AST','Team AST factor','Team AST target','Team assist rate','Expected team FGM','Assist share',
                                   'Pre-team 3PM','Pre-team 3PA','Team 3PM factor','Team 3PM target','Team three rate','Three-point share',*REBOUND_METADATA]:
                        if column in market_player:
                            destination.loc[destination.PLAYER_ID.eq(player_id),column] = market_player[column]
                    payload = sportsbook_props(event['id'],book_key,api_key)
                    if payload.get('id') != event['id'] or {payload.get('home_team'),payload.get('away_team')} != {event['home_team'],event['away_team']}:
                        raise OddsFeedError('Sportsbook matchup could not be verified.')
                    quotes = player_quotes(payload,player_name,book_key)
                    game_time = pd.Timestamp(event['commence_time']).tz_convert('America/Denver')
                    ui.caption(f'{book_name} via The Odds API · {team} vs {opponent} · {game_time:%b %d, %Y at %I:%M %p %Z}. Use Refresh live data for updated quotes; quotes older than 15 minutes are excluded when comparisons are loaded.')
            except OddsFeedError as exc:
                ui.caption(str(exc))
            preview['Change'] = preview.Projected - preview.Baseline
            comparison = compare_with_book(preview,quotes)
            comparison['Book line (Over)'] = [over_line_label(line,price) for line,price in zip(comparison['Book line'],comparison['Over odds'])]
            display_columns = ['Stat','Baseline','Matchup rating','Projected','Book line (Over)','Book implied total','Edge %','Play']
            display = comparison[display_columns]
            ui.caption('Edge % shows the size of the gap from the book-implied total, positive for both Overs and Unders. Play identifies the direction: Over when the projection is higher, Under when lower. Only gaps greater than 5% qualify as plays and receive shading, from light to dark as the edge grows. The 5% buffer is your extra projection margin; the implied total already removes bookmaker margin.')
            ui.dataframe(display.style.apply(comparison_style,axis=1).format(
                {**{column:'{:.2f}' for column in display_columns if column not in ['Stat','Play','Edge %','Matchup rating','Book line (Over)']},'Edge %':'{:+.2f}%'},na_rep='—'),hide_index=True,width='stretch')
            ui.caption('Matchup rating: 1 ⬇️⬇️ poor (−10% or worse), 2 ⬇️ below average, 3 ↔️ average (within ±3%), 4 ⬆️ good, 5 ⬆️⬆️ excellent (+10% or better). Based on opponent lift versus baseline, not sportsbook edge; fewer turnovers or fouls rate higher. Ratings are blank when adjustments are off or data is unavailable. Book line (Over) includes the Over price.')
            ui.caption('PF = personal fouls committed. Uses your projected minutes and the defensive-archetype matchup effect. Missing foul rates stay blank; no sportsbook foul line is connected.')
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
                ui.write('Team assists use last season’s assist-to-made-field-goal rates, blended equally with the opponent’s allowed rate. Expected field goals come from market team points divided by the equally blended historical total-points-to-field-goals ratios. The team assist total is allocated in proportion to existing archetype-adjusted player assists, so shared defensive penalties cancel while relative archetype differences remain. No second assist multiplier is applied. PRA combines the final points, rebounds and assists.')
                ui.write('Team threes use expected team field goals times a 50/50 blend of last season’s team 3PM/FGM and opponent 3PM/FGM allowed. That target is divided among players using their current minutes and normalized archetype-adjusted shooting shares. Supporting three-point attempts scale with makes to preserve shooting percentage. Points and PRA already include the threes; no extra points are added. The player slider recalculates this preview, while game edges use the saved automatic team minutes. Changes to an external workbook must be imported and published before the app can use them.')
                ui.write('Team rebounds use both teams’ expected misses, derived from market team points and last season’s own/opponent rates. Historical credited-player-rebound fractions account approximately for dead-ball/team rebounds and non-reboundable free throws. Each pool splits into offensive rebounds and opposing defensive rebounds. The two components are allocated separately using normalized archetype shares, then added for REB. PRA includes the updated rebounds. Player matchup ratings retain the original archetype effect.')
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
    with edges_tab:
        ui.subheader('💰 Top 20 Sportsbook Edges per Game')
        ui.caption('Ranked by absolute percent edge versus the juice-adjusted book total, including Overs and Unders. Each row is one player/stat. Uses automatic workbook/season-average minutes and opponent adjustments.')
        # Tabs execute together; only fetch props for the selected game, never the full slate.
        try:
            edge_key = get_odds_api_key()
            if not edge_key:
                raise OddsFeedError('Sportsbook feed access is not configured.')
            edge_events = sportsbook_events(edge_key)
            now = datetime.now(timezone.utc)
            future_dates = []
            for item in edge_events:
                try:
                    start = pd.Timestamp(item['commence_time'])
                    if start.tzinfo is not None and start > pd.Timestamp(now):
                        future_dates.append(start.tz_convert('America/Denver').date())
                except (KeyError,ValueError,TypeError):
                    continue
            date_col, book_col, direction_col = ui.columns(3)
            with date_col:
                edge_date = ui.date_input('Game date',value=min(future_dates) if future_dates else datetime.now(ZoneInfo('America/Denver')).date(),key='edges_date')
            with book_col:
                edge_book_name = ui.selectbox('Sportsbook for edges',list(BOOKS),key='edges_book')
            with direction_col:
                edge_direction = ui.selectbox('Edge direction',['Both','Over','Under'],key='edges_direction')
            player_search = ui.text_input(
                'Filter by player',key='edges_player_search',placeholder='Type a player’s name…',
                help='Search any part of a name in the selected game, then press Enter. Clear the field to show all players. Your stat and edge filters still apply.')
            ui.write('Stats to show')
            select_col,clear_col,_ = ui.columns([1,1,6])
            with select_col:
                ui.button('Select all',key='edges_select_all',on_click=set_edge_stats,args=(True,))
            with clear_col:
                ui.button('Clear all',key='edges_clear_all',on_click=set_edge_stats,args=(False,))
            selected_stats = []
            for stat_col,stat in zip(ui.columns(len(EDGE_STATS)),EDGE_STATS):
                with stat_col:
                    if ui.checkbox(stat,value=True,key=f'edges_stat_{stat}'):
                        selected_stats.append(stat)
            plays_only = ui.checkbox('Only show plays above 5% edge',value=True,key='edges_plays_only')
            edge_schedule = today_schedule(SEASON,edge_date)
            games = scheduled_book_games(edge_events,edge_schedule,now=now)
            if not games:
                ui.info('No upcoming regular-season games with a verified sportsbook event were found for this date. Choose another date. Lines are never substituted from a different game.')
            else:
                by_id = {game['id']:game for game in games}
                labels = {game['id']:f"{game['Away']} @ {game['Home']} · {game['GameTimeUTC'].tz_convert('America/Denver'):%I:%M %p %Z}" for game in games}
                if st.session_state.get('edges_game') not in by_id:
                    st.session_state['edges_game'] = games[0]['id']
                game_id = ui.pills('Choose a game',list(by_id),format_func=lambda value:labels[value],selection_mode='single',key='edges_game')
                if game_id is None:
                    ui.info('Select a game to see its top edges.')
                else:
                    game = by_id[game_id]
                    matchup_rows = []
                    for side,opponent in [(game['Home'],game['Away']),(game['Away'],game['Home'])]:
                        # Keep the whole roster so unavailable minutes cannot silently
                        # disappear before team-coverage checks. Rankings exclude inactive rows.
                        team_projection = projections[projections.Team.eq(side)]
                        if not team_projection.empty:
                            matchup_rows.append(apply_matchup(team_projection,roster,teams,prior_effects,current_effects,opponent))
                    payload = sportsbook_props(game_id,BOOKS[edge_book_name],edge_key)
                    combined = pd.concat(matchup_rows,ignore_index=True) if matchup_rows else pd.DataFrame()
                    if not combined.empty:
                        market_event = {**game,'commence_time':game['GameTimeUTC'].isoformat()}
                        combined = team_projection_preview(ui,combined,market_event,BOOKS[edge_book_name],edge_key)
                    ranked,total_edges = game_edges(combined,payload,game,BOOKS[edge_book_name],now=now,plays_only=plays_only,direction=edge_direction,stats=selected_stats,player_search=player_search)
                    ui.subheader(f"{game['Away']} @ {game['Home']} — {edge_book_name}")
                    ui.caption('Snapshot of paired Over/Under prices; quotes older than 15 minutes are excluded when loaded. Use Refresh live data to update. Edge % is a projection gap, not a win probability or expected betting return.')
                    if not selected_stats:
                        ui.info('Check at least one stat category to show edges.')
                    elif ranked.empty:
                        if player_search.strip():
                            ui.info('No eligible edges match this player search in the selected game. Check the name, clear the search, or turn off the 5% filter.')
                        else:
                            ui.info('No eligible edges meet these filters. Props may not be posted yet, quotes may be stale, or projections may lack matchup history. Try another game/book or turn off the 5% filter.')
                    else:
                        ui.success(f'Showing {len(ranked)} of {total_edges} eligible player-stat edges')
                        ranked['Book line (Over)'] = [over_line_label(line,price) for line,price in zip(ranked['Book line'],ranked['Over odds'])]
                        display_columns = ['Rank','Player','Team','Stat','Play','Projected minutes','Matchup rating','Projected','Book line (Over)','Book implied total','Edge %','Over odds','Under odds']
                        edge_display = ranked[display_columns]
                        ui.dataframe(edge_display.style.apply(comparison_style,axis=1).format(
                            {'Projected minutes':'{:.1f}','Projected':'{:.2f}',
                             'Book implied total':'{:.2f}','Edge %':'{:+.2f}%',
                             'Over odds':'{:+.0f}','Under odds':'{:+.0f}'},na_rep='—'),hide_index=True,width='stretch')
                        ui.caption('Matchup rating: 1 ⬇️⬇️ poor, 2 ⬇️ below average, 3 ↔️ average (within ±3%), 4 ⬆️ good, 5 ⬆️⬆️ excellent. Ratings measure opponent lift versus baseline, independent of the sportsbook edge or Over/Under play.')
                        ui.caption('Edge % is positive for both Overs and Unders; Play identifies the direction. Only edges greater than 5% are marked as plays. Color intensity grows with edge magnitude. Multiple stats for the same player may be correlated.')
                        ui.download_button('Download this game’s top edges',ranked.to_csv(index=False),f"{game['Away']}_{game['Home']}_top_edges.csv",'text/csv',key='edges_download')
                    with ui.expander('Game projection coverage'):
                        if not combined.empty:
                            ui.dataframe(combined[['Player','Team','Opponent','Projected minutes','Minutes source','Off samples','Def samples','Stat baseline','Matchup data','Status','Matchup status']],hide_index=True,width='stretch')
                        ui.caption('Players with zero/unavailable minutes, incomplete statistical profiles, or missing archetype matchup history do not enter the rankings. Manual minutes on Player vs Opponent affect that preview only.')
        except OddsFeedError as exc:
            ui.info(str(exc))
        except (requests.RequestException,ValueError,KeyError,TypeError):
            ui.warning('Game edge rankings are temporarily unavailable. Refresh live data to retry.')
    with team_tab:
        render_team_projections(ui, projections, roster, teams, prior_effects, current_effects)
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
        ui.caption('Approved 2026–27 archetypes and projected minutes rounded to the nearest 0.5. Manual overrides remain authoritative.')
        team = ui.selectbox('Team', ['All'] + sorted(roster.TEAM_ABBREVIATION.unique()), key='archetype_team')
        selected = roster if team == 'All' else roster[roster.TEAM_ABBREVIATION.eq(team)]
        if ui.checkbox('Only players with projected minutes', value=True, key='archetype_rotation'):
            selected = selected[selected.PROJECTED_MIN.gt(0)]
        columns = ['PLAYER_NAME','TEAM_ABBREVIATION','PROJECTED_MIN','OffArchetype','DefArchetype',
                   'OffAssignmentSource','DefAssignmentSource','HasHistoricalProfile']
        ui.dataframe(selected[columns], hide_index=True, width='stretch')
        ui.download_button('Download archetypes CSV', selected.to_csv(index=False), 'preseason_archetypes.csv', 'text/csv')
