"""Streamlit player-stat projections, opponent previews and sportsbook comparison."""
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import hashlib
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

from live_projections import (ESPN, build_projections, fetch_json, load_live_state, name_key, next_regular_season_game, round_projected_minutes)
from archetype_fallback import build_archetype_priors
from season_pipeline import ROOT, SEASON, OPENING_GAMES, apply_roster_rules
from matchup_data import load_game_logs
from matchup_projections import (assign_game_archetypes, matchup_effects, apply_matchup, HEAT_STATS,
                                 prior_season_profiles, MATCHUP_PRIOR_GAMES, MATCHUP_POLICY_VERSION)
from rotowire_fallback import normalize_rotowire, read_rotowire_csv
from FanduelScrape import get_odds_api_key
from market_points import market_targets, apply_market_points
from edge_minutes import add_minutes_cushion
from team_assists import apply_team_assists
from team_threes import apply_team_threes
from career_shooting import SHOOTING_POLICY_VERSION
from player_rate_priors import PLAYER_RATE_POLICY_VERSION
from team_rebounds import apply_team_rebounds, REBOUND_METADATA
from team_rate_blend import (season_rates, blend_team_rates, PRIOR_GAMES, POLICY_VERSION,
                            DEFENSIVE_PRIOR_GAMES, DEFENSIVE_FULL_CURRENT_GAMES)
from team_projection_view import upcoming_team_games, team_projection_tables, TEAM_STATS, TEAM_GAME_RETENTION
from slate_player_view import PLAYER_SORTS, rank_slate_players
from sportsbook_edges import scheduled_book_games, game_edges, EDGE_STATS, EVENT_TIME_TOLERANCE
from sportsbook_injuries import KEY_MINUTES, latest_report, slate_questionables
from edge_rating_table import render_edge_table
from projection_presentation import over_line_label, matchup_rating
from recent_role import recent_role_comparison, add_recent_columns, DISPLAY_COLUMNS as RECENT_COLUMNS, CAPTION as RECENT_CAPTION
from points_probability import (scoring_context, points_quote, game_points_probabilities, PROBABILITY_COLUMNS)
from points_tracking import snapshot as probability_snapshot, render_probability_tracking
from opponent_selection import render_opponent_selector
from player_navigation import (apply_pending_navigation, selected_game_context,
                               render_player_navigation, queue_player_selection, send_player_navigation)
from approved_minutes import render_import, for_matchup, projection_gaps, game_minutes_caption, unmatched_minutes_message
from import_minutes import TEAM_IDS
from schedule_feed import load_daily_schedule, ScheduleMemory
from sportsbook_lines import (BOOKS, MARKETS, OddsFeedError, read_feed, next_matchup,
                              player_quotes, compare_with_book, comparison_style, TEAM_NAMES)


@st.cache_data(ttl=300, show_spinner=False)
def live_state(season):
    return load_live_state(season)


@st.cache_data(ttl=240, show_spinner=False)
def edge_injury_report():
    try:
        return latest_report()
    except (requests.RequestException, ValueError, OSError):
        return None


@st.fragment(run_every='5m')
def render_edge_injuries(ui, games, selected_id, date, roster, projections, bundle):
    now = pd.Timestamp.now(tz='UTC')
    upcoming = [g for g in games if g['GameTimeUTC'] > now]
    if not upcoming:
        return
    ui.markdown(f'**Key questionable players** · {date:%b %d}')
    report = edge_injury_report()
    if report is None:
        ui.caption(f'Current injury status unavailable. [NBA injury reports](https://official.nba.com/nba-injury-report-{SEASON}-season/)')
        return
    stamp, url, teams_reported = report
    upcoming.sort(key=lambda g: g['id'] != selected_id)
    rows, pending, unverified = slate_questionables(upcoming, teams_reported, roster, projections, bundle)
    lines = []
    for game in upcoming:
        pair = f"{game['Away']}@{game['Home']}"
        players = sorted((r for r in rows if r[0] == pair), key=lambda r: (-r[4], r[2]))
        if players:
            names = [f"{name} ({team}, {mins:g} min)" if pd.notna(mins) else f"{name} ({team}, minutes unknown)"
                     for _, team, name, mins, _ in players]
            lines.append(f"**{game['Away']} @ {game['Home']}** · " + ' · '.join(names))
    if lines:
        ui.markdown('  \n'.join(lines))
    elif not pending and not unverified:
        ui.write('No key players listed as questionable on this slate.')
    if pending:
        ui.caption('Awaiting a verified team report · ' + ', '.join(pending))
    if unverified:
        ui.caption('Coverage to verify · ' + ' · '.join(unverified))
    older = ' · Latest quarter-hour update unconfirmed' if now - stamp >= pd.Timedelta(minutes=30) else ''
    ui.caption(f'Over {KEY_MINUTES} minutes in the workbook rotation or game projection. Minutes shown are for this game. '
               f'[NBA report · {stamp:%b %d, %I:%M %p %Z}]({url}){older} · Checks every 5 minutes.')


@st.cache_data(ttl=300, show_spinner=False)
def today_schedule(season, date):
    return load_daily_schedule(season, date)


@st.cache_resource(show_spinner=False)
def verified_schedules():
    return ScheduleMemory()


def schedule_for_view(ui, date, key):
    """Keep a recently verified slate visible while its live refresh is down."""
    memory = verified_schedules()
    using_backup = False
    try:
        schedule = today_schedule(SEASON, date)
        memory.remember(SEASON, date, schedule)
    except (requests.RequestException, ValueError, KeyError, TypeError):
        schedule = memory.recover(SEASON, date)
        using_backup = True
        if schedule is None:
            ui.warning(f'The schedule for {date:%b %d, %Y} is temporarily unavailable. Retry the schedule below.')
        else:
            checked = pd.Timestamp(schedule.attrs['schedule_checked_at']).tz_convert('America/Denver')
            ui.warning(f'Live schedule refresh is temporarily unavailable. Showing the schedule verified at '
                       f'{checked:%I:%M %p %Z} for {date:%b %d}. Recent schedule changes may not appear. '
                       'This backup expires after 30 minutes.')
    ui.button('Retry schedule' if using_backup else 'Reload schedule', key=key,
              on_click=today_schedule.clear, args=(SEASON, date),
              help='Refresh only this date’s schedule, without reloading player stats or your minutes.')
    if schedule is not None and not using_backup and schedule.attrs.get('schedule_checked_at'):
        checked = pd.Timestamp(schedule.attrs['schedule_checked_at']).tz_convert('America/Denver')
        ui.caption(f'Schedule verified: {checked:%b %d at %I:%M %p %Z} · ESPN')
    return schedule


@st.cache_data(ttl=300, show_spinner=False)
def team_schedule(season, espn_team_id):
    year = int(season[:4])+1
    payload = fetch_json(f'{ESPN}/site/v2/sports/basketball/nba/teams/{espn_team_id}/schedule?season={year}&seasontype=2')
    if str(payload.get('team',{}).get('id')) != str(espn_team_id):
        raise ValueError('Schedule returned the wrong team')
    return payload


def next_team_game(season, espn_team_id, abbreviation):
    # Recheck the selection window on every render, even for cached schedules.
    # Keep the current matchup for the same first hour as Team Projections.
    return next_regular_season_game(team_schedule(season, espn_team_id),
                                    int(season[:4])+1, abbreviation,
                                    post_tip_window=TEAM_GAME_RETENTION.to_pytimedelta())


@st.cache_data(ttl=300, show_spinner=False)
def current_matchup_logs(year, team_ids):
    """Complete team boxes, including rounded-zero-minute attempts (coverage v2)."""
    cache = Path(tempfile.gettempdir()) / f'nba-matchup-boxes-{year}'
    cache.mkdir(exist_ok=True)
    return load_game_logs(year, team_ids, cache)


@st.cache_data(show_spinner=False)
def prior_matchup_effects(roster, role_policy_version):
    logs = pd.read_csv(ROOT / 'data' / 'game_logs_2026.csv.gz', dtype={'GAME_ID':str,'ESPN_PLAYER_ID':str})
    return matchup_effects(assign_game_archetypes(logs, roster))


@st.cache_data(max_entries=4, show_spinner=False)
def current_matchup_effects(logs, roster, role_policy_version):
    """Reuse matchup effects and recent peer averages across all edge filters."""
    return matchup_effects(assign_game_archetypes(logs, roster), include_recent=True)


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
def prior_team_rates():
    logs = pd.read_csv(ROOT / 'data' / 'game_logs_2026.csv.gz', dtype={'GAME_ID':str})
    return season_rates(logs, expected_games=82)


def team_projection_preview(ui, frame, event, book, api_key, team_rates):
    try:
        targets = market_targets(sportsbook_game_lines(event['id'],book,api_key),event,book)
    except (OddsFeedError,ValueError,KeyError,TypeError,AttributeError):
        ui.caption('Team points, assist, three-point and rebound adjustments unavailable: no usable current spread/total for this sportsbook and game. Existing model estimates are retained.')
        return frame
    adjusted,summary,issues = apply_market_points(frame,targets)
    coverage = team_rates['coverage']
    home,away = TEAM_NAMES[event['home_team']],TEAM_NAMES[event['away_team']]
    opponents = {home:away,away:home}
    details = []
    for team in adjusted.Team.unique():
        if team not in coverage.index:
            continue
        row = coverage.loc[team]
        mask = adjusted.Team.eq(team)
        for field, source in [('Team rate current weight','Current weight'),('Team rate games','Verified games'),
                              ('Team rate prior games','Prior games'),('Team rate status','Status')]:
            adjusted.loc[mask,field] = row[source]
        opponent = opponents[team]
        allowed = coverage.loc[opponent]
        for field,source in [('Opponent allowed current weight','Defensive current weight'),
                             ('Opponent allowed prior games','Defensive prior games'),
                             ('Opponent allowed rate status','Defensive status')]:
            adjusted.loc[mask,field] = allowed[source]
        details.append(f"{team} own/conversion rates: {row['Current weight']:.1%} this season; "
                       f"{opponent} allowed AST/3PA/OREB rates: {allowed['Defensive current weight']:.1%} "
                       f"({allowed['Verified games']:g} verified games; {allowed['Defensive status']})")
    ui.caption('Season weights · ' + '; '.join(details) + '.')
    if not summary.empty:
        details = [f"{r.Team}: {r['Original PTS']:.2f} → {r['Market team total']:.2f} points ({(r.Factor-1)*100:+.2f}%)" for _,r in summary.iterrows()]
        ui.caption('Full market points adjustment · '+'; '.join(details)+'.')
    if issues:
        ui.caption('Points kept at model estimates for incomplete coverage: '+'; '.join(issues))
    try:
        home,away = TEAM_NAMES[event['home_team']],TEAM_NAMES[event['away_team']]
        adjusted,assist_summary,assist_issues = apply_team_assists(adjusted,targets,team_rates['assists'],{home:away,away:home})
        if not assist_summary.empty:
            details = [f"{r.Team}: {r['Original AST']:.2f} → {r['Team AST target']:.2f} assists ({(r.Factor-1)*100:+.2f}%)" for _,r in assist_summary.iterrows()]
            ui.caption('Team assist adjustment · '+'; '.join(details)+'. Uses a 50/50 mix of the progressively blended team and opponent assist rates. Player shares preserve relative archetype effects and sum to the team target.')
        if assist_issues:
            ui.caption('Assists kept at model estimates for incomplete coverage: '+'; '.join(assist_issues))
    except (ValueError,KeyError,TypeError,OSError):
        ui.caption('Team assist adjustment unavailable: historical team rates could not be verified. Existing assist estimates are retained.')
    try:
        home,away = TEAM_NAMES[event['home_team']],TEAM_NAMES[event['away_team']]
        adjusted,three_summary,three_issues = apply_team_threes(adjusted,targets,team_rates['threes'],{home:away,away:home})
        if not three_summary.empty:
            details = [f"{r.Team}: {r['Original 3PM']:.2f} → {r['Team 3PM target']:.2f} threes ({(r.Factor-1)*100:+.2f}%)" for _,r in three_summary.iterrows()]
            ui.caption('Team three-point adjustment · '+'; '.join(details)+'. Uses progressively blended 3PA/FGA rates to target attempts. Normalized archetype attempt shares and each player’s shooting percentage determine makes; changing minutes recalculates both.')
        if three_issues:
            ui.caption('Threes kept at model estimates: '+'; '.join(three_issues))
    except (ValueError,KeyError,TypeError,OSError):
        ui.caption('Team three-point adjustment unavailable: historical team rates could not be verified. Existing three-point estimates are retained.')
    try:
        home,away = TEAM_NAMES[event['home_team']],TEAM_NAMES[event['away_team']]
        own,allowed = team_rates['rebounds_own'],team_rates['rebounds_allowed']
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


def team_game_projection(ui, game, book_name, schedule, now, snapshots, projections, roster, teams, prior_effects, current_effects, team_rates, project_for_game, approved_bundle=None):
    if project_for_game is not None:
        projections = project_for_game(game['Home'], game['Away'], game['GameTimeUTC'])
    snapshot_key = (game['key'], BOOKS[book_name], POLICY_VERSION, MATCHUP_POLICY_VERSION, SHOOTING_POLICY_VERSION, PLAYER_RATE_POLICY_VERSION)
    order = [game['Away'], game['Home']]
    base = projections[projections.Team.isin(order)].copy()
    ui.caption(game_minutes_caption(base, order))
    notice = unmatched_minutes_message(approved_bundle, game['Away'], game['Home'], game['GameTimeUTC'])
    if notice:
        ui.warning(notice)
    rows = [apply_matchup(base[base.Team.eq(side)], roster, teams, prior_effects, current_effects, opponent)
            for side, opponent in [(game['Away'], game['Home']), (game['Home'], game['Away'])]
            if not base[base.Team.eq(side)].empty]
    if len(rows) != 2:
        ui.warning('Both teams need roster projections before a game total can be shown.')
        return
    before = pd.concat(rows, ignore_index=True)
    final = before.copy()
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
        final = team_projection_preview(ui, before, event, BOOKS[book_name], api_key, team_rates)
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
    return base, before, final


class ProjectionNotes:
    """Collect the shared model's explanations without drawing a panel per game."""
    def __init__(self):
        self.messages = []

    def caption(self, message):
        self.messages.append(message)

    warning = caption


@st.fragment
def render_slate_player_table(ui, frame, by_key, book_name, date, notes):
    # Searching, sorting and row clicks reuse this captured frame. They never fetch odds,
    # rebuild the slate, or rerun any other dashboard tab.
    ui.subheader('All player projections')
    player_search = ui.text_input(
        'Filter by player', key='team_players_search', placeholder='Type a player’s name…',
        help='Search any part of a name across all games, then press Enter. Clear the field to show all players. Sorting and the zero-minute filter still apply.')
    sort_col, zero_col = ui.columns([2, 1])
    with sort_col:
        sort_label = ui.selectbox('Rank players by', list(PLAYER_SORTS), key='team_players_sort')
    with zero_col:
        include_zero = ui.checkbox('Include zero-minute players', key='team_players_include_zero')
    ranked = rank_slate_players(frame, PLAYER_SORTS[sort_label], include_zero)
    if ranked.empty:
        ui.info('No players remain in the selected games. Games stay available until one hour after tip-off.')
        return
    if player_search.strip():
        query = name_key(player_search)
        ranked = ranked.loc[ranked['Player'].map(name_key).str.contains(query, regex=False) if query
                            else pd.Series(False, index=ranked.index)].reset_index(drop=True)
        if ranked.empty:
            ui.info('No players match your search. Try another name or clear the search to show all players.')
            return
    count = ranked['Game key'].nunique()
    ui.caption(f'{len(ranked)} players across {count} games · {date:%b %d, %Y} · {book_name}. '
               'Highest projections first. You can also click any stat column to sort. '
               'Click a player name to open that matchup.')
    missing = ranked[PLAYER_SORTS[sort_label]].isna().sum()
    if missing:
        ui.caption(f'{missing} players lack a projection for this category and appear at the bottom.')
    model_only = ranked.loc[ranked['Projected minutes'].gt(0) &
                            ranked['Projection source'].eq('Player model — no team target'), 'Game'].unique()
    if len(model_only):
        ui.info('Team targets are unavailable for some players in ' + ', '.join(model_only) +
                '. Those rows retain their player-model projections. See Projection coverage and sources.')
    columns = ['Rank', 'Player', 'Team', 'Opponent', 'Game', 'Projected minutes',
               *TEAM_STATS.values(), 'OREB', 'DREB', 'PF', 'Minutes source',
               'Projection source', 'Stat baseline', 'Status', 'Matchup status', 'Fallback details']
    display = ranked[[c for c in columns if c in ranked]].rename(columns={'FG3M': '3PM'})
    identity = ranked[['Game key', 'PLAYER_ID']].to_csv(index=False) + book_name + sort_label
    token = hashlib.sha256(identity.encode()).hexdigest()[:16]
    widget_key = f'all_game_players_{token}_{st.session_state.get("contribution_selection_revision", 0)}'
    ui.dataframe(display.round(2), hide_index=True, width='stretch', height=650, key=widget_key,
                 on_select=lambda: queue_player_selection(widget_key, ranked, by_key, book_name),
                 selection_mode=['single-row', 'single-cell'])
    send_player_navigation()
    ui.caption('Uses each game’s saved minutes and the same team adjustments as the single-game view. '
               'Unknown stats stay blank; zero-minute players are hidden unless included above. '
               'Rank uses unrounded values and keeps each player’s place across the slate when searching. Refresh live data updates projections and sportsbook targets.')
    ui.download_button('Download filtered player projections' if player_search.strip() else 'Download all player projections', display.to_csv(index=False),
                       f'all_player_projections_{date.isoformat()}.csv', 'text/csv', key='all_player_projections_download')
    with ui.expander('Projection coverage and sources'):
        ui.dataframe(pd.DataFrame(notes), hide_index=True, width='stretch')


def render_all_team_players(ui, games, book_name, date, schedule, now, snapshots,
                            projections, roster, teams, prior_effects, current_effects,
                            team_rates, project_for_game, approved_bundle):
    frames, notes = [], []
    with ui.spinner('Loading player projections across all games…'):
        for game in games:
            messages = ProjectionNotes()
            result = team_game_projection(messages, game, book_name, schedule, now, snapshots,
                projections, roster, teams, prior_effects, current_effects, team_rates, project_for_game, approved_bundle)
            label = f"{game['Away']} @ {game['Home']}"
            if result is None:
                notes.append({'Game': label, 'Coverage': 'Unavailable', 'Details': ' '.join(messages.messages)})
                continue
            base, before, final = result
            _, _, issues = team_projection_tables(base, before, final, [game['Away'], game['Home']])
            rows = final.copy()
            rows['Game key'], rows['Game'], rows['GameTimeUTC'] = game['key'], label, game['GameTimeUTC']
            rows['Opponent'] = rows.Team.map({game['Home']: game['Away'], game['Away']: game['Home']})
            market = rows.get('Market team total', pd.Series(float('nan'), index=rows.index)).notna()
            rows['Projection source'] = market.map({True: 'Market + team model', False: 'Player model — no team target'})
            frames.append(rows)
            notes.append({'Game': label, 'Coverage': 'Review data gaps' if issues else 'Available',
                          'Details': ' '.join([*issues, *messages.messages])})
    if not frames:
        ui.warning('Player projections could not be built for these games.')
        ui.dataframe(pd.DataFrame(notes), hide_index=True, width='stretch')
        return
    unavailable = [row['Game'] for row in notes if row['Coverage'] == 'Unavailable']
    if unavailable:
        ui.warning('Missing game projections: ' + ', '.join(unavailable))
    render_slate_player_table(ui, pd.concat(frames, ignore_index=True),
                             {g['key']: g for g in games}, book_name, date, notes)


def render_team_projections(ui, projections, roster, teams, prior_effects, current_effects, team_rates, project_for_game=None, approved_bundle=None):
    ui.subheader('🏀 Team Projections')
    ui.caption('Compare team totals or rank all players using the same saved minutes and adjustments as Top Sportsbook Edges. Player sliders affect individual previews only.')
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
    schedule = schedule_for_view(ui, date, 'team_schedule_reload')
    if schedule is None:
        return
    try:
        games = upcoming_team_games(schedule, now=now)
    except (requests.RequestException, ValueError, KeyError, TypeError):
        ui.warning('The game schedule could not be loaded. Use Refresh live data to retry.')
        return
    if not games:
        ui.info('No upcoming games or games within one hour of scheduled tip-off were found for this date. Choose another date.')
        return
    if approved_bundle:
        uncovered = [f"{g['Away']} @ {g['Home']}" for g in games
                     if unmatched_minutes_message(approved_bundle, g['Away'], g['Home'], g['GameTimeUTC'])]
        ui.caption(f'Saved rotations cover {len(games)-len(uncovered)} of {len(games)} games on this slate. Open tabs sync when the saved file changes.')
        if uncovered:
            ui.warning('Games without an approved rotation in this file: '+', '.join(uncovered)+
                       '. Import the full slate to use your saved minutes for these games.')
    by_key = {game['key']: game for game in games}
    view = ui.radio('Projection view', ['One game', 'All games — player projections'],
                    horizontal=True, key='team_projection_view')
    if view == 'All games — player projections':
        render_all_team_players(ui, games, book_name, date, schedule, now, snapshots,
            projections, roster, teams, prior_effects, current_effects, team_rates, project_for_game, approved_bundle)
        return
    if st.session_state.get('team_projection_game') not in by_key:
        st.session_state['team_projection_game'] = games[0]['key']
    selected = ui.pills('Team matchup', list(by_key), selection_mode='single', key='team_projection_game',
                        format_func=lambda key: f"{by_key[key]['Away']} @ {by_key[key]['Home']} · {by_key[key]['GameTimeUTC'].tz_convert('America/Denver'):%I:%M %p %Z}")
    if selected is None:
        ui.info('Select a game to view its team projections.')
        return
    game = by_key[selected]
    ui.subheader(f"{game['Away']} @ {game['Home']} — {book_name}")
    with ui.expander('Adjustment details'):
        result = team_game_projection(ui, game, book_name, schedule, now, snapshots,
            projections, roster, teams, prior_effects, current_effects, team_rates, project_for_game, approved_bundle)
    if result is None:
        return
    base, before, final = result
    order = [game['Away'], game['Home']]
    totals, breakdown, issues = team_projection_tables(base, before, final, order)
    for issue in issues:
        ui.warning(issue)
    fallback = final.loc[final['Projected minutes'].gt(0) &
                         final.get('Matchup fallback stats', pd.Series('', index=final.index)).ne('')]
    if not fallback.empty:
        details = '; '.join(f"{row['Player']} ({row['Matchup fallback stats']})" for _, row in fallback.iterrows())
        ui.info('Missing archetype matchup history: ' + details +
                '. These categories retain player baseline rates before team adjustments. No opponent effect is assumed.')
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
    ui.caption('PTS targets come from the selected sportsbook’s spread and total. Allowed assist, three-attempt and offensive-rebound shares use a 5-game prior that phases out by 20 completed games; other team rates keep a 10-game prior. 3PM uses the roster’s shooting percentages. Steals, blocks and turnovers retain their player matchup projections. PRA = final PTS + REB + AST.')
    with ui.expander('Player contributions and minutes'):
        active = final['Projected minutes'].gt(0) | final['Projected minutes'].isna()
        columns = ['Player', 'Team', 'Projected minutes', *TEAM_STATS.values(), 'OREB', 'DREB', 'Minutes source', 'Stat baseline', 'Fallback details', 'Status', 'Matchup status', 'Matchup fallback stats']
        player_rows = final.loc[active].sort_values(['Team', 'Projected minutes'], ascending=[True, False])
        display = player_rows[[c for c in columns if c in final]].rename(columns={'FG3M': '3PM'})
        ui.caption('Click a player’s name or select their row to open Player vs Opponent for this matchup.')
        # Include row identity in the key so changing games/rotations cannot
        # apply an old selection to a different player. A navigation revision
        # clears the selection, allowing the same player to be opened again.
        row_ids = ','.join(player_rows.PLAYER_ID.astype(str))
        table_key = hashlib.sha256(f'{selected}:{row_ids}'.encode()).hexdigest()[:16]
        widget_key = f'contribution_players_{table_key}_{st.session_state.get("contribution_selection_revision", 0)}'
        ui.dataframe(display.round(2), hide_index=True, width='stretch', key=widget_key,
                     on_select=lambda: queue_player_selection(widget_key, player_rows, game, book_name),
                     selection_mode=['single-row', 'single-cell'])
        send_player_navigation()
        ui.caption('Zero-minute players are excluded from totals. Approved game minutes retain their exact 240-minute allocation. Other minute sources round individually to 0.5. Importing a new approved file recalculates matching games; the local NBA model does not sync automatically.')
    with ui.expander('Where the assist and three-point targets come from'):
        inputs = ['Market team total', 'Expected team FGM', 'Expected team FGA', 'Team 3PA target', 'Team assist rate', 'Team three attempt rate']
        if set(inputs).issubset(final):
            drivers = final.loc[final['Projected minutes'].gt(0), ['Team', *inputs]].drop_duplicates()
            drivers = drivers.rename(columns={'Market team total': 'Market points', 'Expected team FGM': 'Expected FGM', 'Expected team FGA':'Expected FGA',
                                              'Team assist rate': 'Blended assist rate', 'Team three attempt rate': 'Blended 3PA/FGA'})
            ui.dataframe(drivers.style.format({'Market points': '{:.2f}', 'Expected FGM': '{:.2f}',
                                               'Blended assist rate': '{:.1%}', 'Blended 3PA/FGA': '{:.1%}', 'Expected FGA':'{:.2f}', 'Team 3PA target':'{:.2f}'}, na_rep='—'),
                         hide_index=True, width='stretch')
        ui.write('Expected made field goals = sportsbook team points ÷ the 50/50 average of progressively blended team points per made field goal and opponent points allowed per made field goal. Points include free throws.')
        ui.write('Assist target = expected field goals × the average of team AST/FGM and opponent AST allowed/FGM allowed. Expected attempts = market points ÷ blended PTS/FGA. Team three-point attempts = expected attempts × blended 3PA/FGA; player makes retain individual shooting percentages.')
        ui.write('Each target is distributed using the roster’s minutes and normalized archetype effects. The final player totals add up to the team target. Threes change the scoring mix within the points projection; they do not add points again.')
    with ui.expander('Rebound targets and opportunity pools'):
        fields = ['Team OREB target', 'Team DREB target', 'Team REB target', 'Expected FG misses',
                  'Expected FT misses', 'Player credited fraction', 'Player rebound pool', 'Offensive rebound share']
        if set(fields).issubset(final):
            rebound_inputs = final.loc[final['Projected minutes'].gt(0), ['Team', *fields]].drop_duplicates()
            ui.dataframe(rebound_inputs.style.format({field: '{:.1%}' if field in ['Player credited fraction', 'Offensive rebound share'] else '{:.2f}' for field in fields}, na_rep='—'),
                         hide_index=True, width='stretch')
        ui.write('Each row’s misses belong to that shooting team. Its credited rebound pool splits into its own offensive rebounds and the opponent’s defensive rebounds. A team’s defensive target therefore comes from the other team’s missed-shot pool.')
        ui.write('Misses per made basket, the fraction producing credited player rebounds, and offensive capture shares use progressively blended seasons, then mix team and opponent rates equally. The credited fraction accounts approximately for non-reboundable free throws and dead-ball/team rebounds. Offensive and defensive targets are allocated separately using normalized player archetype shares. REB is their sum; PRA uses the updated rebounds.')
    ui.download_button('Download team projections', totals.to_csv(index=False),
                       f"{game['Away']}_{game['Home']}_team_projections.csv", 'text/csv', key='team_projection_download')
    ui.download_button('Download team adjustment breakdown', breakdown.to_csv(index=False),
                       f"{game['Away']}_{game['Home']}_team_adjustments.csv", 'text/csv', key='team_adjustment_download')


@st.fragment
def render_live_dashboard(ui, roster):
    roster = apply_roster_rules(roster)
    roster['PROJECTED_MIN'] = roster.PROJECTED_MIN.map(round_projected_minutes)
    ui.title(f'🏀 Player Stat Projections ({SEASON})')
    ui.caption(f'Your workbook minutes for each team’s first {OPENING_GAMES} games; season-average minutes from game {OPENING_GAMES + 1}. Counts include completed regular-season team games, not player appearances. Imported approved minutes take priority for matching games.')
    approved_bundle = render_import(ui, roster, SEASON)
    if ui.button('Refresh live data'):
        live_state.clear()
        today_schedule.clear()
        team_schedule.clear()
        current_matchup_logs.clear()
        sportsbook_events.clear()
        sportsbook_props.clear()
        sportsbook_game_lines.clear()
        edge_injury_report.clear()
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
        ui.write('Verified current-season stats replace RotoWire and archetype estimates from a player’s first regular-season appearance. RotoWire fills only players or stat categories without usable actuals. Your approved minutes take priority for matching games; otherwise workbook/season-average minutes apply. Unassigned archetype sides use the baseline without an opponent multiplier. An uploaded CSV applies to this session and is not saved publicly.')
        upload = ui.file_uploader('RotoWire 2026–27 season projections CSV',type=['csv'],key='rotowire_csv')
        if upload is not None:
            units = ui.selectbox('Export stat units',['Per game','Season totals','Per 36 minutes'],key='rotowire_units')
            try:
                fallback, unmatched = normalize_rotowire(read_rotowire_csv(upload),roster,
                    {'Per game':'per_game','Season totals':'season_totals','Per 36 minutes':'per_36'}[units])
                ui.caption(f'{len(fallback)} player rows matched; {len(unmatched)} unmatched rows ignored. Estimates are used only where actual stats are unavailable.')
            except (ValueError, KeyError) as exc:
                ui.error(str(exc))
                fallback = None
        elif fallback is not None:
            ui.caption(f'{len(fallback)} RotoWire baseline rows loaded from the October 16, 2026 export; verified actual stats take priority.')
        elif fallback is None:
            ui.caption('No RotoWire export has been loaded yet. Players without individual profiles can use a flagged archetype-average estimate when both assigned roles have enough historical coverage.')
    archetype_priors = build_archetype_priors(roster, history)
    projections = build_projections(roster, history, teams, players, opening_games=OPENING_GAMES, fallback=fallback,
                                    archetype_priors=archetype_priors)
    gaps = projection_gaps(approved_bundle, projections)
    if not gaps.empty:
        ui.info(f'{len(gaps)} active player entries need projection data. Where flagged below, temporary archetype averages keep team totals available until individual stats arrive. Players without archetypes retain available individual baseline rates without an adjustment for the unassigned role. Unresolved statistical gaps still leave affected totals blank.')
        with ui.expander('Players needing projection data'):
            ui.dataframe(gaps, hide_index=True, width='stretch')
    game_projection_cache = {}
    def project_for_game(team, opponent, start):
        minutes = for_matchup(approved_bundle, roster, team, opponent, start)
        if not minutes: return projections
        key = tuple(sorted(minutes.items()))
        if key not in game_projection_cache:
            changed = build_projections(roster[roster.PLAYER_ID.isin(minutes)], history, teams, players,
                opening_games=OPENING_GAMES, fallback=fallback, approved_minutes=minutes, archetype_priors=archetype_priors)
            game_projection_cache[key] = pd.concat(
                [projections[~projections.PLAYER_ID.isin(minutes)], changed], ignore_index=True)
        return game_projection_cache[key]
    role_policy_version = hashlib.sha256(b''.join((ROOT / p).read_bytes() for p in ['archetype_reviews.csv','archetype_reference_reviews.csv','archetype_rotation_reviews.csv','archetype_league_reviews.csv','archetype_overrides.json','archetype_model.py','matchup_projections.py','matchup_data.py'])).hexdigest()
    prior_effects = prior_matchup_effects(roster, role_policy_version)
    current_effects = {}
    logs = pd.DataFrame()
    if not teams.empty and teams.GAMES_PLAYED.gt(0).any():
        with ui.spinner('Refreshing current-season team rates and matchup history…'):
            try:
                logs, details = current_matchup_logs(int(SEASON[:4])+1, tuple(teams.ESPN_TEAM_ID))
                if not logs.empty:
                    current_effects = current_matchup_effects(logs, roster, role_policy_version)
                if details['errors']:
                    ui.warning('Some current-season matchup games could not be loaded. Sample counts may be incomplete.')
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                ui.warning(f'Current-season matchup data unavailable: {exc}')
    team_rates = blend_team_rates(prior_team_rates(), logs, teams)
    probability_context = scoring_context(roster, players)
    ui.caption(f'Allowed assist, three-attempt and offensive-rebound shares use a {DEFENSIVE_PRIOR_GAMES:g}-game prior through eight completed games, then phase out last season by {DEFENSIVE_FULL_CURRENT_GAMES:g}. Other team rates keep a {PRIOR_GAMES:g}-game prior. Threes preserve individual shooting percentages; PRA uses final points + rebounds + assists.')
    ui.caption('Three-point accuracy blends prior-career NBA shooting with this season using a 200-attempt prior. Career weight never falls below 50%. Minutes and normalized archetype attempt shares still determine volume. Players without prior NBA attempts retain their existing baseline.')
    ui.caption(f'Archetype matchup effects blend both seasons from the first usable game. Last season counts as {MATCHUP_PRIOR_GAMES:g} matchup games; each category uses its own relevant game count. Steals and blocks retain their stronger smoothing.')
    with ui.expander('Team rate weights and data coverage'):
        coverage = team_rates['coverage'].reset_index().rename(columns={
            'Current weight':'Other rates: this season', 'Prior games':'Other prior games',
            'Defensive current weight':'Allowed rates: this season',
            'Defensive prior games':'Allowed prior games','Defensive status':'Allowed rate status'})
        ui.dataframe(coverage.style.format({'Other rates: this season':'{:.1%}', 'Completed games':'{:.0f}',
                                            'Verified games':'{:.0f}', 'Other prior games':'{:.1f}',
                                            'Allowed rates: this season':'{:.1%}','Allowed prior games':'{:.2f}'},na_rep='—'),
                     hide_index=True,width='stretch')
        ui.caption('Allowed rates means assisted baskets, three-point attempt share and offensive-rebound share allowed. Each team uses its own completed regular-season game count. Incomplete current box scores retain last-season team rates, even after game 20; no missing games are invented. All projection pages share this snapshot; Refresh live data updates it.')
    # Each tab owns its reruns. Player/slider changes reuse this shared snapshot
    # and cannot trigger league-wide ESPN refreshes or hidden-tab calculations.
    # Imports and Refresh live data rerun the parent and rebuild every tab.
    projection_tab, best_tab, edges_tab, team_tab, archetype_tab = ui.tabs(['Player vs Opponent', "🔥 Today's Best Matchups", '💰 Top Sportsbook Edges', '🏀 Team Projections', 'Archetypes'])
    ui.caption('Player baselines reach 100% this season after 20 player appearances. Historical priors stay unchanged through eight appearances, then phase out; career three-point accuracy keeps at least 50% career weight. Rookies use actual rates from their first appearance. Opponent archetype effects keep their separate 10-game season blend.')
    @st.fragment
    def render_player_page():
        probability_observations = []
        render_player_navigation()
        apply_pending_navigation(st.session_state, roster)
        team_col, player_col, opponent_col = ui.columns(3)
        with team_col:
            team = ui.selectbox('Player team', sorted(roster.TEAM_ABBREVIATION.unique()), key='projection_team')
        team_players = projections[projections.Team.eq(team)]
        names = team_players.sort_values('Projected minutes',ascending=False).Player.tolist()
        with player_col:
            player_name = ui.selectbox('Player',names,key='projection_player')
        upcoming = selected_game_context(st.session_state, team, st.session_state.get('projection_opponent'))
        try:
            team_record = teams[teams.TEAM_ABBREVIATION.eq(team)] if 'TEAM_ABBREVIATION' in teams else teams.iloc[:0]
            if upcoming is None and not team_record.empty:
                upcoming = next_team_game(SEASON,str(team_record.iloc[0].ESPN_TEAM_ID),team)
        except (requests.RequestException,ValueError,KeyError,TypeError):
            pass
        opponents = [t for t in sorted(roster.TEAM_ABBREVIATION.unique()) if t != team]
        with opponent_col:
            opponent = render_opponent_selector(ui, team, opponents, upcoming)
        selected_game = selected_game_context(st.session_state, team, opponent)
        if selected_game:
            upcoming = selected_game
        if upcoming:
            next_time = pd.Timestamp(upcoming['GameTimeUTC']).tz_convert('America/Denver')
            game_label = 'Selected game' if selected_game else 'Current matchup' if next_time <= pd.Timestamp.now(tz='UTC') else 'Next scheduled game'
            ui.caption(f'{game_label}: {team} vs {upcoming["Opponent"]} · {next_time:%b %d at %I:%M %p %Z}. The default keeps this matchup until one hour after scheduled tip-off. You can change the opponent for any preview.')
        else:
            ui.caption('The next regular-season opponent could not be verified. Choose an opponent manually.')
        if opponent is None:
            ui.info('Choose an opponent to view this player’s projection.')
        else:
            if upcoming and upcoming['Opponent'] == opponent:
                matched_projections = project_for_game(team, opponent, upcoming['GameTimeUTC'])
                team_players = matched_projections[matched_projections.Team.eq(team)]
                notice = unmatched_minutes_message(approved_bundle, team, opponent, upcoming['GameTimeUTC'])
                if notice:
                    ui.warning(notice)
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
                    opening_games=OPENING_GAMES,fallback=fallback,minutes_overrides={player_id:custom_minutes},archetype_priors=archetype_priors)
                ui.caption('Custom minutes apply to this player’s projection in your current session. Use automatic minutes to return to approved game minutes when imported, otherwise the workbook/season-average setting.')
            if pd.isna(automatic_minutes):
                ui.caption('Automatic minutes are unavailable. Adjust the slider to set your own minutes.')
            else:
                ui.caption(f'Automatic setting: {automatic_minutes:.1f} minutes ({team_players.loc[team_players.Player.eq(player_name),"Minutes source"].iloc[0]}).')
            adjusted = apply_matchup(baseline,roster,teams,prior_effects,current_effects,opponent)
            row = adjusted.iloc[0]
            ui.subheader(f'{player_name} vs {opponent}', anchor='selected-player-projection')
            ui.caption(f"{row['Offensive archetype']} · {row['Defensive archetype']} · {row['Projected minutes']:.1f} projected minutes ({row['Minutes source']})")
            ui.caption(f"Stat baseline: {row['Stat baseline']} · Opponent history: {row['Matchup data']} · Offensive samples: {row['Off prior samples']} prior / {row['Off current samples']} current · Defensive samples: {row['Def prior samples']} prior / {row['Def current samples']} current")
            ui.caption(f"Player rates: {row['Player current weight']:.1%} this season · Steals/blocks: {row['STL/BLK current weight']:.1%} this season · {row['Season appearances']:g} appearances. Three-point accuracy uses its separate career blend; missing actual categories retain labeled fallbacks.")
            if pd.notna(row.get('Career 3P weight')):
                shooting = (f"3P accuracy: {row['Expected 3P%']:.1%} · "
                            f"Career {row['Career 3P%']:.1%} at {row['Career 3P weight']:.1%} weight")
                if pd.notna(row['Season 3P%']):
                    shooting += (f" · This season {row['Season 3P%']:.1%} on "
                                 f"{row['Season 3PA sample']:g} attempts")
                ui.caption(shooting + '. Career excludes the current season; 200-attempt prior, 50% career minimum.')
            else:
                ui.caption('3P accuracy: ' + row.get('3P accuracy source', 'Existing baseline'))
            with ui.expander('Archetype matchup blend details'):
                blend_stats = [('PTS','PTS'),('AST','AST'),('OREB','OREB'),('DREB','DREB'),('3PA / 3PM','FG3A'),('STL','STL'),('BLK','BLK'),('TOV','TOV'),('PF','PF')]
                detail = pd.DataFrame([{'Stat':label,'Last-season effect %':row[f'Prior {stat} %'],
                    'Current-season effect %':row[f'Current {stat} %'],'Current-season weight':row[f'{stat} matchup current weight'],
                    'Blended effect %':row[f'{stat} %']} for label,stat in blend_stats])
                ui.dataframe(detail.style.format({'Last-season effect %':'{:+.2f}%', 'Current-season effect %':'{:+.2f}%',
                    'Current-season weight':'{:.1%}', 'Blended effect %':'{:+.2f}%'},na_rep='—'),hide_index=True,width='stretch')
                ui.caption('Weights count distinct games with usable observations for this archetype and opponent. Samples above count player appearances. Missing current history retains last season; missing both seasons retains the player baseline. Without a historical effect, current evidence is shrunk toward an average matchup. Team adjustments then normalize player shares as before.')
            if str(row['Stat baseline']).startswith('Current season'):
                ui.caption(f"Actual stat sample: {row['Season appearances']:g} regular-season appearances · "
                           f"{row['Actual sample minutes']:.1f} minutes. Per-minute rates scale to your projected minutes.")
            if row.get('Estimate fallback stats', ''):
                ui.caption('Actual stats are unavailable for these categories; estimates remain: ' + row['Estimate fallback stats'])
            if row['Status'] != 'Available':
                ui.warning(row['Status'])
            if row.get('Estimated stat profile', False):
                ui.caption(row['Fallback details'] + ' This temporary estimate is excluded from the top-edge rankings.')
            if row['Matchup status'] == 'Baseline fallback — missing archetype':
                ui.info('Using baseline rates for the unassigned archetype, then the normal team adjustments. No archetype has been assigned. Matchup ratings stay blank, and this player is excluded from the top-edge rankings until classified.')
            elif row['Matchup status'] == 'Baseline fallback — missing matchup history':
                ui.info('No usable archetype matchup history for: ' + row['Matchup fallback stats'] +
                        '. These categories retain baseline rates, then normal team adjustments apply. Matchup ratings stay blank and this player is excluded from top-edge rankings until matchup coverage is available.')
            elif row['Matchup status'] != 'Available':
                ui.warning(row['Matchup status'] + '. Unavailable adjustments stay blank; turn off matchup adjustments to view the baseline.')
            stats = ['PTS','REB','AST','FG3M','STL','BLK','TOV','PF','PRA']
            preview = pd.DataFrame({'Stat':['3PM' if x=='FG3M' else x for x in stats],
                'Baseline':[baseline.iloc[0][x] for x in stats],
                'Projected':[row[x] if adjusted_on else baseline.iloc[0][x] for x in stats]})
            preview['Matchup rating'] = [matchup_rating(base,projected,stat,adjusted_on and row['Matchup status'] == 'Available') for base,projected,stat in zip(preview.Baseline,preview.Projected,preview.Stat)]
            preview['Change'] = preview.Projected - preview.Baseline
            book_name = ui.selectbox('Sportsbook',list(BOOKS),key='projection_sportsbook')
            show_recent = ui.toggle('Show recent-role comparison', value=False, key='player_recent_role',
                                    help='Optional 10-appearance shot-volume/assist model. The current projection still drives edges and probabilities.')
            recent_source = None
            book_key = BOOKS[book_name]
            quotes = player_quotes({},player_name,book_key)
            event = None
            try:
                api_key = get_odds_api_key()
                if not api_key:
                    raise OddsFeedError('Sportsbook feed access is not configured.')
                scheduled_start = upcoming['GameTimeUTC'] if upcoming and upcoming['Opponent'] == opponent else None
                event = next_matchup(sportsbook_events(api_key),team,opponent,
                                     scheduled_start=scheduled_start,time_tolerance=EVENT_TIME_TOLERANCE)
                if event is None:
                    ui.caption(f'No matching pregame sportsbook event is available for {team} vs {opponent}. Lines stay blank after tip-off; prices from a later rematch are not substituted.')
                else:
                    team_baselines = team_players.copy()
                    if flag_key in st.session_state:
                        for column in baseline.columns:
                            if column in team_baselines:
                                team_baselines.loc[team_baselines.PLAYER_ID.eq(player_id),column] = baseline.iloc[0][column]
                    full_team = apply_matchup(team_baselines,roster,teams,prior_effects,current_effects,opponent) if adjusted_on else team_baselines
                    market_team = team_projection_preview(ui,full_team,event,book_key,api_key,team_rates)
                    recent_source = market_team
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
                                   'Pre-team 3PM','Pre-team 3PA','Team 3PM factor','Team 3PM target','Team 3PA target','Expected team FGA','Team three attempt rate','Three-point share',
                                   'Team rate current weight','Team rate games','Team rate prior games','Team rate status',
                                   'Opponent allowed current weight','Opponent allowed prior games','Opponent allowed rate status',*REBOUND_METADATA]:
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
            final_player = (adjusted if adjusted_on else baseline).iloc[0].copy()
            final_player['PTS'] = float(preview.loc[preview.Stat.eq('PTS'), 'Projected'].iloc[0])
            final_player['Opponent'] = opponent
            points_line = comparison.loc[comparison.Stat.eq('PTS')].iloc[0]
            probability = points_quote(final_player, points_line, probability_context.get(player_id))
            for column in PROBABILITY_COLUMNS:
                comparison[column] = float('nan')
                comparison.loc[comparison.Stat.eq('PTS'), column] = probability[column]
            if event is not None:
                mode = ('Manual minutes + opponent adjustment off' if flag_key in st.session_state and not adjusted_on
                        else 'Manual minutes' if flag_key in st.session_state
                        else 'Opponent adjustment off' if not adjusted_on else 'Standard')
                saved = probability_snapshot({**final_player.to_dict(), **points_line.to_dict(), **probability},
                    event, book_key, probability_context, 'Player vs Opponent', mode)
                if saved:
                    probability_observations.append(saved)
            comparison['Book line (Over)'] = [over_line_label(line,price) for line,price in zip(comparison['Book line'],comparison['Over odds'])]
            display_columns = ['Stat','Baseline','Matchup rating','Projected','Book line (Over)','Book implied total','Edge %','Play', *PROBABILITY_COLUMNS]
            if show_recent:
                if recent_source is None:
                    recent_source = team_players.copy()
                    for column in baseline.columns:
                        if column in recent_source:
                            recent_source.loc[recent_source.PLAYER_ID.eq(player_id),column] = baseline.iloc[0][column]
                    if adjusted_on:
                        recent_source = apply_matchup(recent_source,roster,teams,prior_effects,current_effects,opponent)
                recent = recent_role_comparison(recent_source, roster, players, logs, SEASON)
                comparison = add_recent_columns(comparison, recent, player_id)
                display_columns[4:4] = RECENT_COLUMNS
                ui.caption(RECENT_CAPTION)
            display = comparison[display_columns]
            ui.caption('Edge % measures the gap from the book-implied total. Plays require a gap above 5%. Steals and blocks Unders also require an Under price of −160 or better. Darker shading indicates a larger edge.')
            ui.dataframe(display.style.apply(comparison_style,axis=1).format(
                {**{column:'{:.2f}' for column in display_columns if column not in ['Stat','Play','Edge %','Matchup rating','Book line (Over)','Recent role status']},
                 **{column:'{:.1%}' for column in PROBABILITY_COLUMNS},'Edge %':'{:+.2f}%', 'Recent change %':'{:+.1f}%'},na_rep='—'),hide_index=True,width='stretch')
            ui.caption('Exp. probabilities are experimental and available for points only. They do not affect projected averages or the 5% edge rule. Extreme probabilities and low-minute players are less reliable; integer lines include a push.')
            if probability['Probability status'] != 'Experimental':
                ui.caption('Points probability: ' + probability['Probability status'] + '.')
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
                ui.write('Steals and blocks use stronger smoothing toward an average matchup, reducing the impact of small or noisy samples while retaining archetype differences.')
                ui.write('Each archetype’s per-36 production against the selected opponent is compared with its league-wide per-36 average. Percent changes are weighted toward recent player-games and shrunk toward zero with 20 neutral samples for steals and blocks, and three for other stats. Offensive archetypes drive points, assists, offensive rebounds and shooting; defensive archetypes drive defensive rebounds, steals and blocks. Projected rebounds equal offensive plus defensive rebounds. Three-point makes use the attempts adjustment to preserve the player’s own shooting rate.')
                ui.write('Minutes use your workbook for team games 1–3, then observed season minutes per appearance from game 4. Approved game minutes take priority. Rookies and other players using estimates switch to actual rates after their first verified appearance. Historical player rates start blending after four completed team games. The starting prior is 200 minutes, or 1,000 for steals/blocks when a player logged at least 500 actual minutes last season. These priors stay intact through eight player appearances, then decrease linearly to zero at 20 appearances. Current weight is actual season minutes divided by actual minutes plus the remaining prior. Missed team games do not advance this transition. Missing actual categories retain labeled fallbacks. Career three-point accuracy keeps its 200-attempt prior and 50% career minimum. Archetype matchup effects keep their separate relevant games / (relevant games + 10) season blend and existing smoothing.')
                ui.write('Team assists keep a 10-game prior for own rates and points-to-field-goals conversion. The opponent’s assisted-basket rate uses a 5-game prior through eight completed games, then tapers to zero prior games at 20. Team and opponent rates still combine equally. The team assist total is allocated in proportion to existing archetype-adjusted player assists, so shared defensive penalties cancel while relative archetype differences remain. No second assist multiplier is applied. PRA combines the final points, rebounds and assists.')
                ui.write('Team threes use expected FGA times a 50/50 mix of progressively blended team and opponent 3PA/FGA. Team attempts are allocated using normalized archetype-adjusted player attempt shares. Makes scale equally to preserve each player’s expected shooting percentage. Points and PRA already include the threes; no extra points are added. The player slider recalculates this preview, while game edges use the saved automatic team minutes. Changes to an external workbook must be imported and published before the app can use them.')
                ui.write('Team rebounds use both teams’ expected misses, derived from market points and conversion rates with a 10-game prior. The opponent’s allowed offensive-rebound share uses the faster 5-game prior, phased out by 20 completed games; own rebound shares retain 10. Credited-player-rebound fractions account approximately for dead-ball/team rebounds and non-reboundable free throws. Each pool splits into offensive rebounds and opposing defensive rebounds. The two components are allocated separately using normalized archetype shares, then added for REB. PRA includes updated rebounds. Player matchup ratings retain the original archetype effect.')
                ui.write('These are matchup estimates, not confirmed lineups or availability. The model does not adjust for pace, injuries, starters or home court. Last season’s box scores are assigned the approved current archetypes, preserving manual overrides. Only players with matched archetype labels contribute to the matchup samples; sample counts show that coverage.')
        render_probability_tracking(ui, probability_observations, logs, 'player')
    with projection_tab:
        render_player_page()

    @st.fragment
    def render_best_matchups_page():
        ui.subheader("🔥 Today's Best Matchup Opportunities")
        date_col, rank_col, sample_col = ui.columns(3)
        with date_col:
            date = ui.date_input('Slate date',value=datetime.now(ZoneInfo('America/Denver')).date(),key='matchup_date')
        schedule = schedule_for_view(ui, date, 'matchup_schedule_reload')
        if schedule is None:
            return
        if schedule.empty:
            ui.info(f'No upcoming regular-season games were found for {date:%b %d, %Y}. Choose another date to preview its matchups.')
        else:
            scheduled_frames = []
            uncovered = {}
            for fixture in schedule.itertuples():
                side = next((name for name,tid in TEAM_IDS.items() if tid == fixture.TEAM_ID), None)
                frame = project_for_game(side, fixture.Opponent, fixture.GameTimeUTC)
                scheduled_frames.append(frame[frame.TEAM_ID.eq(fixture.TEAM_ID)])
                notice = unmatched_minutes_message(approved_bundle, side, fixture.Opponent, fixture.GameTimeUTC)
                if notice:
                    uncovered[tuple(sorted([side, fixture.Opponent]))] = notice
            if uncovered:
                extra = f' {len(uncovered)-1} other scheduled games are also not covered.' if len(uncovered)>1 else ''
                ui.warning(next(iter(uncovered.values())) + extra)
            slate_projections = pd.concat(scheduled_frames,ignore_index=True)
            slate = slate_projections.merge(schedule,on='TEAM_ID',how='inner',validate='many_to_one')
            matchups = apply_matchup(slate,roster,teams,prior_effects,current_effects)
            # Calibrate complete rosters before sample, minute or rank filters.
            # Ratings remain the original archetype effects; stat lines and CSVs
            # use the same market/team pipeline as the other projection pages.
            matchups['Team rate status'] = 'No verified game total'
            matchup_book = ui.selectbox('Sportsbook for matchup projections',list(BOOKS),key='matchup_book')
            with ui.expander('Team projection adjustments'):
                try:
                    key = get_odds_api_key()
                    if not key:
                        raise OddsFeedError('Sportsbook feed access is not configured.')
                    events = scheduled_book_games(sportsbook_events(key),schedule)
                    for game in events:
                        mask = ((matchups.Team.eq(game['Home']) & matchups.Opponent.eq(game['Away']))
                                | (matchups.Team.eq(game['Away']) & matchups.Opponent.eq(game['Home'])))
                        if not mask.any():
                            continue
                        event = {**game,'commence_time':game['GameTimeUTC'].isoformat()}
                        try:
                            calibrated = team_projection_preview(ui,matchups.loc[mask],event,BOOKS[matchup_book],key,team_rates)
                            matchups = pd.concat([matchups.loc[~mask],calibrated],ignore_index=True)
                        except (OddsFeedError,requests.RequestException,ValueError,KeyError,TypeError):
                            ui.caption(f"{game['Away']} @ {game['Home']}: game team adjustments unavailable; retaining matchup projections.")
                except (OddsFeedError,requests.RequestException,ValueError,KeyError,TypeError):
                    ui.caption('Verified game totals are unavailable; retaining matchup projections.')
            matchups = matchups[matchups['Projected minutes'].gt(0)]
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
                ui.caption('Uses the same team adjustments as Player vs Opponent, Top Sportsbook Edges and Team Projections when the selected sportsbook has a verified game total. Matchup percentages above remain archetype effects.')
                ui.dataframe(eligible[['Player','Team','Opponent','Projected minutes','PTS','REB','AST','FG3M','STL','BLK','TOV','PRA','Stat baseline','Status','Team rate status']].rename(columns={'FG3M':'3PM'}).round(2),hide_index=True,width='stretch')
            ui.download_button('Download matchup opportunities CSV',eligible.to_csv(index=False),'matchup_opportunities.csv','text/csv')
    with best_tab:
        render_best_matchups_page()

    @st.fragment
    def render_edges_page():
        probability_observations = []
        ui.subheader('💰 Sportsbook Edges by Team')
        injury_panel = ui.container()
        ui.caption('Edges are ranked within each team by the gap from the juice-adjusted book total. Steals and blocks Unders require an Under price of −160 or better.')
        # Only fetch props for the selected game, never the full slate.
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
            show_recent = ui.toggle('Show recent-role comparison', value=False, key='edges_recent_role',
                                    help='Compare recent shot volume and assists beside the current model. Rankings and the 5% rule stay unchanged.')
            edge_schedule = schedule_for_view(ui, edge_date, 'edges_schedule_reload')
            if edge_schedule is None:
                return
            games = scheduled_book_games(edge_events,edge_schedule,now=now)
            if not games:
                ui.info('No upcoming regular-season games with a verified sportsbook event were found for this date. Choose another date. Lines are never substituted from a different game.')
            else:
                by_id = {game['id']:game for game in games}
                labels = {game['id']:f"{game['Away']} @ {game['Home']} · {game['GameTimeUTC'].tz_convert('America/Denver'):%I:%M %p %Z}" for game in games}
                if st.session_state.get('edges_game') not in by_id:
                    st.session_state['edges_game'] = games[0]['id']
                game_id = ui.pills('Choose a game',list(by_id),format_func=lambda value:labels[value],selection_mode='single',key='edges_game')
                with injury_panel:
                    render_edge_injuries(ui, games, game_id, edge_date, roster, projections, approved_bundle)
                if game_id is None:
                    ui.info('Select a game to see its top edges.')
                else:
                    game = by_id[game_id]
                    game_projections = project_for_game(game['Home'], game['Away'], game['GameTimeUTC'])
                    ui.caption(game_minutes_caption(game_projections, [game['Away'], game['Home']]))
                    notice = unmatched_minutes_message(approved_bundle, game['Away'], game['Home'], game['GameTimeUTC'])
                    if notice:
                        ui.warning(notice)
                    matchup_rows = []
                    for side,opponent in [(game['Home'],game['Away']),(game['Away'],game['Home'])]:
                        # Keep the whole roster so unavailable minutes cannot silently
                        # disappear before team-coverage checks. Rankings exclude inactive rows.
                        team_projection = game_projections[game_projections.Team.eq(side)]
                        if not team_projection.empty:
                            matchup_rows.append(apply_matchup(team_projection,roster,teams,prior_effects,current_effects,opponent))
                    payload = sportsbook_props(game_id,BOOKS[edge_book_name],edge_key)
                    combined = pd.concat(matchup_rows,ignore_index=True) if matchup_rows else pd.DataFrame()
                    if not combined.empty:
                        market_event = {**game,'commence_time':game['GameTimeUTC'].isoformat()}
                        combined = team_projection_preview(ui,combined,market_event,BOOKS[edge_book_name],edge_key,team_rates)
                    ranked,total_edges = game_edges(combined,payload,game,BOOKS[edge_book_name],now=now,plays_only=plays_only,direction=edge_direction,stats=selected_stats,player_search=player_search)
                    if not combined.empty:
                        probabilities = game_points_probabilities(combined, payload, BOOKS[edge_book_name], probability_context, now=now)
                        ranked = ranked.merge(probabilities[['Player', 'Team', 'Stat', *PROBABILITY_COLUMNS]],
                                              on=['Player', 'Team', 'Stat'], how='left', validate='many_to_one', sort=False)
                        for _, probability_row in probabilities.iterrows():
                            saved = probability_snapshot(probability_row, market_event, BOOKS[edge_book_name],
                                probability_context, 'Top Sportsbook Edges')
                            if saved:
                                probability_observations.append(saved)
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
                        ui.success(f'Showing all {total_edges} eligible player-stat edges, grouped by team')
                        ranked['Book line (Over)'] = [over_line_label(line,price) for line,price in zip(ranked['Book line'],ranked['Over odds'])]
                        ranked = add_minutes_cushion(ranked, combined)
                        display_columns = ['Rank','Player','Stat','Play','Projected minutes','Good-to minutes','Minutes cushion','Matchup rating','Projected','Book line (Over)','Book implied total','Edge %','Over odds','Under odds', *PROBABILITY_COLUMNS]
                        ui.caption('Good-to minutes shows how low an Over or how high an Under can go while its edge stays above 5%. Minutes cushion is the room from the projection. Tested in half-minute steps through 48 with the same prices and teammate minutes.')
                        if show_recent:
                            recent = recent_role_comparison(combined, roster, players, logs, SEASON)
                            ranked = add_recent_columns(ranked, recent)
                            after_projection = display_columns.index('Projected') + 1
                            display_columns[after_projection:after_projection] = RECENT_COLUMNS
                            ui.caption(RECENT_CAPTION)
                        team_exports = []
                        for team in [game['Away'], game['Home']]:
                            team_edges = ranked.loc[ranked.Team.eq(team)].copy()
                            ui.subheader(f"{team} — {len(team_edges)} edges")
                            if team_edges.empty:
                                ui.info('No eligible edges for this team match the current filters.')
                                continue
                            team_edges['Rank'] = range(1, len(team_edges) + 1)
                            team_exports.append(team_edges.drop(columns='PLAYER_ID'))
                            render_edge_table(team_edges, display_columns, roster, current_effects, team)
                        ui.caption('Exp. probabilities are experimental, points only, and do not change rankings or the 5% rule. Extreme probabilities and low-minute players are less reliable. Over + Under + Push = 100%.')
                        ui.caption('Hover or tap a matchup rating for recent results by archetype. Ratings measure opponent lift on a scale from 1 (poor) to 5 (excellent).')
                        ui.caption('Edge % is positive for both Overs and Unders; Play identifies the direction. Only edges greater than 5% are marked as plays. Color intensity grows with edge magnitude. Multiple stats for the same player may be correlated.')
                        ui.download_button('Download all edges for this game',pd.concat(team_exports,ignore_index=True).to_csv(index=False),f"{game['Away']}_{game['Home']}_edges.csv",'text/csv',key='edges_download')
                    with ui.expander('Game projection coverage'):
                        if not combined.empty:
                            ui.dataframe(combined[['Player','Team','Opponent','Projected minutes','Minutes source','Off samples','Def samples','Stat baseline','Matchup data','Status','Matchup status']],hide_index=True,width='stretch')
                        ui.caption('Players with zero/unavailable minutes, incomplete statistical profiles, or missing archetype matchup history do not enter the rankings. Manual minutes on Player vs Opponent affect that preview only.')
        except OddsFeedError as exc:
            ui.info(str(exc))
        except (requests.RequestException,ValueError,KeyError,TypeError):
            ui.warning('Game edge rankings are temporarily unavailable. Refresh live data to retry.')
        render_probability_tracking(ui, probability_observations, logs, 'edges')
    with edges_tab:
        render_edges_page()

    @st.fragment
    def render_team_page():
        render_team_projections(ui, projections, roster, teams, prior_effects, current_effects, team_rates, project_for_game, approved_bundle)
    with team_tab:
        render_team_page()

    @st.fragment
    def render_archetypes_page():
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
    with archetype_tab:
        render_archetypes_page()
