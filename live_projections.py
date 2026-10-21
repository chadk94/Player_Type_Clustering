"""Player projections from validated regular-season summaries, independent of odds feeds."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import re
import unicodedata

import numpy as np
import pandas as pd
import requests

from import_minutes import TEAM_IDS

ESPN = 'https://site.api.espn.com/apis'
ALIASES = {'NY': 'NYK', 'GS': 'GSW', 'SA': 'SAS', 'NO': 'NOP', 'UTAH': 'UTA', 'WSH': 'WAS', 'PHO': 'PHX'}
CORE_STATS = ('PTS','REB','AST','FG3M','STL','BLK','TOV')
STAT_NAMES = {'PTS': 'avgPoints', 'REB': 'avgRebounds', 'AST': 'avgAssists',
              'STL': 'avgSteals', 'BLK': 'avgBlocks', 'FG3M': 'avgThreePointFieldGoalsMade',
              'TOV': 'avgTurnovers', 'FGM': 'avgFieldGoalsMade', 'FGA': 'avgFieldGoalsAttempted',
              'FG3A': 'avgThreePointFieldGoalsAttempted', 'FTM': 'avgFreeThrowsMade',
              'FTA': 'avgFreeThrowsAttempted', 'OREB': 'avgOffensiveRebounds', 'DREB': 'avgDefensiveRebounds',
              'PF': 'avgFouls'}


def round_projected_minutes(value):
    """Round effective projected minutes to half-minutes; quarter ties round up."""
    return float(np.floor(value * 2 + 0.5) / 2)


def name_key(name):
    text = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode().lower()
    words = re.sub(r'[^a-z0-9 ]', '', text).split()
    # Retain suffixes to avoid matching related players; only punctuation/diacritics change.
    return ' '.join(words)


def fetch_json(url):
    response = requests.get(url, timeout=(3, 6), headers={'User-Agent': 'PlayerArchetypes/2026-27'})
    response.raise_for_status()
    return response.json()


def parse_standings(payload, season_year):
    rows = []
    for conference in payload.get('children', []):
        table = conference.get('standings', {})
        if table.get('season') != season_year or int(table.get('seasonType', 0)) != 2:
            raise ValueError('Standings have the wrong season or include preseason/playoffs')
        for entry in table.get('entries', []):
            team = entry['team']
            abbr = ALIASES.get(team['abbreviation'], team['abbreviation'])
            if abbr not in TEAM_IDS:
                raise ValueError(f'Unknown team abbreviation: {abbr}')
            values = {s['name']: s.get('value') for s in entry['stats']}
            wins, losses = values.get('wins'), values.get('losses')
            if any(v is None or not np.isfinite(v) or v < 0 or int(v) != v for v in (wins, losses)):
                raise ValueError('Missing or invalid current team record')
            rows.append({'TEAM_ID': TEAM_IDS[abbr], 'TEAM_ABBREVIATION': abbr,
                         'ESPN_TEAM_ID': str(team['id']), 'GAMES_PLAYED': int(wins + losses)})
    frame = pd.DataFrame(rows)
    if len(frame) != 30 or frame.TEAM_ID.nunique() != 30:
        raise ValueError('Expected all 30 unique NBA regular-season team records')
    return frame


def parse_player_summary(payload, season_year):
    # ESPN can silently return last year when no current games exist. Reject that fallback.
    scope = payload.get('requestedSeason', {})
    if scope.get('year') != season_year or int(scope.get('type', 0)) != 2:
        raise ValueError('Player feed returned a different season or season type')
    rows = []
    for group in payload.get('results', []):
        for leader in group.get('leaders', []):
            athlete = leader.get('athlete', {})
            values = {stat['name']: stat.get('value')
                      for category in leader.get('statistics', []) for stat in category.get('stats', [])}
            gp, mpg = values.get('gamesPlayed'), values.get('avgMinutes')
            if gp is None or mpg is None or not np.isfinite(gp) or not np.isfinite(mpg) or gp <= 0 or mpg <= 0 or int(gp) != gp:
                continue
            row = {'ESPN_PLAYER_ID': str(athlete['id']), 'PLAYER_NAME': athlete['displayName'],
                   'GP': float(gp), 'MIN': float(mpg)}
            for key, source in STAT_NAMES.items():
                row[key] = values.get(source, np.nan)
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=['ESPN_PLAYER_ID', 'PLAYER_NAME', 'GP', 'MIN', *STAT_NAMES])
    frame = pd.DataFrame(rows)
    # Multiple result sections may repeat a player. Identical IDs refer to the same athlete.
    return frame.sort_values('GP', ascending=False).drop_duplicates('ESPN_PLAYER_ID')


def load_live_state(season):
    year = int(season[:4]) + 1  # ESPN uses the year in which the season ends.
    teams = parse_standings(fetch_json(f'{ESPN}/v2/sports/basketball/nba/standings?season={year}&seasontype=2'), year)
    frames, errors = [], []
    # A team with zero completed regular-season games must not contribute preseason stats.
    playing = teams[teams.GAMES_PLAYED.gt(0)]
    def read_team(row):
        url = f'{ESPN}/site/v2/sports/basketball/nba/teams/{row.ESPN_TEAM_ID}/athletes/statistics?season={year}&seasontype=2'
        return parse_player_summary(fetch_json(url), year)
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = {pool.submit(read_team, row): row.TEAM_ABBREVIATION for row in playing.itertuples()}
        for job in as_completed(jobs):
            try:
                frames.append(job.result())
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                errors.append(f'{jobs[job]}: {type(exc).__name__}: {exc}')
    stats = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['PLAYER_NAME','GP','MIN',*STAT_NAMES])
    if not stats.empty:
        stats = stats.sort_values('GP', ascending=False).drop_duplicates('ESPN_PLAYER_ID')
    return {'teams': teams, 'players': stats, 'errors': errors,
            'as_of': datetime.now(timezone.utc).isoformat(), 'source': 'ESPN regular-season standings and player summaries'}


def parse_schedule(payload, season_year):
    rows = []
    for event in payload.get('events', []):
        scope = event.get('season', {})
        if scope.get('year') != season_year or scope.get('type') != 2:
            continue
        if event.get('status', {}).get('type', {}).get('completed'):
            continue
        competition = event.get('competitions', [{}])[0]
        competitors = competition.get('competitors', [])
        if len(competitors) != 2:
            continue
        a, b = competitors
        aa = ALIASES.get(a['team']['abbreviation'], a['team']['abbreviation'])
        bb = ALIASES.get(b['team']['abbreviation'], b['team']['abbreviation'])
        if aa not in TEAM_IDS or bb not in TEAM_IDS:
            continue
        for team, opponent, entry in [(aa,bb,a), (bb,aa,b)]:
            rows.append({'TEAM_ID': TEAM_IDS[team], 'Opponent': opponent,
                         'Venue': entry['homeAway'], 'GameTimeUTC': event['date'],
                         'GameStatus': event.get('status', {}).get('type', {}).get('description', '')})
    return pd.DataFrame(rows, columns=['TEAM_ID','Opponent','Venue','GameTimeUTC','GameStatus'])


def match_player_stats(roster, current):
    """Normalize names only; ambiguous matches remain unavailable instead of fuzzy guessed IDs."""
    if current.empty:
        return {}
    current = current.copy()
    current['_key'] = current.PLAYER_NAME.map(name_key)
    unique = current[~current['_key'].duplicated(keep=False)].set_index('_key')
    matches = {}
    for row in roster.itertuples():
        keys = {name_key(row.PLAYER_NAME), name_key(getattr(row,'WorkbookPlayerName',row.PLAYER_NAME))}
        found = [unique.loc[key] for key in keys if key in unique.index]
        if len({str(r.get('ESPN_PLAYER_ID', r.name)) for r in found}) == 1:
            matches[row.PLAYER_ID] = found[0]
    return matches


def build_projections(roster, history, teams, current, opening_games=3, prior_minutes=200, fallback=None, stats_start_game=5, minutes_overrides=None, approved_minutes=None):
    """Workbook MPG for games 1–3, season MPG for game 4 onward; no fabricated fallbacks."""
    if opening_games < 0 or prior_minutes < 0:
        raise ValueError('Projection settings must be nonnegative')
    minutes_overrides = minutes_overrides or {}
    approved_minutes = approved_minutes or {}
    if any(not np.isfinite(value) or not 0 <= value <= 48 for value in approved_minutes.values()):
        raise ValueError('Approved minutes must be between 0 and 48')
    if any(not np.isfinite(value) or not 0 <= value <= 48 for value in minutes_overrides.values()):
        raise ValueError('Custom minutes must be between 0 and 48')
    counts = teams.set_index('TEAM_ID').GAMES_PLAYED.to_dict()
    profiles = history.set_index('PLAYER_ID')
    matches = match_player_stats(roster, current)
    fallback_profiles = fallback.set_index('PLAYER_ID') if fallback is not None and not fallback.empty else pd.DataFrame()
    output = []
    for player in roster.itertuples():
        played = counts.get(player.TEAM_ID)
        observed = matches.get(player.PLAYER_ID)
        profile = profiles.loc[player.PLAYER_ID] if player.PLAYER_ID in profiles.index else None
        fallback_used = False
        unclassified = any(pd.isna(getattr(player, f'{side}Archetype', None)) for side in ('Off', 'Def'))
        if (unclassified or profile is None or not pd.notna(profile.MIN) or profile.MIN <= 0) and player.PLAYER_ID in fallback_profiles.index:
            profile = fallback_profiles.loc[player.PLAYER_ID]
            fallback_used = True
        source = 'Unavailable'
        minutes = np.nan
        if played is not None and played < opening_games:
            minutes, source = player.PROJECTED_MIN, 'Your workbook'
        elif played is not None and observed is not None and observed.GP > 0:
            minutes, source = observed.MIN, 'Season average'
        if player.PLAYER_ID in approved_minutes:
            minutes, source = float(approved_minutes[player.PLAYER_ID]), 'Approved game minutes'
        if played is not None and player.PLAYER_ID in minutes_overrides:
            minutes, source = float(minutes_overrides[player.PLAYER_ID]), 'Manual adjustment'
        if source != 'Approved game minutes': minutes = round_projected_minutes(minutes)
        row = {'PLAYER_ID':player.PLAYER_ID,'Player':player.PLAYER_NAME,'Team':player.TEAM_ABBREVIATION,
               'TEAM_ID':player.TEAM_ID, 'Team games completed':played, 'Projected minutes':minutes,'Minutes source':source,
               'Offensive archetype':player.OffArchetype,'Defensive archetype':player.DefArchetype,
               'Season appearances':0 if observed is None else observed.GP}
        total_current_minutes = 0 if observed is None else observed.MIN * observed.GP
        prior_ok = profile is not None and pd.notna(profile.MIN) and profile.MIN > 0
        current_ok = observed is not None and total_current_minutes > 0 and played is not None and played >= stats_start_game - 1
        prior_label = profile.get('BaselineLabel') if prior_ok else None
        if not isinstance(prior_label,str) or not prior_label:
            prior_label = '2025–26 archived snapshot'
        if prior_ok and current_ok:
            weight = total_current_minutes / (total_current_minutes + prior_minutes)
            baseline_source = 'RotoWire projection + current season' if fallback_used else 'Prior season + current season'
        elif current_ok:
            weight, baseline_source = 1., 'Current season'
        elif prior_ok:
            weight, baseline_source = 0., ('RotoWire 2026–27 projection' if fallback_used else prior_label)
        else:
            weight, baseline_source = 0., 'Unavailable — no statistical profile'
        row['Stat baseline'] = baseline_source
        missing = []
        for stat in STAT_NAMES:
            prior_rate = profile.get(stat, np.nan) / profile.MIN if prior_ok else np.nan
            new_rate = observed.get(stat, np.nan) / observed.MIN if current_ok else np.nan
            if prior_ok and current_ok:
                if stat == 'PF' and not np.isfinite(prior_rate):
                    rate = new_rate
                elif stat == 'PF' and not np.isfinite(new_rate):
                    rate = prior_rate
                else:
                    rate = (1-weight)*prior_rate + weight*new_rate
            else:
                rate = new_rate if current_ok else prior_rate
            row[stat] = rate * minutes if np.isfinite(rate) and np.isfinite(minutes) else np.nan
            if stat in CORE_STATS and pd.isna(row[stat]):
                missing.append(stat)
        if source == 'Unavailable':
            row['Status'] = 'Team count unavailable' if played is None else 'Needs current-season minutes (no matched season appearances)'
        elif not prior_ok and not current_ok:
            row['Status'] = 'Needs statistical profile'
        elif missing:
            row['Status'] = 'Some stat rates unavailable'
        else:
            row['Status'] = 'Available'
        if np.isfinite(row['PTS']) and np.isfinite(row['REB']) and np.isfinite(row['AST']):
            row['PRA'] = row['PTS'] + row['REB'] + row['AST']
        else:
            row['PRA'] = np.nan
        output.append(row)
    return pd.DataFrame(output)


def next_regular_season_game(payload, season_year, team_abbreviation, now=None,
                             post_tip_window=timedelta(0)):
    """Earliest verified regular-season game still inside the selection window."""
    now = now or datetime.now(timezone.utc)
    games=[]
    for event in payload.get('events',[]):
        try:
            if event.get('season',{}).get('year') != season_year or int(event.get('seasonType',{}).get('type',0)) != 2:
                continue
            competition=event['competitions'][0]
            status=competition.get('status',{}).get('type',{})
            if status.get('completed') or status.get('name') in ['STATUS_POSTPONED','STATUS_CANCELED','STATUS_CANCELLED','STATUS_SUSPENDED']:
                continue
            start=datetime.fromisoformat(event['date'].replace('Z','+00:00'))
            if start.tzinfo is None or start + post_tip_window <= now:
                continue
            participants=competition['competitors']
            abbreviations=[ALIASES.get(p['team']['abbreviation'],p['team']['abbreviation']) for p in participants]
            if len(abbreviations)!=2 or team_abbreviation not in abbreviations or any(t not in TEAM_IDS for t in abbreviations):
                continue
            opponent=next(t for t in abbreviations if t != team_abbreviation)
            games.append({'Opponent':opponent,'GameTimeUTC':start.astimezone(timezone.utc).isoformat(),'GameID':str(event['id'])})
        except (KeyError,ValueError,TypeError,StopIteration):
            continue
    return min(games,key=lambda game:game['GameTimeUTC']) if games else None
