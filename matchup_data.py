"""Validated ESPN regular-season box scores for archetype matchup estimates."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import json
import hashlib
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
import numpy as np
import pandas as pd
from live_projections import ESPN, ALIASES, fetch_json
from season_pipeline import ROOT

COLUMNS = ['ESPN_PLAYER_ID','PLAYER_NAME','GAME_ID','GAME_DATE','Team','Opponent','MIN',
           'PTS','AST','REB','OREB','DREB','STL','BLK','TOV','FGM','FGA','FG3M','FG3A','FTM','FTA','PF']


def parse_boxscore(payload, year):
    header = payload.get('header', {})
    season = header.get('season', {})
    if season.get('year') != year or int(season.get('type', 0)) != 2:
        raise ValueError('Box score has wrong season or is not regular season')
    # NBA Cup final is tagged type 2 by ESPN but excluded from season statistics.
    if 'cup' in header.get('gameNote','').lower() and ('championship' in header.get('gameNote','').lower() or header.get('gameNote','').lower().strip().endswith('cup final')):
        return []
    competition = header['competitions'][0]
    if not competition.get('status', {}).get('type', {}).get('completed'):
        return []
    teams = {str(c['team']['id']): ALIASES.get(c['team']['abbreviation'], c['team']['abbreviation'])
             for c in competition['competitors']}
    rows = []
    for group in payload.get('boxscore', {}).get('players', []):
        team_id = str(group['team']['id'])
        opponent = next((v for k,v in teams.items() if k != team_id), None)
        if team_id not in teams or opponent is None:
            continue
        for category in group.get('statistics', []):
            for entry in category.get('athletes', []):
                if entry.get('didNotPlay') or not entry.get('stats'):
                    continue
                values = dict(zip(category['labels'], entry['stats']))
                try:
                    minute = str(values['MIN'])
                    minutes = float(minute) if ':' not in minute else sum(float(v)/60**i for i,v in enumerate(minute.split(':')))
                    # ESPN rounds brief appearances to zero minutes, even when
                    # a shot was attempted. Team rates must include that shot.
                    if not np.isfinite(minutes) or minutes < 0:
                        continue
                    row = {'ESPN_PLAYER_ID':str(entry['athlete']['id']), 'PLAYER_NAME':entry['athlete']['displayName'],
                           'GAME_ID':str(header['id']), 'GAME_DATE':competition['date'],
                           'Team':teams[team_id], 'Opponent':opponent, 'MIN':minutes}
                    for stat in ['PTS','AST','REB','OREB','DREB','STL','BLK','TOV']:
                        row[stat] = float(values['TO' if stat == 'TOV' else stat])
                    # Optional fouls must not discard otherwise usable box-score rows.
                    try:
                        fouls = float(values.get('PF', 'nan'))
                        row['PF'] = fouls if fouls >= 0 and fouls.is_integer() else float('nan')
                    except (ValueError, TypeError):
                        row['PF'] = float('nan')
                    for source, made, attempted in [('FG','FGM','FGA'),('3PT','FG3M','FG3A'),('FT','FTM','FTA')]:
                        row[made], row[attempted] = map(float,values[source].split('-'))
                    rows.append(row)
                except (KeyError, ValueError):
                    continue
    return rows


def validate_team_boxscores(payload, rows):
    """Reject partial player coverage before it can enter current team rates."""
    if not rows:
        return  # Includes the explicitly excluded NBA Cup final.
    frame = pd.DataFrame(rows)
    if frame.duplicated(['GAME_ID','ESPN_PLAYER_ID']).any():
        raise ValueError('Duplicate box-score player rows')
    box = frame.groupby('Team').sum(numeric_only=True)
    if len(box) != 2 or len(payload.get('boxscore',{}).get('teams',[])) != 2:
        raise ValueError('Incomplete team box scores')
    for team in payload['boxscore']['teams']:
        abbreviation = team['team']['abbreviation']
        abbreviation = ALIASES.get(abbreviation, abbreviation)
        stats = {s['name']:s['displayValue'] for s in team['statistics']}
        for key,cols in [('fieldGoalsMade-fieldGoalsAttempted',['FGM','FGA']),
                         ('threePointFieldGoalsMade-threePointFieldGoalsAttempted',['FG3M','FG3A']),
                         ('freeThrowsMade-freeThrowsAttempted',['FTM','FTA'])]:
            if not np.allclose(box.loc[abbreviation,cols], list(map(float,stats[key].split('-')))):
                raise ValueError(f'{abbreviation}: incomplete {key} coverage')
        for key,col in [('assists','AST'),('offensiveRebounds','OREB'),('defensiveRebounds','DREB'),('totalRebounds','REB')]:
            if box.loc[abbreviation,col] != float(stats[key]):
                raise ValueError(f'{abbreviation}: incomplete {col} coverage')
    for competitor in payload['header']['competitions'][0]['competitors']:
        abbreviation = competitor['team']['abbreviation']
        if box.loc[ALIASES.get(abbreviation,abbreviation),'PTS'] != float(competitor['score']):
            raise ValueError('Incomplete box-score points coverage')


def verified_box(payload, year, event_id):
    if str(payload.get('header', {}).get('id')) != event_id:
        raise ValueError('Box score returned a different game')
    rows = parse_boxscore(payload, year)
    validate_team_boxscores(payload, rows)
    return rows


@lru_cache(maxsize=1400)
def cached_box(path, year, event_id, modified_ns):
    return verified_box(json.loads(path.read_bytes()), year, event_id)


@lru_cache(maxsize=2)
def bundled_box_rows(year):
    path = ROOT / 'data' / f'game_logs_{year}.csv.gz'
    try:
        metadata = json.loads(path.with_suffix('.source.json').read_bytes())
        if metadata.get('season_ending_year') != year or hashlib.sha256(path.read_bytes()).hexdigest() != metadata['sha256']:
            return None
        frame = pd.read_csv(path, usecols=COLUMNS, dtype={'GAME_ID': str, 'ESPN_PLAYER_ID': str})
        if len(frame) != metadata['player_games'] or frame.GAME_ID.nunique() != metadata['loaded_games']:
            return None
        return frame
    except (OSError, TypeError, ValueError, KeyError):
        return None  # Live validation remains available without the bundled checkpoint.


def load_game_logs(year, espn_team_ids, cache_dir=None, *, game_counts=None, team_cache=None, archive=None):
    """Fetch each completed game once. Per-event season guards reject silent feed fallback."""
    errors, events = [], set()
    def schedule(team_id):
        def fetch_completed():
            payload = fetch_json(f'{ESPN}/site/v2/sports/basketball/nba/teams/{team_id}/schedule?season={year}&seasontype=2')
            if str(payload.get('team', {}).get('id')) != str(team_id):
                raise ValueError('Schedule returned the wrong team')
            return {str(event['id']) for event in payload['events']
                    if event.get('season', {}).get('year') == year
                    and int(event.get('seasonType', {}).get('type', 0)) == 2
                    and event.get('competitions', [{}])[0].get('status', {}).get('type', {}).get('completed')}
        if team_cache is None or game_counts is None:
            return fetch_completed()
        games = game_counts[str(team_id)]
        return team_cache.get(('schedule', year, str(team_id)), games, fetch_completed, lambda ids: len(ids) == games)
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = {pool.submit(schedule, team):str(team) for team in espn_team_ids}
        for job in as_completed(jobs):
            try:
                events.update(job.result())
            except Exception as exc:
                errors.append(f'Schedule {jobs[job]}: {exc}')
    def box(event_id):
        path = cache_dir / f'{event_id}.json' if cache_dir else None
        if path and path.exists():
            try:
                rows = cached_box(path, year, event_id, path.stat().st_mtime_ns)
                if rows:
                    return rows
            except (KeyError, TypeError, ValueError, OSError):
                pass  # Retry a partial or corrupt download from an older app version.
        payload = fetch_json(f'{ESPN}/site/v2/sports/basketball/nba/summary?event={event_id}')
        rows = verified_box(payload, year, event_id)
        if path and rows:
            with NamedTemporaryFile(dir=cache_dir, delete=False, mode='w') as tmp:
                temp_path = Path(tmp.name)
                try:
                    json.dump(payload, tmp)
                    tmp.close()
                    temp_path.replace(path)
                finally:
                    temp_path.unlink(missing_ok=True)
        return rows
    # Use only games that the current season schedules still list as completed.
    archived = archive.loc[archive.GAME_ID.isin(events)] if archive is not None else pd.DataFrame()
    pending = events.difference(archived.GAME_ID.unique()) if not archived.empty else events
    rows = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = {pool.submit(box, event_id):event_id for event_id in pending}
        for i,job in enumerate(as_completed(jobs),1):
            try:
                rows.extend(job.result())
            except Exception as exc:
                errors.append(f'Game {jobs[job]}: {exc}')
            if cache_dir and i % 100 == 0:
                print(f'{i}/{len(events)} completed box scores',flush=True)
    frame = pd.DataFrame(rows, columns=COLUMNS)
    if not archived.empty:
        frame = pd.concat([archived, frame], ignore_index=True) if rows else archived
    frame = (frame.drop_duplicates(['ESPN_PLAYER_ID','GAME_ID'])
             .sort_values(['GAME_DATE', 'GAME_ID', 'ESPN_PLAYER_ID']).reset_index(drop=True))
    return frame, {'season_ending_year':year,'scheduled_completed_games':len(events),
                   'loaded_games':int(frame.GAME_ID.nunique()),'player_games':len(frame),'errors':errors,
                   'retrieved_at':datetime.now(timezone.utc).isoformat(),
                   'source':'ESPN team schedules and completed regular-season summary box scores',
                   'endpoint':f'{ESPN}/site/v2/sports/basketball/nba/summary?event={{event_id}}'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year',type=int,default=2026)
    parser.add_argument('--cache',type=str,required=True)
    args = parser.parse_args()
    from pathlib import Path
    from live_projections import load_live_state
    cache = Path(args.cache); cache.mkdir(parents=True,exist_ok=True)
    teams = load_live_state('2026-27')['teams']
    frame, metadata = load_game_logs(args.year, teams.ESPN_TEAM_ID, cache)
    target = ROOT / 'data' / f'game_logs_{args.year}.csv.gz'
    frame.sort_values(['GAME_DATE','GAME_ID','ESPN_PLAYER_ID']).to_csv(target,index=False,compression='gzip')
    metadata['sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
    metadata['excluded_non_statistical_games'] = ['NBA Cup Championship (excluded despite ESPN season type 2)']
    target.with_suffix('.source.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))
