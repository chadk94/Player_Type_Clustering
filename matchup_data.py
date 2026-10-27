"""Validated ESPN regular-season box scores for archetype matchup estimates."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import json
import hashlib
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


def load_game_logs(year, espn_team_ids, cache_dir=None):
    """Fetch each completed game once. Per-event season guards reject silent feed fallback."""
    errors, events = [], {}
    def schedule(team_id):
        return fetch_json(f'{ESPN}/site/v2/sports/basketball/nba/teams/{team_id}/schedule?season={year}&seasontype=2')
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = {pool.submit(schedule, team):str(team) for team in espn_team_ids}
        for job in as_completed(jobs):
            try:
                for event in job.result().get('events', []):
                    if event.get('season', {}).get('year') == year and int(event.get('seasonType', {}).get('type',0)) == 2:
                        comp = event.get('competitions',[{}])[0]
                        if comp.get('status',{}).get('type',{}).get('completed'):
                            events[str(event['id'])] = event
            except Exception as exc:
                errors.append(f'Schedule {jobs[job]}: {exc}')
    def box(event_id):
        path = cache_dir / f'{event_id}.json' if cache_dir else None
        if path and path.exists():
            payload = json.loads(path.read_text())
        else:
            payload = fetch_json(f'{ESPN}/site/v2/sports/basketball/nba/summary?event={event_id}')
            if path:
                path.write_text(json.dumps(payload))
        rows = parse_boxscore(payload,year)
        validate_team_boxscores(payload, rows)
        return rows
    rows = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = {pool.submit(box, event_id):event_id for event_id in events}
        for i,job in enumerate(as_completed(jobs),1):
            try:
                rows.extend(job.result())
            except Exception as exc:
                errors.append(f'Game {jobs[job]}: {exc}')
            if cache_dir and i % 100 == 0:
                print(f'{i}/{len(events)} completed box scores',flush=True)
    frame = pd.DataFrame(rows, columns=COLUMNS).drop_duplicates(['ESPN_PLAYER_ID','GAME_ID'])
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
