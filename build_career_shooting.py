"""Build the public, season-scoped career shooting snapshot offline.

Usage: python build_career_shooting.py --identities identities.json --cache DIR
Identity JSON: [{"PLAYER_ID":123,"PLAYER_NAME":"Name","ESPN_PLAYER_ID":"456"}].
Never fetch career histories during a dashboard interaction.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path

import requests

from career_shooting import parse_career_totals


def run(identities, cache, season, output):
    cache.mkdir(parents=True, exist_ok=True)
    def read(identity):
        espn_id = identity['ESPN_PLAYER_ID']
        path = cache / f'{espn_id}.json'
        url = f'https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{espn_id}/stats?seasontype=2&league=nba'
        if path.exists():
            payload = json.loads(path.read_text())
        else:
            response = requests.get(url, timeout=(5, 20))
            response.raise_for_status()
            payload = response.json()
            path.write_text(json.dumps(payload))
        totals = parse_career_totals(payload, season)
        return str(identity['PLAYER_ID']), {**identity, **totals, 'source_url': url}
    players, errors = {}, []
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = {pool.submit(read, identity): identity for identity in identities}
        for i, job in enumerate(as_completed(jobs), 1):
            try:
                key, result = job.result()
                if key in players:
                    raise ValueError('Duplicate player identity')
                players[key] = result
            except Exception as exc:
                errors.append({'player': jobs[job], 'error': f'{type(exc).__name__}: {exc}'})
            if i % 50 == 0:
                print(f'{i}/{len(jobs)} checked; {len(errors)} unresolved', flush=True)
    payload = {'target_season': season, 'checked_at': datetime.now(timezone.utc).isoformat(),
               'scope': 'NBA regular season, all completed seasons before target season; no preseason/playoffs/current season',
               'players': dict(sorted(players.items())), 'errors': errors}
    output.write_text(json.dumps(payload, indent=2)+'\n')
    print(f'{len(players)} players saved; {len(errors)} unresolved', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identities', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--season', default='2026-27')
    parser.add_argument('--output', type=Path, default=Path('data/career_three_point_totals.json'))
    args = parser.parse_args()
    run(json.loads(args.identities.read_text()), args.cache, args.season, args.output)
