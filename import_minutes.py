"""Read the user's minutes workbook without modifying it."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

# Stable NBA franchise IDs; workbook membership is the user's roster scenario.
TEAM_IDS = {
    'ATL': 1610612737, 'BOS': 1610612738, 'CLE': 1610612739,
    'NOP': 1610612740, 'CHI': 1610612741, 'DAL': 1610612742,
    'DEN': 1610612743, 'GSW': 1610612744, 'HOU': 1610612745,
    'LAC': 1610612746, 'LAL': 1610612747, 'MIA': 1610612748,
    'MIL': 1610612749, 'MIN': 1610612750, 'BKN': 1610612751,
    'NYK': 1610612752, 'ORL': 1610612753, 'IND': 1610612754,
    'PHI': 1610612755, 'PHX': 1610612756, 'POR': 1610612757,
    'SAC': 1610612758, 'SAS': 1610612759, 'OKC': 1610612760,
    'TOR': 1610612761, 'UTA': 1610612762, 'MEM': 1610612763,
    'WAS': 1610612764, 'DET': 1610612765, 'CHA': 1610612766,
}


def normalize_rows(rows, season_id=22026):
    result, seen = [], set()
    for row_number, row in enumerate(rows, start=2):
        if all(value is None for value in row.values()):
            continue
        team = str(row['Team']).strip().upper()
        player_id = float(row['Player ID'])
        minutes = float(row['Minutes'])
        if team not in TEAM_IDS or not math.isfinite(player_id) or player_id <= 0 or not player_id.is_integer():
            raise ValueError(f'Invalid NBA identifiers at Minutes row {row_number}')
        player_id = int(player_id)
        if player_id in seen:
            raise ValueError(f'Duplicate player ID at Minutes row {row_number}')
        if not math.isfinite(minutes) or not 0 <= minutes <= 48:
            raise ValueError(f'Invalid minutes at Minutes row {row_number}')
        if not row['Player']:
            raise ValueError(f'Missing player name at Minutes row {row_number}')
        seen.add(player_id)
        result.append({'PLAYER_ID': player_id, 'TEAM_ID': TEAM_IDS[team],
                       'SEASON_ID': season_id, 'PROJECTED_MIN': minutes,
                       'PLAYER_NAME': str(row['Player']).strip(), 'TEAM_ABBREVIATION': team,
                       'Position': row.get('Position'), 'Starter': row.get('Starter'),
                       'StarterSource': row.get('Starter Source'),
                       'SourceSheet': 'Minutes', 'SourceRow': row_number})
    if not result:
        raise ValueError('No minutes rows')
    return result


def extract(workbook_path, output, season_id=22026):
    import openpyxl  # Read-only input dependency; no workbook authoring.
    workbook = openpyxl.load_workbook(workbook_path, data_only=True, read_only=True)
    sheet = workbook['Minutes']
    values = iter(sheet.values)
    headers = next(values)
    required = {'Team', 'Player', 'Minutes', 'Player ID'}
    if not required.issubset(headers):
        raise ValueError('Missing required Minutes headers')
    rows = normalize_rows((dict(zip(headers, row)) for row in values), season_id)
    workbook.close()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'projected_minutes.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    totals = {}
    for row in rows:
        totals[row['TEAM_ABBREVIATION']] = totals.get(row['TEAM_ABBREVIATION'], 0) + row['PROJECTED_MIN']
    manifest = {'workbook': str(Path(workbook_path).resolve()),
                'sha256': hashlib.sha256(Path(workbook_path).read_bytes()).hexdigest(),
                'sheet': 'Minutes', 'range': f'A1:G{max(r["SourceRow"] for r in rows)}',
                'rows': len(rows), 'teams': len(totals), 'team_minutes': totals,
                'all_teams_240': all(abs(total - 240) < 1e-8 for total in totals.values()),
                'precision': 'Original stored values, no rescaling or rounding',
                'roster_policy': 'User workbook team membership; not externally verified rosters'}
    (output / 'minutes_import.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workbook')
    parser.add_argument('--output', default='.')
    parser.add_argument('--season-id', type=int, default=22026)
    args = parser.parse_args()
    rows = extract(args.workbook, args.output, args.season_id)
    print(f'Extracted {len(rows)} player projections')
