"""In-season team changes for the workbook roster. Dates are NBA game dates."""
from datetime import date
from pathlib import Path

import pandas as pd

from import_minutes import TEAM_IDS


# None represents the interval between a waiver and a signing.
ROSTER_MOVES = (
    ('2026-27', 1642383, date(2026, 10, 19), 'MEM', None),
    ('2026-27', 1642383, date(2026, 10, 21), None, 'DAL'),
    ('2026-27', 1642363, date(2026, 11, 8), 'SAC', 'OKC'),
)


def add_roster_profiles(roster, season):
    """Add reviewed in-season identities without replacing saved roster rows.

    Zero is a reserve default, never a new allocation. Approved game rotations
    and the session's Tabs blend supply minutes through the normal paths.
    """
    path = Path(__file__).parent / 'data/roster_additions.csv'
    additions = pd.read_csv(path)
    additions = additions.loc[additions.Season.eq(season) & ~additions.PLAYER_ID.isin(roster.PLAYER_ID)].copy()
    if additions.empty:
        return roster
    if additions.PLAYER_ID.duplicated().any() or additions.PLAYER_NAME.duplicated().any():
        raise ValueError('In-season roster additions must have unique player identities')
    additions['TEAM_ID'] = additions.TEAM_ABBREVIATION.map(TEAM_IDS)
    if additions.TEAM_ID.isna().any():
        raise ValueError('Unknown team in roster additions')
    additions['SEASON_ID'] = int(season[:4]) + 20000
    additions['WorkbookPlayerName'] = additions.PLAYER_NAME
    additions['PROJECTED_MIN'] = 0.
    additions['HasHistoricalProfile'] = False
    additions['ModelEligibleMinutes'] = False
    additions['FeatureDataStatus'] = 'provisional_in_season_profile'
    for side in ('Off', 'Def'):
        additions[side+'AssignmentSource'] = 'provisional_in_season_review'
    return pd.concat([roster, additions.reindex(columns=roster.columns)], ignore_index=True)


def roster_on_date(roster, season, game_day):
    """Carry a player's existing profile to the team that owned it on game day."""
    result = roster
    destinations = {}
    for move_season, player_id, effective, previous, new in sorted(ROSTER_MOVES, key=lambda m: m[2]):
        if season == move_season:
            destinations.setdefault(player_id, previous)
            if effective <= game_day:
                destinations[player_id] = new
    for player_id, team in destinations.items():
        # Zero cannot match a team in an approved NBA rotation.
        team_id = TEAM_IDS[team] if team is not None else 0
        mask = result.PLAYER_ID.eq(player_id) & result.TEAM_ID.ne(team_id)
        if mask.any():
            if result is roster:
                result = roster.copy()
            result.loc[mask, 'TEAM_ID'] = team_id
            if 'TEAM_ABBREVIATION' in result:
                result.loc[mask, 'TEAM_ABBREVIATION'] = team
    return result


def roster_for_approved_games(roster, season, game_day, bundle):
    """Use identity-verified linked approvals for membership on that slate only.

    Always pass the source roster, not yesterday's derived game roster. The
    authenticated NBA-model approval is the owner's roster for this exact game;
    player history, labels, manual overrides and baseline minutes are untouched.
    """
    result = roster_on_date(roster, season, game_day)
    if not bundle or bundle.get('schema') != 'nba_linked_minutes_v1':
        return result
    destinations = {p['player_id']: a['team_id'] for a in bundle['approvals']
                    if a['game_date'] == str(game_day) for p in a['players']}
    teams = result.PLAYER_ID.map(destinations)
    changed = teams.notna() & teams.ne(result.TEAM_ID)
    if changed.any():
        result = result.copy()
        result.loc[changed, 'TEAM_ID'] = teams[changed].astype(int)
        result.loc[changed, 'TEAM_ABBREVIATION'] = teams[changed].map({v:k for k,v in TEAM_IDS.items()})
    return result
