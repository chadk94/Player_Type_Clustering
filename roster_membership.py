"""In-season team changes for the workbook roster. Dates are NBA game dates."""
from datetime import date

from import_minutes import TEAM_IDS


# Oklahoma City announced the acquisition on November 8, 2026:
# https://www.nba.com/thunder/news/nique-clifford-261108
ROSTER_MOVES = (('2026-27', 1642363, date(2026, 11, 8), 'SAC', 'OKC'),)


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
        mask = result.PLAYER_ID.eq(player_id) & result.TEAM_ID.ne(TEAM_IDS[team])
        if mask.any():
            if result is roster:
                result = roster.copy()
            result.loc[mask, 'TEAM_ID'] = TEAM_IDS[team]
            if 'TEAM_ABBREVIATION' in result:
                result.loc[mask, 'TEAM_ABBREVIATION'] = team
    return result
