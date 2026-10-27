"""Progressive season-rate shrinkage, with last season worth ten team games."""
import numpy as np
import pandas as pd

from team_assists import historical_assist_rates
from team_threes import historical_three_rates
from team_rebounds import historical_rebound_rates

PRIOR_GAMES = 10.
POLICY_VERSION = 'team-rates-10-game-prior-attempts-v1'


def season_rates(logs, expected_games=None):
    own, allowed = historical_rebound_rates(logs, expected_games)
    return {'assists': historical_assist_rates(logs, expected_games).drop(columns='Games'),
            'threes': historical_three_rates(logs, expected_games).drop(columns='Games'),
            'rebounds_own': own, 'rebounds_allowed': allowed}


def blend_team_rates(prior, current_logs, teams, prior_games=PRIOR_GAMES):
    """Posterior rate = (prior_games * prior + games * current)/(prior_games+games).

    This is game-equivalent empirical-Bayes shrinkage of season ratios, not
    pooling 82 historical games or treating player appearances as team games.
    A team needs every completed game in the snapshot before using current rates.
    Both own/allowed rates use that team's weight; the matchup remains 50/50.
    """
    if not np.isfinite(prior_games) or prior_games <= 0:
        raise ValueError('Prior games must be positive')
    index = prior['assists'].index
    if any(not frame.index.equals(index) for frame in prior.values()):
        raise ValueError('Historical rate coverage differs between categories')
    expected = pd.Series(dtype=float)
    if {'TEAM_ABBREVIATION','GAMES_PLAYED'}.issubset(teams):
        if teams.TEAM_ABBREVIATION.duplicated().any():
            raise ValueError('Duplicate current team records')
        expected = pd.to_numeric(teams.set_index('TEAM_ABBREVIATION').GAMES_PLAYED, errors='coerce')
    loaded = pd.Series(dtype=float)
    current = {}
    invalid = False
    if not current_logs.empty:
        try:
            if {'GAME_ID','ESPN_PLAYER_ID'}.issubset(current_logs) and current_logs.duplicated(['GAME_ID','ESPN_PLAYER_ID']).any():
                raise ValueError('Duplicate player appearances')
            current = season_rates(current_logs)
            loaded = current_logs.groupby('Team').GAME_ID.nunique()
        except (ValueError, KeyError, TypeError):
            invalid = True
    rows = []
    for team in index:
        games, seen = expected.get(team, np.nan), loaded.get(team, 0)
        valid_count = pd.notna(games) and np.isfinite(games) and games >= 0 and games == int(games)
        complete = (valid_count and seen == games and games > 0 and bool(current)
                    and all(team in frame.index and np.isfinite(frame.loc[team]).all() for frame in current.values()))
        weight = games/(prior_games+games) if complete else 0.
        if complete:
            status = 'Blended'
        elif valid_count and games == 0 and seen == 0 and not invalid:
            status = 'Prior only — no completed games'
        else:
            status = 'Prior only — current team box scores incomplete or unverified'
        rows.append({'Team':team, 'Completed games':games, 'Verified games':seen,
                     'Current weight':weight, 'Prior games':prior_games, 'Status':status})
    metadata = pd.DataFrame(rows).set_index('Team')
    result = {}
    for kind, old in prior.items():
        blended = old.copy(deep=True)
        for team in index[metadata['Current weight'].gt(0)]:
            weight = metadata.loc[team,'Current weight']
            blended.loc[team] = (1-weight)*old.loc[team] + weight*current[kind].loc[team]
        result[kind] = blended
    result['coverage'] = metadata
    return result
