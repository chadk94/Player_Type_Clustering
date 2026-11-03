"""Progressive team rates; selected defensive tendencies phase out their prior."""
import numpy as np
import pandas as pd

from team_assists import historical_assist_rates
from team_threes import historical_three_rates
from team_rebounds import historical_rebound_rates

PRIOR_GAMES = 10.
DEFENSIVE_PRIOR_GAMES = 5.
DEFENSIVE_TAPER_START = 8.
DEFENSIVE_FULL_CURRENT_GAMES = 20.
DEFENSIVE_RATE_COLUMNS = {'assists': ('Allowed assist rate',),
                          'threes': ('Allowed three attempt rate',),
                          'rebounds_allowed': ('Offensive rebound share',)}
POLICY_VERSION = 'team-rates-allowed5-taper8-current20-v1'


def defensive_prior_games(games):
    """Keep the approved 5-game preview through eight games; taper to zero at 20."""
    if not np.isfinite(games) or games < 0:
        raise ValueError('Completed games must be finite and nonnegative')
    fraction = np.clip((DEFENSIVE_FULL_CURRENT_GAMES-games)
                       / (DEFENSIVE_FULL_CURRENT_GAMES-DEFENSIVE_TAPER_START),0.,1.)
    return float(DEFENSIVE_PRIOR_GAMES*fraction)


def defensive_current_weight(games):
    prior = defensive_prior_games(games)
    return games/(games+prior) if games else 0.


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
    Three allowed tendencies use a five-game prior through eight completed games,
    then taper that prior to zero at 20. Other rates keep the ten-game prior.
    Each side uses its own team's game count; the matchup remains 50/50.
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
        allowed_weight = defensive_current_weight(games) if complete else 0.
        allowed_prior = defensive_prior_games(games) if valid_count else np.nan
        if complete:
            status = 'Blended'
        elif valid_count and games == 0 and seen == 0 and not invalid:
            status = 'Prior only — no completed games'
        else:
            status = 'Prior only — current team box scores incomplete or unverified'
        rows.append({'Team':team, 'Completed games':games, 'Verified games':seen,
                     'Current weight':weight, 'Prior games':prior_games, 'Status':status,
                     'Defensive current weight':allowed_weight,
                     'Defensive prior games':allowed_prior,
                     'Defensive status':('Current season only' if allowed_weight == 1. else status)})
    metadata = pd.DataFrame(rows).set_index('Team')
    result = {}
    for kind, old in prior.items():
        blended = old.copy(deep=True)
        for team in index[metadata['Current weight'].gt(0)]:
            weight = metadata.loc[team,'Current weight']
            blended.loc[team] = (1-weight)*old.loc[team] + weight*current[kind].loc[team]
            for column in DEFENSIVE_RATE_COLUMNS.get(kind,()):
                allowed_weight = metadata.loc[team,'Defensive current weight']
                blended.loc[team,column] = (current[kind].loc[team,column] if allowed_weight == 1.
                    else (1-allowed_weight)*old.loc[team,column]+allowed_weight*current[kind].loc[team,column])
        result[kind] = blended
    result['coverage'] = metadata
    return result
