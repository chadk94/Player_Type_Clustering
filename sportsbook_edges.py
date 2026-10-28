"""Per-game edge rankings using verified events and existing prop comparisons."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from import_minutes import TEAM_IDS
from live_projections import name_key
from projection_presentation import matchup_rating
from team_rebounds import REBOUND_METADATA
from matchup_projections import MATCHUP_METADATA
from sportsbook_lines import TEAM_NAMES, STAT_ORDER, OddsFeedError, player_quotes, compare_with_book

EDGE_STATS = [stat for stat in STAT_ORDER if stat != 'TOV']
# Books can move from the advertised start to the expected actual tip near game time.
EVENT_TIME_TOLERANCE = pd.Timedelta(minutes=15)

EDGE_COLUMNS = ['Rank','Player','Team','Opponent','Stat','Play','Projected minutes',
                'Baseline','Matchup rating','Projected','Book line','Book implied total','Edge %','Signed edge %','Over odds',
                'Under odds','Updated UTC','Off samples','Def samples','Stat baseline','Matchup data',*MATCHUP_METADATA,'Pre-market PTS','Pre-market PRA','Market PTS factor','Market team total','Pre-team AST','Team AST factor','Team AST target','Team assist rate','Expected team FGM','Assist share','Pre-team 3PM','Pre-team 3PA','Team 3PM factor','Team 3PM target','Team three rate','Three-point share',*REBOUND_METADATA]


def scheduled_book_games(events, schedule, now=None):
    """Only future book events matching an ESPN regular-season pair and tip time."""
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    games = []
    if schedule.empty:
        return games
    for event in events:
        try:
            home,away = TEAM_NAMES[event['home_team']],TEAM_NAMES[event['away_team']]
            start = pd.Timestamp(event['commence_time'])
            if start.tzinfo is None or start <= now:
                continue
            matched = schedule[schedule.TEAM_ID.map({value:abbr for abbr,value in TEAM_IDS.items()}).eq(home) &
                               schedule.Opponent.eq(away) & schedule.Venue.eq('home')]
            if matched.empty:
                continue
            matched = matched[matched.GameTimeUTC.map(lambda value: abs(pd.Timestamp(value)-start) <= EVENT_TIME_TOLERANCE)]
            # A later book tip must never keep an already-started game eligible.
            matched = matched[matched.GameStatus.str.strip().str.lower().eq('scheduled')]
            if schedule.attrs.get('schedule_is_backup'):
                # A backup cannot verify a late tip or a still-scheduled status.
                matched = matched[matched.GameTimeUTC.map(lambda value: pd.Timestamp(value) > now)]
            if matched.empty:
                continue
            games.append({'id':event['id'],'Home':home,'Away':away,'GameTimeUTC':start,
                          'home_team':event['home_team'],'away_team':event['away_team']})
        except (KeyError,ValueError,TypeError):
            continue
    # A duplicate event ID must not create a second game selector option.
    return sorted({g['id']:g for g in games}.values(),key=lambda g:(g['GameTimeUTC'],g['Away']))


def game_edges(matchups, payload, event, bookmaker, now=None, plays_only=True, direction='Both', stats=None, player_search=''):
    """Top 20 distinct player-stat edges in one event/book, strongest absolute % first.

    Both over and under display the magnitude of the projection/implied-mean gap.
    The signed gap determines direction and is retained in exports. This
    metric is not win-probability edge or expected return on a wager.
    """
    if payload.get('id') != event['id'] or (payload.get('home_team'),payload.get('away_team')) != (event['home_team'],event['away_team']):
        raise OddsFeedError('Sportsbook matchup could not be verified.')
    if direction not in ['Both','Over','Under']:
        raise ValueError('Invalid edge direction')
    selected_stats = set(EDGE_STATS if stats is None else stats)
    if not selected_stats.issubset(EDGE_STATS):
        raise ValueError('Unknown stat category')
    query = name_key(player_search)
    rows = []
    for _,player in matchups.iterrows():
        # Filter after team calibration, but before ranking/limiting to 20.
        if player_search.strip() and (not query or query not in name_key(player.Player)):
            continue
        expected = event['Away'] if player.Team == event['Home'] else event['Home'] if player.Team == event['Away'] else None
        if expected is None or player.get('Opponent') != expected:
            continue
        if not np.isfinite(player['Projected minutes']) or player['Projected minutes'] <= 0:
            continue
        if player.get('Status') != 'Available' or player.get('Matchup status') != 'Available':
            continue
        preview = pd.DataFrame({'Stat':STAT_ORDER,
            'Projected':[player.get('FG3M' if stat=='3PM' else stat,np.nan) for stat in STAT_ORDER]})
        preview['Baseline'] = [sum(player.get(f'Baseline {part}',np.nan) for part in ['PTS','REB','AST']) if stat=='PRA' else player.get('Baseline '+('FG3M' if stat=='3PM' else stat),np.nan) for stat in STAT_ORDER]
        preview['Matchup rating'] = [matchup_rating(base,player.get('Pre-market '+stat,projected) if stat in ['PTS','PRA'] else player.get('Pre-team '+stat,projected) if stat in ['AST','REB','3PM'] else projected,stat) for base,projected,stat in zip(preview.Baseline,preview.Projected,preview.Stat)]
        comparison = compare_with_book(preview,player_quotes(payload,player.Player,bookmaker,now=now))
        comparison = comparison[comparison.Stat.isin(selected_stats) & comparison['Market status'].eq('Current') & comparison['Edge %'].notna()]
        if plays_only:
            comparison = comparison[comparison.Play.ne('')]
        if direction == 'Over':
            comparison = comparison[comparison['Signed edge %'].gt(0)]
        elif direction == 'Under':
            comparison = comparison[comparison['Signed edge %'].lt(0)]
        for field in ['Player','Team','Opponent','Projected minutes','Off samples','Def samples','Stat baseline','Matchup data',*MATCHUP_METADATA,'Pre-market PTS','Pre-market PRA','Market PTS factor','Market team total','Pre-team AST','Team AST factor','Team AST target','Team assist rate','Expected team FGM','Assist share','Pre-team 3PM','Pre-team 3PA','Team 3PM factor','Team 3PM target','Team three rate','Three-point share',*REBOUND_METADATA]:
            comparison[field] = player.get(field,np.nan)
        rows.append(comparison)
    if not rows:
        return pd.DataFrame(columns=EDGE_COLUMNS),0
    ranked = pd.concat(rows,ignore_index=True).drop_duplicates(['Player','Stat'])
    ranked['_magnitude'] = ranked['Edge %'].abs()
    ranked = ranked.sort_values(['_magnitude','Player','Stat'],ascending=[False,True,True],kind='stable')
    count = len(ranked)
    ranked = ranked.head(20).copy()
    ranked['Rank'] = range(1,len(ranked)+1)
    return ranked[EDGE_COLUMNS],count
