"""Per-game edge rankings using verified events and existing prop comparisons."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from import_minutes import TEAM_IDS
from sportsbook_lines import TEAM_NAMES, STAT_ORDER, OddsFeedError, player_quotes, compare_with_book

EDGE_STATS = [stat for stat in STAT_ORDER if stat != 'TOV']

EDGE_COLUMNS = ['Rank','Player','Team','Opponent','Stat','Play','Projected minutes',
                'Projected','Book line','Book implied total','Edge %','Over odds',
                'Under odds','Updated UTC','Off samples','Def samples','Stat baseline','Matchup data']


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
            matched = matched[matched.GameTimeUTC.map(lambda value: abs(pd.Timestamp(value)-start) <= pd.Timedelta(minutes=5))]
            matched = matched[~matched.GameStatus.str.lower().str.contains('postpon|cancel|suspend|delay',na=False)]
            if matched.empty:
                continue
            games.append({'id':event['id'],'Home':home,'Away':away,'GameTimeUTC':start,
                          'home_team':event['home_team'],'away_team':event['away_team']})
        except (KeyError,ValueError,TypeError):
            continue
    # A duplicate event ID must not create a second game selector option.
    return sorted({g['id']:g for g in games}.values(),key=lambda g:(g['GameTimeUTC'],g['Away']))


def game_edges(matchups, payload, event, bookmaker, now=None, plays_only=True, direction='Both', stats=None):
    """Top 20 distinct player-stat edges in one event/book, strongest absolute % first.

    Both over and under use the signed projection/implied-mean difference. This
    metric is not win-probability edge or expected return on a wager.
    """
    if payload.get('id') != event['id'] or (payload.get('home_team'),payload.get('away_team')) != (event['home_team'],event['away_team']):
        raise OddsFeedError('Sportsbook matchup could not be verified.')
    if direction not in ['Both','Over','Under']:
        raise ValueError('Invalid edge direction')
    selected_stats = set(EDGE_STATS if stats is None else stats)
    if not selected_stats.issubset(EDGE_STATS):
        raise ValueError('Unknown stat category')
    rows = []
    for _,player in matchups.iterrows():
        expected = event['Away'] if player.Team == event['Home'] else event['Home'] if player.Team == event['Away'] else None
        if expected is None or player.get('Opponent') != expected:
            continue
        if not np.isfinite(player['Projected minutes']) or player['Projected minutes'] <= 0:
            continue
        if player.get('Status') != 'Available' or player.get('Matchup status') != 'Available':
            continue
        preview = pd.DataFrame({'Stat':STAT_ORDER,
            'Projected':[player.get('FG3M' if stat=='3PM' else stat,np.nan) for stat in STAT_ORDER]})
        comparison = compare_with_book(preview,player_quotes(payload,player.Player,bookmaker,now=now))
        comparison = comparison[comparison.Stat.isin(selected_stats) & comparison['Market status'].eq('Current') & comparison['Edge %'].notna()]
        if plays_only:
            comparison = comparison[comparison.Play.ne('')]
        if direction == 'Over':
            comparison = comparison[comparison['Edge %'].gt(0)]
        elif direction == 'Under':
            comparison = comparison[comparison['Edge %'].lt(0)]
        for field in ['Player','Team','Opponent','Projected minutes','Off samples','Def samples','Stat baseline','Matchup data']:
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
