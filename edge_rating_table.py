"""Edge-ranked tables with one-click confidence controls and matchup evidence."""
from collections import defaultdict
from functools import lru_cache
from math import isfinite
from pathlib import Path

import pandas as pd
import streamlit as st

from edge_calculation_cache import GameCalculationCache
from matchup_projections import RECENT_OPPONENT_GAMES
from matchup_evidence import peer_panels
from points_probability import PROBABILITY_COLUMNS
from sportsbook_lines import comparison_style, PLAY_THRESHOLD_PCT, FANDUEL_THREES_OVER_ONLY
from todays_plays import edge_pick

EDGE_TABLE_UI_VERSION = 'tabs_price_agreement_v3'

PLAYER_PREVIEW_STATS = {'PTS': 'PTS', 'REB': 'REB', 'AST': 'AST', '3PM': 'FG3M',
                        'STL': 'STL', 'BLK': 'BLK', 'TOV': 'TOV', 'PF': 'PF', 'PRA': 'PRA'}

# Match the component statistics used by apply_matchup, including the attempts
# effect on made threes. Keep the two rebound populations separate.
ROLE_STATS = {
    'PTS': {'Off': ('PTS',)}, 'AST': {'Off': ('AST',)},
    '3PM': {'Off': ('FG3A',)}, 'STL': {'Def': ('STL',)}, 'BLK': {'Def': ('BLK',)},
    'REB': {'Off': ('OREB',), 'Def': ('DREB',)},
    'PRA': {'Off': ('PTS', 'AST', 'OREB'), 'Def': ('DREB',)},
}
FORMATS = {'Projected minutes': '.1f', 'Reference minutes': '.1f', 'Good-to minutes': 'g', 'Projected': '.2f', 'Recent role': '.2f',
           'Recent change %': '+.1f', 'Book implied total': '.2f', 'Edge %': '+.2f',
           'Over odds': '+.0f', 'Under odds': '+.0f', **{key: '.1%' for key in PROBABILITY_COLUMNS}}
HEADER_LABELS = {'Rank':'Player #', 'Projected minutes':'Min', 'Reference minutes':'Ref. min',
                 'Good-to minutes':'Good-to min', 'Minutes cushion':'Cushion',
                 'Matchup rating':'Matchup', 'Review flags':'Flags', 'Projected':'Projection',
                 'Book line (Over)':'Line / Over odds', 'Book implied total':'Implied total',
                 'Exp. P(over)':'Over %', 'Exp. P(under)':'Under %', 'Exp. P(push)':'Push %'}
HEADER_HELP = {
    'Projected minutes':'Hover or click to compare your projected minutes with TabsNBA and the minutes used by the model.',
    'Rank':'Player rank in this list, based on their highest qualifying Edge %. All their displayed props stay together.',
    'Edge %':'Players are ordered by their highest qualifying edge; props within each player are highest first. This projection gap is not win probability or expected return.',
    'Good-to minutes':'Minutes at which this side still clears 5% edge: lower limit for an Over, upper limit for an Under. Tested in half-minute steps through 48.',
    'Matchup rating':'1 = poor, 3 = average, 5 = excellent. Hover for 3 recent similar-player performances; click for up to 10.',
    'Review flags':'Minutes unverified = no completed minutes check; Minutes risk = playing time could weaken the bet; ROLE = recent-role risk; Q = availability risk; Tabs agrees = more than 5% price-adjusted edge on our side; Tabs disagrees = more than 5% against; Tabs neutral = within ±5%; green History + supports this pick, red History − opposes it (informational); ? = unchecked. Click a badge for details.',
    'Book line (Over)':'Sportsbook line followed by the Over price. Under prices appear in the next column.',
    'Exp. P(over)':'Experimental probability of going Over, conditional on playing. Whole-number lines can push.',
    'Exp. P(under)':'Experimental probability of going Under, conditional on playing. Whole-number lines can push.',
}


def minutes_popup(row):
    """Describe the existing stress test; never imply its minutes came from a source."""
    def finite(value):
        return isinstance(value, (int, float)) and isfinite(value)

    if row.get('Tabs minutes saved',False):
        return dict(detail='Your projection is still using your minutes averaged with the last saved Tabs minutes. Tabs is not currently showing a fresh, complete rotation for this team. Show all teams in Tabs or reconnect it to refresh the minutes check.',
                    technical=str(row.get('Minutes detail','')))

    if row.get('Minutes comparison mode') == 'Both sources':
        units = {'PTS':'points','REB':'rebounds','AST':'assists','3PM':'threes','PRA':'PRA',
                 'STL':'steals','BLK':'blocks'}.get(row.get('Stat'), row.get('Stat',''))
        side = row.get('Play', '')
        parts = []
        for name, label in [('Your','Your minutes'),('Tabs','TabsNBA minutes')]:
            values = [row.get(name+' minutes'),row.get(name+' minutes projection'),row.get(name+' minutes edge %')]
            if all(finite(value) for value in values):
                minutes, projection, edge = values
                edge_label = f'{edge:+.2f}' if round(edge,1)==PLAY_THRESHOLD_PCT and edge!=PLAY_THRESHOLD_PCT else f'{edge:+.1f}'
                parts.append(f'{label}: {minutes:.1f} min → {projection:.2f} {units}, {edge_label}% {side} edge.')
        if len(parts) == 2 and row.get('Minutes flag') != 'Unchecked':
            summary = ('This bet does not clear 5% edge or has zero projected minutes under at least one source. '
                       if row.get('Minutes flag') == 'MIN' else 'This bet clears 5% edge with either minutes source. ')
            summary += '\n' + '\n'.join(parts)
            summary += f'\nBlended model: {row["Projected minutes"]:.1f} min → {row["Projected"]:.2f} {units}.'
        else:
            summary = 'We need complete, current minutes from both your rotation and TabsNBA to check this bet. Your minutes stay in use where the blend is unavailable.'
        return dict(detail=summary, technical=str(row.get('Minutes detail','')))

    reference = row.get('Reference minutes')
    original = row.get('Projected minutes')
    has_reference = finite(reference)
    if has_reference and finite(original):
        summary = (f'Your projection uses {original:g} minutes. '
                   f'{row.get("Reference source", "The comparison source")} has {reference:g} minutes.')
    else:
        summary = 'We haven’t verified your minutes against another source.'
    minutes = row.get('Minutes test minutes')
    value, edge = row.get('Minutes test projection'), row.get('Minutes test edge %')
    side = row.get('Play')
    units = {'PTS':'points', 'REB':'rebounds', 'AST':'assists', '3PM':'threes',
             'STL':'steals', 'BLK':'blocks', 'PRA':'points + rebounds + assists'}.get(row.get('Stat'), row.get('Stat', ''))
    if all(finite(x) for x in (minutes, value, edge)) and side in ('Over', 'Under'):
        edge_label = f'{edge:.2f}' if round(edge, 1) == PLAY_THRESHOLD_PCT and edge != PLAY_THRESHOLD_PCT else f'{edge:.1f}'.rstrip('0').rstrip('.')
        if edge_label == '-0':
            edge_label = '0'
        summary += (f' If he plays {minutes:g} minutes, we still project {value:.2f} {units}, '
                    f'leaving a {edge_label}% {side} edge.')
        if edge <= PLAY_THRESHOLD_PCT:
            summary += f' That no longer clears your {PLAY_THRESHOLD_PCT:g}% cutoff.'
        if not has_reference:
            summary += ' The minutes check is still pending.'
    else:
        summary += ' We couldn’t complete the minutes stress test. The minutes check is still pending.'
    technical = str(row.get('Minutes detail', 'Minutes review unavailable.'))
    technical += '\nThe tested minutes are a what-if scenario, not a new minutes forecast.'
    return dict(detail=summary, technical=technical)


def role_popup(row):
    """Explain the existing role check without changing any model values."""
    stat = row.get('Stat')
    role, units = {
        'PTS': ('scoring', 'points'), '3PM': ('three-point shooting', 'threes'),
        'AST': ('passing', 'assists'), 'PRA': ('scoring and passing', 'points + rebounds + assists'),
        'REB': ('rebounding', 'rebounds'), 'STL': ('defensive', 'steals'),
        'BLK': ('defensive', 'blocks'),
    }.get(stat, ('playing', str(stat)))
    technical = str(row.get('Role detail', 'Recent role history unavailable.'))
    if stat in ('PTS', '3PM'):
        technical = technical.replace(' Passing-opportunity trend unavailable; assists not inferred from raw assist counts.', '')
    player = row.get('Player', 'This player')
    value, edge = row.get('Recent test projection'), row.get('Recent test edge %')
    side = row.get('Play')
    valid = all(isinstance(x, (int, float)) and isfinite(x) for x in (value, edge))
    if row.get('Role flag') == 'ROLE' and valid and side in ('Over', 'Under'):
        # Preserve the strict cutoff even when a rounded number looks like 5.0%.
        edge_label = f'{edge:.2f}' if round(edge, 1) == PLAY_THRESHOLD_PCT and edge != PLAY_THRESHOLD_PCT else f'{edge:.1f}'
        outcome = ('still clears' if edge > PLAY_THRESHOLD_PCT else 'no longer clears')
        if stat == 'REB':
            trend = row.get('Rebound trend %')
            direction = f' ({"down" if trend < 0 else "up"} {abs(trend):.1f}% per minute)' if pd.notna(trend) else ''
            summary = (f'{player}’s recent rebounding trend works against this {side}{direction}. '
                       f'The recent-rate projection is {value:.1f} rebounds at your projected minutes. '
                       f'The remaining {edge_label}% {side} edge {outcome} your {PLAY_THRESHOLD_PCT:g}% cutoff.')
        else:
            summary = (f'{player}’s recent {role} role weakens this {side}. '
                       f'The recent-role projection is {value:.1f} {units}. '
                       f'The {edge_label}% {side} edge {outcome} your {PLAY_THRESHOLD_PCT:g}% cutoff.')
    elif stat in ('STL', 'BLK'):
        summary = (f'Recent-role checks for {units} aren’t available yet. '
                   'This bet has not passed the role check.')
    elif stat == 'AST' and row.get('Role flag') == 'Unchecked':
        summary = f'{player}’s passing-role check is pending. ' + technical.split('\n')[0]
    else:
        summary = (f'{player}’s recent {role} role couldn’t be fully checked. '
                   'There isn’t enough reliable recent data to assess this bet’s role risk.')
    return dict(detail=summary, technical=technical)


def rating_detail(stat, player, opponent, effects):
    sections = []
    window = None
    for side, stats in ROLE_STATS[stat].items():
        name = player.get(side + 'Archetype')
        cluster = player.get(side + 'Cluster')
        recent = (effects.get((side, int(cluster), opponent), {}).get('recent', {})
                  if pd.notna(cluster) else {})
        if recent:
            window = recent['games']
        role = 'Offense' if side == 'Off' else 'Defense'
        lines = [f"{role} · {name if pd.notna(name) else 'Unassigned'}"]
        by_count = defaultdict(list)
        for key in stats:
            value = recent.get('stats', {}).get(key)
            if value is not None:
                average, n = value
                label = '3PA' if key == 'FG3A' else key
                by_count[n].append(f'{average:.1f} {label}')
        for n, values in by_count.items():
            lines.append(' · '.join(values) + f" ({n} player appearance{'s' if n != 1 else ''})")
        if not by_count:
            lines.append('No qualifying results in the available box scores.')
        else:
            missing = [key for key in stats if key not in recent['stats']]
            if missing:
                lines.append('No recent sample for ' + ', '.join(missing) + '.')
        sections.append('\n'.join(lines))
    period = f"last {window} available game{'s' if window != 1 else ''}" if window else f'up to {RECENT_OPPONENT_GAMES} recent games'
    basis = 'Rating uses 3PA per 36 versus the archetype’s usual rate.\n\n' if stat == '3PM' else ''
    return basis + f'vs {opponent} · {period} this season\nAverage per 36 minutes · 5+ minutes played\n\n' + '\n\n'.join(sections)


def group_player_edges(edges):
    """Group displayed props by identity, ranking players by their best remaining edge."""
    if edges.empty:
        return edges.copy()
    ordered = edges.sort_values(['Edge %', 'Player', 'Stat'],
                                ascending=[False, True, True], na_position='last', kind='stable')
    identity = (['Event ID'] if 'Event ID' in ordered else []) + ['Team', 'PLAYER_ID']
    return pd.concat([rows.assign(Rank=rank) for rank, (_, rows) in enumerate(
        ordered.groupby(identity, sort=False, dropna=False), start=1)], ignore_index=True)


def minutes_source_detail(row):
    """Show source allocations, including valid zeroes, without new calculations."""
    def minutes(field):
        try:
            value = float(row.get(field))
            return f'{value:.1f} min' if isfinite(value) and 0 <= value <= 48 else 'Unavailable'
        except (TypeError, ValueError):
            return 'Unavailable'

    lines = [f'Your projection: {minutes("Your minutes")}',
             f'TabsNBA projection: {minutes("Tabs minutes")}',
             f'Used by model: {minutes("Projected minutes")}']
    basis = str(row.get('Tabs minutes basis', ''))
    if basis.startswith('Assumed 0'):
        lines.append('TabsNBA’s 0 is inferred from its complete rotation; this player was not listed.')
    elif basis == 'Explicit OUT':
        lines.append('TabsNBA lists this player as OUT.')
    status = str(row.get('Minutes blend status', ''))
    if row.get('Tabs minutes saved',False):
        stamp=pd.to_datetime(row.get('Tabs source time'),utc=True,errors='coerce')
        time=stamp.tz_convert('America/Denver').strftime('%b %d at %I:%M %p %Z') if pd.notna(stamp) else 'the last successful connection'
        lines.append(f'Using saved Tabs minutes from {time}; live check pending.')
    if status.startswith('Pending'):
        lines.append(status)
    return '\n'.join(lines)


def player_projection_cards(projections):
    """Format the loaded final game frame, independently of qualifying prop filters."""
    if projections is None or projections.empty:
        return {}

    def number(value, decimals=2):
        try:
            value = float(value)
            return f'{value:.{decimals}f}' if isfinite(value) and value >= 0 else '—'
        except (ValueError, TypeError):
            return '—'

    cards = {}
    identity = (['Event ID'] if 'Event ID' in projections else []) + ['PLAYER_ID', 'Team', 'Opponent']
    for key, rows in projections.groupby(identity, sort=False):
        if len(rows) != 1:
            continue  # Never pick an arbitrary conflicting game projection.
        row = rows.iloc[0]
        cards[key] = dict(player=str(row.Player), team=str(row.Team), opponent=str(row.Opponent),
                          minutes=number(row.get('Projected minutes'), 1),
                          minutesDetail=minutes_source_detail(row),
                          stats=[dict(stat=label, value=number(row.get(field)))
                                 for label, field in PLAYER_PREVIEW_STATS.items()])
    return cards


def table_payload(edges, columns, roster, effects, team, game, book, day, projections=None, games=None):
    players = roster.set_index('PLAYER_ID')
    cards = player_projection_cards(projections)
    output = []
    grouped = group_player_edges(edges)
    counts = grouped['Rank'].value_counts()
    last_rank = None
    for _, row in grouped.iterrows():
        event = games[row['Event ID']] if games is not None else game
        card_key = ((row['Event ID'],) if games is not None else ()) + (row.PLAYER_ID, row.Team, row.Opponent)
        card = cards.get(card_key)
        raw = row[columns]
        values = [None if pd.isna(v) else v for v in raw.tolist()]
        labels = []
        for col, v in zip(columns, values):
            if v is None:
                labels.append('—')
            elif col == 'Minutes cushion':
                labels.append(f'{v:.2f}'.rstrip('0').rstrip('.'))
            elif col == 'EV %':
                side = f" ({row['EV side']})" if not row.Play else ''
                labels.append(f'{v:+.2f}%{side}')
            elif col in FORMATS:
                labels.append(format(v, FORMATS[col]) + ('%' if col in ('Edge %', 'Recent change %') else ''))
            else:
                labels.append(str(v))
            if col == 'Book line (Over)' and row.get('Quote basis') == FANDUEL_THREES_OVER_ONLY:
                labels[-1] += f' · {row["Book line"] + .5:g}+ threes'
        player = players.loc[row.PLAYER_ID] if row.PLAYER_ID in players.index else {}
        badges=[]
        for flag,title,color in [('Minutes','MIN','minutes'),('Role','ROLE','role'),('Injury','Q','injury')]:
            status=row.get(flag+' flag')
            if status in ('MIN','ROLE','Q','Unchecked'):
                label = ('Minutes unverified' if status == 'Unchecked' else 'Minutes risk') if flag == 'Minutes' else (title if status != 'Unchecked' else title+'?')
                if flag == 'Minutes' and row.get('Tabs minutes saved',False):label='Saved Tabs minutes'
                popup = minutes_popup(row) if flag == 'Minutes' else role_popup(row) if flag == 'Role' else dict(detail=row.get(flag+' detail','Unchecked'))
                badges.append(dict(label=label,
                    color=color if status!='Unchecked' else 'unchecked',
                    **popup))
        tabs_badges = {'OPPOSES': ('Tabs disagrees', 'tabs'),
                       'Agrees': ('Tabs agrees', 'tabs-agrees'),
                       'Neutral': ('Tabs neutral', 'tabs-neutral')}
        if row.get('Tabs projection flag') in tabs_badges:
            label, color = tabs_badges[row['Tabs projection flag']]
            badges.append(dict(label=label, color=color,
                               detail=row.get('Tabs projection detail',''),
                               technical=row.get('Tabs projection technical','')))
        if row.get('History flag') in ('HISTORY', 'SUPPORTS'):
            supports = row['History flag'] == 'SUPPORTS'
            badges.append(dict(label='History +' if supports else 'History −',
                               color='history-support' if supports else 'history', popupStyle='history',
                               detail=row.get('History detail',''), technical=row.get('History technical','')))
        if not badges and 'Review group' in row:
            badges.append(dict(label='✓' if row.get('Robust play') else 'Review',color='clear' if row.get('Robust play') else 'unchecked',detail=row.get('Review detail','')))
        output.append(dict(pick=edge_pick(row, event, book, day), values=values, labels=labels, styles=comparison_style(raw),
                           quoteDetail=('FanDuel X+ threes, equivalent to this Over line. No Under price is posted. '
                                        'The implied total retains the quoted margin; no vig removal is assumed.'
                                        if row.get('Quote basis') == FANDUEL_THREES_OVER_ONLY else ''),
                           playerId=int(row.PLAYER_ID),
                           projection=card if row.Rank != last_rank else None,
                           minutesDetail=card['minutesDetail'] if card else 'Minutes comparison unavailable for this game.',
                           playerRowSpan=int(counts[row.Rank]) if row.Rank != last_rank else 0,
                           detail=rating_detail(row.Stat, player, row.Opponent, effects),
                           matchup=peer_panels(row.Stat, player, row.Opponent, effects,
                                              ROLE_STATS[row.Stat], event.get('GameTimeUTC')),
                           reviewBadges=badges))
        last_rank = row.Rank
    return dict(columns=columns, rows=output, team=team,
                headers=[HEADER_LABELS.get(c,c) for c in columns],
                headerHelp=[HEADER_HELP.get(c,c) for c in columns])


@lru_cache(maxsize=1)
def _component(runtime, ui_version):
    # A control release must replace the registered frontend, even in a warm session.
    return st.components.v2.component('sportsbook_edge_table_' + ui_version, html='<section></section>',
        css=Path(__file__).with_suffix('.css').read_text(), js=Path(__file__).with_name('play_storage.mjs').read_text()+'\n'+Path(__file__).with_suffix('.mjs').read_text())


def render_edge_table(edges, columns, roster, effects, team, game, book, day, projections, games=None):
    # Browser components remount on game switches. Reuse their display payload,
    # including peer history, without caching UI calls or anyone's saved picks.
    memo = st.session_state.get('edge_table_payload_cache')
    if memo is None or memo[0] is not roster or memo[1] is not effects:
        memo = (roster, effects, GameCalculationCache())
        st.session_state['edge_table_payload_cache'] = memo
    display_edges = edges.drop(columns='Review captured UTC', errors='ignore')
    payload = memo[2].get((display_edges, columns, team, game, book, day, projections, games),
        pd.Timestamp(0), lambda: table_payload(display_edges, columns, roster, effects, team, game, book, day, projections, games))
    _component(st.runtime.get_instance(), EDGE_TABLE_UI_VERSION)(data=payload,
                                          key='sportsbook_edge_table_' + team)
