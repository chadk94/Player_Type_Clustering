"""Edge-ranked tables with one-click confidence controls and matchup evidence."""
from collections import defaultdict
from functools import lru_cache
from math import isfinite
from pathlib import Path

import pandas as pd
import streamlit as st

from matchup_projections import RECENT_OPPONENT_GAMES
from points_probability import PROBABILITY_COLUMNS
from sportsbook_lines import comparison_style, PLAY_THRESHOLD_PCT
from todays_plays import edge_pick

EDGE_TABLE_UI_VERSION = 'player_photos_v1'

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
    'Rank':'Player rank within this team, based on their highest qualifying Edge %. All their displayed props stay together.',
    'Edge %':'Players are ordered by their highest qualifying edge; props within each player are highest first. This projection gap is not win probability or expected return.',
    'Good-to minutes':'Minutes at which this side still clears 5% edge: lower limit for an Over, upper limit for an Under. Tested in half-minute steps through 48.',
    'Matchup rating':'1 = poor, 3 = average, 5 = excellent. Click a rating for archetype matchup evidence.',
    'Review flags':'Minutes unverified = no completed minutes check; Minutes risk = playing time could weaken the bet; ROLE = recent-role risk; Q = availability risk; ? = unchecked. Click a badge for details.',
    'Book line (Over)':'Sportsbook line followed by the Over price. Under prices appear in the next column.',
    'Exp. P(over)':'Experimental probability of going Over, conditional on playing. Whole-number lines can push.',
    'Exp. P(under)':'Experimental probability of going Under, conditional on playing. Whole-number lines can push.',
}


def minutes_popup(row):
    """Describe the existing stress test; never imply its minutes came from a source."""
    def finite(value):
        return isinstance(value, (int, float)) and isfinite(value)

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
    return pd.concat([rows.assign(Rank=rank) for rank, (_, rows) in enumerate(
        ordered.groupby(['Team', 'PLAYER_ID'], sort=False, dropna=False), start=1)], ignore_index=True)


def table_payload(edges, columns, roster, effects, team, game, book, day):
    players = roster.set_index('PLAYER_ID')
    output = []
    grouped = group_player_edges(edges)
    counts = grouped['Rank'].value_counts()
    last_rank = None
    for _, row in grouped.iterrows():
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
        player = players.loc[row.PLAYER_ID] if row.PLAYER_ID in players.index else {}
        badges=[]
        for flag,title,color in [('Minutes','MIN','minutes'),('Role','ROLE','role'),('Injury','Q','injury')]:
            status=row.get(flag+' flag')
            if status in ('MIN','ROLE','Q','Unchecked'):
                label = ('Minutes unverified' if status == 'Unchecked' else 'Minutes risk') if flag == 'Minutes' else (title if status != 'Unchecked' else title+'?')
                popup = minutes_popup(row) if flag == 'Minutes' else role_popup(row) if flag == 'Role' else dict(detail=row.get(flag+' detail','Unchecked'))
                badges.append(dict(label=label,
                    color=color if status!='Unchecked' else 'unchecked',
                    **popup))
        if not badges and 'Review group' in row:
            badges.append(dict(label='✓' if row.get('Robust play') else 'Review',color='clear' if row.get('Robust play') else 'unchecked',detail=row.get('Review detail','')))
        output.append(dict(pick=edge_pick(row, game, book, day), values=values, labels=labels, styles=comparison_style(raw),
                           playerId=int(row.PLAYER_ID),
                           playerRowSpan=int(counts[row.Rank]) if row.Rank != last_rank else 0,
                           detail=rating_detail(row.Stat, player, row.Opponent, effects),reviewBadges=badges))
        last_rank = row.Rank
    return dict(columns=columns, rows=output, team=team,
                headers=[HEADER_LABELS.get(c,c) for c in columns],
                headerHelp=[HEADER_HELP.get(c,c) for c in columns])


@lru_cache(maxsize=1)
def _component(runtime, ui_version):
    # A control release must replace the registered frontend, even in a warm session.
    return st.components.v2.component('sportsbook_edge_table_' + ui_version, html='<section></section>',
        css=Path(__file__).with_suffix('.css').read_text(), js=Path(__file__).with_name('play_storage.mjs').read_text()+'\n'+Path(__file__).with_suffix('.mjs').read_text())


def render_edge_table(edges, columns, roster, effects, team, game, book, day):
    _component(st.runtime.get_instance(), EDGE_TABLE_UI_VERSION)(data=table_payload(edges, columns, roster, effects, team, game, book, day),
                                          key='sportsbook_edge_table_' + team)
