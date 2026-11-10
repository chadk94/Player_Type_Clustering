"""Sortable sportsbook edges with opponent-specific evidence on each rating."""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import pandas as pd
import streamlit as st

from matchup_projections import RECENT_OPPONENT_GAMES
from points_probability import PROBABILITY_COLUMNS
from sportsbook_lines import comparison_style
from todays_plays import edge_pick


# Match the component statistics used by apply_matchup, including the attempts
# effect on made threes. Keep the two rebound populations separate.
ROLE_STATS = {
    'PTS': {'Off': ('PTS',)}, 'AST': {'Off': ('AST',)},
    '3PM': {'Off': ('FG3A',)}, 'STL': {'Def': ('STL',)}, 'BLK': {'Def': ('BLK',)},
    'REB': {'Off': ('OREB',), 'Def': ('DREB',)},
    'PRA': {'Off': ('PTS', 'AST', 'OREB'), 'Def': ('DREB',)},
}
FORMATS = {'Projected minutes': '.1f', 'Good-to minutes': 'g', 'Projected': '.2f', 'Recent role': '.2f',
           'Recent change %': '+.1f', 'Book implied total': '.2f', 'Edge %': '+.2f',
           'Over odds': '+.0f', 'Under odds': '+.0f', **{key: '.1%' for key in PROBABILITY_COLUMNS}}


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


def table_payload(edges, columns, roster, effects, team, game, book, day):
    players = roster.set_index('PLAYER_ID')
    output = []
    for _, row in edges.iterrows():
        raw = row[columns]
        values = [None if pd.isna(v) else v for v in raw.tolist()]
        labels = []
        for col, v in zip(columns, values):
            if v is None:
                labels.append('—')
            elif col == 'Minutes cushion':
                labels.append(f'{v:.2f}'.rstrip('0').rstrip('.'))
            elif col in FORMATS:
                labels.append(format(v, FORMATS[col]) + ('%' if col in ('Edge %', 'Recent change %') else ''))
            else:
                labels.append(str(v))
        player = players.loc[row.PLAYER_ID] if row.PLAYER_ID in players.index else {}
        output.append(dict(pick=edge_pick(row, game, book, day), values=values, labels=labels, styles=comparison_style(raw),
                           detail=rating_detail(row.Stat, player, row.Opponent, effects)))
    return dict(columns=columns, rows=output, team=team)


@lru_cache(maxsize=1)
def _component(runtime):
    return st.components.v2.component('sportsbook_edge_table', html='<section></section>',
        css=Path(__file__).with_suffix('.css').read_text(), js=Path(__file__).with_name('play_storage.mjs').read_text()+'\n'+Path(__file__).with_suffix('.mjs').read_text())


def render_edge_table(edges, columns, roster, effects, team, game, book, day):
    _component(st.runtime.get_instance())(data=table_payload(edges, columns, roster, effects, team, game, book, day),
                                          key='sportsbook_edge_table_' + team)
