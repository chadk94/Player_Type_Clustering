"""Small game-scoped source marker; never fetches or recalculates projections."""
from functools import lru_cache
from pathlib import Path

import pandas as pd
import streamlit as st


def minutes_metadata(refs):
    if refs is None or refs.empty:
        return []
    rows = []
    for (team, opponent, day), group in refs.groupby(['Team', 'Opponent', 'game_date']):
        def timestamp(column):
            values = pd.to_datetime(group[column], utc=True, errors='coerce')
            return values.min().isoformat() if values.notna().all() else None
        rows.append(dict(team=team, opponent=opponent, day=str(day),
            published=timestamp('source_updated_at'), captured=timestamp('updated_at'),
            saved=bool(group.get('source_saved', pd.Series(False, index=group.index)).any()),
            complete=bool(team not in refs.attrs.get('incomplete_teams', [])
                          and group.minutes.gt(0).sum() >= 5 and abs(group.minutes.sum()-240) <= 2)))
    return rows


@lru_cache(maxsize=1)
def _status_component(runtime):
    return st.components.v2.component('tabs_minutes_status',
        html='''<section aria-label="Tabs minutes update"><div>
        <strong>Tabs minutes last updated:</strong> <span data-summary></span>
        <button type="button">Refresh Tabs minutes</button>
        <a href="https://app.tabsnba.com/projections" target="_blank" rel="noopener noreferrer">Open TabsNBA ↗</a>
        </div><span data-feedback role="status" aria-live="polite"></span>
        <details><summary>By team</summary><div data-teams></div></details></section>''',
        css='''section{font-size:1rem;line-height:1.6;margin:.4rem 0 .7rem}
        button,a{font:inherit;color:var(--st-link-color,#1676c4);text-decoration:underline;margin-left:1rem}
        button{padding:0;border:0;background:none;cursor:pointer}button:disabled{opacity:.6;cursor:wait}
        [data-feedback]{display:block;font-size:.9rem;opacity:.85}details{font-size:.9rem}summary{cursor:pointer}
        [data-teams]{display:flex;gap:.2rem 1.4rem;flex-wrap:wrap}''',
        js=Path(__file__).with_suffix('.mjs').read_text())


def render_tabs_minutes_status(games, key):
    scopes = []
    for game in games:
        day = str(pd.Timestamp(game['GameTimeUTC']).tz_convert('America/New_York').date())
        scopes.extend(dict(team=team, opponent=opponent, day=day)
                      for team, opponent in ((game['Away'], game['Home']), (game['Home'], game['Away'])))
    _status_component(st.runtime.get_instance())(key=key, data=dict(
        scopes=scopes, rows=minutes_metadata(st.session_state.get('tabs_minutes_reference')),
        browser=st.session_state.get('tabs_minutes_mode', 'Browser connection') == 'Browser connection'))
