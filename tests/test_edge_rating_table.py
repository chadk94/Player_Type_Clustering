import unittest

import pandas as pd

from edge_rating_table import rating_detail
from matchup_projections import matchup_effects


class EdgeRatingDetailsTests(unittest.TestCase):
    def test_window_uses_distinct_opponent_games_and_rates_use_qualifying_player_games(self):
        rows = [dict(GAME_DATE=f'2026-11-0{i}', GAME_ID=str(i), ESPN_PLAYER_ID='1',
                     MIN=36, PTS=99 if i==1 else 10, FG3M=0, FG3A=4,
                     OffCluster=1, DefCluster=2, Opponent='BOS') for i in range(1,7)]
        rows.extend([dict(rows[-1], ESPN_PLAYER_ID='2', MIN=18, PTS=10),
                     dict(rows[-1], ESPN_PLAYER_ID='3', MIN=4, PTS=20),
                     dict(rows[-1], Opponent='NYK', PTS=99)])
        effects = matchup_effects(pd.DataFrame(rows), include_recent=True)
        recent = effects[('Off',1,'BOS')]['recent']
        self.assertEqual(recent['games'], 5)
        self.assertEqual(recent['stats']['PTS'][1],6)
        self.assertAlmostEqual(recent['stats']['PTS'][0],70/6)
        self.assertEqual(recent['stats']['FG3M'],(0,6))
        self.assertNotIn('AST',recent['stats'])

    def test_tooltip_uses_the_actual_prop_components_and_reports_missing_current_games(self):
        player = dict(OffCluster=1,OffArchetype='Creator',DefCluster=2,DefArchetype='Wing')
        effects = {('Off',1,'BOS'):dict(recent=dict(games=3,stats={
            'PTS':(25,6), 'AST':(5,6), 'OREB':(1,6), 'FG3A':(6,4), 'FG3M':(2,4)}))}
        text = rating_detail('PRA', player, 'BOS', effects)
        self.assertIn('last 3 available games',text)
        self.assertIn('Offense · Creator\n25.0 PTS · 5.0 AST · 1.0 OREB (6 player appearances)',text)
        self.assertIn('Defense · Wing\nNo qualifying results',text)
        threes = rating_detail('3PM',player,'BOS',effects)
        self.assertIn('6.0 3PA (4 player appearances)',threes)
        self.assertNotIn('3PM',threes)
        self.assertNotIn('Offense',rating_detail('STL',player,'BOS',effects))
        self.assertNotIn('25.0',rating_detail('PTS',player,'NYK',effects))
