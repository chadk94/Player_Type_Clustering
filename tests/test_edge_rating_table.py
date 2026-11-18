import unittest
from datetime import date

import pandas as pd

from edge_rating_table import group_player_edges, rating_detail, table_payload
from todays_plays import edge_pick
from matchup_projections import matchup_effects


class EdgeRatingDetailsTests(unittest.TestCase):
    def test_player_groups_use_best_qualifying_edge_and_preserve_prop_terms(self):
        edges = pd.DataFrame([
            dict(PLAYER_ID=1, Player='Cameron Boozer', Stat='AST', **{'Edge %': 21.07}),
            dict(PLAYER_ID=2, Player='Cedric Coward', Stat='AST', **{'Edge %': 17.53}),
            dict(PLAYER_ID=1, Player='Cameron Boozer', Stat='PRA', **{'Edge %': 15.10}),
            dict(PLAYER_ID=1, Player='Cameron Boozer', Stat='REB', **{'Edge %': 9.89}),
            dict(PLAYER_ID=2, Player='Cedric Coward', Stat='PTS', **{'Edge %': 8.08}),
        ]).assign(Team='MEM', Rank=range(1, 6))
        original = edges.copy(deep=True)
        grouped = group_player_edges(edges)
        self.assertEqual(list(zip(grouped.PLAYER_ID, grouped.Stat, grouped.Rank)),
                         [(1,'AST',1), (1,'PRA',1), (1,'REB',1), (2,'AST',2), (2,'PTS',2)])
        # Filtering out the strongest prop changes the player order, not any terms.
        filtered = group_player_edges(edges.loc[edges.Stat.ne('AST') | edges.PLAYER_ID.ne(1)])
        self.assertEqual(filtered.PLAYER_ID.tolist(), [2, 2, 1, 1])
        self.assertEqual(filtered.Rank.tolist(), [1, 1, 2, 2])
        pd.testing.assert_frame_equal(edges, original)
        pd.testing.assert_frame_equal(
            grouped.drop(columns='Rank').sort_values(['PLAYER_ID', 'Stat']).reset_index(drop=True),
            original.drop(columns='Rank').sort_values(['PLAYER_ID', 'Stat']).reset_index(drop=True))

    def test_grouped_payload_keeps_save_identity_and_separate_player_ids(self):
        base = dict(Player='Same Name', Team='MEM', Opponent='CLE', Play='Over', Projected=5.5,
                    **{'Book line': 4.5, 'Over odds': -110, 'Under odds': -110,
                       'Signed edge %': 10., 'Edge %': 10.})
        edges = pd.DataFrame([dict(base, PLAYER_ID=1, Stat='AST'),
                              dict(base, PLAYER_ID=2, Stat='PTS'),
                              dict(base, PLAYER_ID=1, Stat='REB')])
        roster = pd.DataFrame([dict(PLAYER_ID=1), dict(PLAYER_ID=2)])
        event = dict(id='demo', Away='MEM', Home='CLE')
        day = date(2026, 11, 18)
        columns = ['Rank', 'Player', 'Stat', 'Edge %', 'Projected', 'Play']
        payload = table_payload(edges, columns, roster, {}, 'MEM', event, 'FanDuel', day)
        self.assertEqual([row['playerRowSpan'] for row in payload['rows']], [2, 0, 1])
        self.assertEqual([row['labels'][2] for row in payload['rows']], ['AST', 'REB', 'PTS'])
        self.assertEqual([row['labels'][0] for row in payload['rows']], ['1', '1', '2'])
        expected = {edge_pick(row, event, 'FanDuel', day)['id']: edge_pick(row, event, 'FanDuel', day)
                    for _, row in edges.iterrows()}
        self.assertEqual({row['pick']['id']: row['pick'] for row in payload['rows']}, expected)
        self.assertTrue(group_player_edges(edges.iloc[:0]).empty)

    def test_hover_card_uses_full_final_projection_even_when_only_one_prop_qualifies(self):
        edges = pd.DataFrame([dict(PLAYER_ID=1, Player='Demo Player', Team='MEM', Opponent='CLE',
            Stat='AST', Play='Over', Projected=8.12,
            **{'Book line': 6.5, 'Over odds': -110, 'Under odds': -110,
               'Signed edge %': 10., 'Edge %': 10.})])
        projections = pd.DataFrame([
            dict(PLAYER_ID=1, Player='Demo Player', Team='MEM', Opponent='CLE',
                 PTS=16.25, REB=5.31, AST=8.12, FG3M=2.13, STL=0., BLK=.43,
                 TOV=2.4, PF=float('nan'), PRA=29.68,
                 **{'Projected minutes':31.5, 'Baseline PTS':99.}),
            dict(PLAYER_ID=2, Player='Demo Player', Team='MEM', Opponent='CLE', PTS=50.),
            dict(PLAYER_ID=1, Player='Demo Player', Team='MEM', Opponent='BOS', PTS=60.),
        ])
        original = projections.copy(deep=True)
        args = (edges, ['Rank','Player','Stat','Edge %','Projected','Play'],
                pd.DataFrame([dict(PLAYER_ID=1)]), {}, 'MEM',
                dict(id='demo', Away='MEM', Home='CLE'), 'FanDuel', date(2026,11,18))
        card = table_payload(*args, projections=projections)['rows'][0]['projection']
        self.assertEqual(card['minutes'], '31.5')
        self.assertEqual(card['opponent'], 'CLE')
        self.assertEqual({s['stat']: s['value'] for s in card['stats']},
                         dict(PTS='16.25', REB='5.31', AST='8.12', **{'3PM':'2.13'},
                              STL='0.00', BLK='0.43', TOV='2.40', PF='—', PRA='29.68'))
        pd.testing.assert_frame_equal(projections, original)
        wrong_game = projections.loc[projections.Opponent.eq('BOS')]
        self.assertIsNone(table_payload(*args, projections=wrong_game)['rows'][0]['projection'])
        ambiguous = pd.concat([projections, projections.iloc[:1]], ignore_index=True)
        self.assertIsNone(table_payload(*args, projections=ambiguous)['rows'][0]['projection'])

    def test_slate_table_saves_each_event_and_shows_its_own_projection(self):
        base = dict(PLAYER_ID=1, Player='Same Player', Team='MEM', Opponent='CLE', Stat='PTS',
                    Play='Over', Projected=20., **{'Book line': 15.5, 'Over odds': -110,
                    'Under odds': -110, 'Signed edge %': 20., 'Edge %': 20.})
        # Even the same player/team/opponent must not merge across event IDs.
        edges = pd.DataFrame([dict(base, **{'Event ID':'first'}),
                              dict(base, Projected=25., **{'Event ID':'second', 'Edge %':30.})])
        games = {key: dict(id=key, Away='MEM', Home='CLE') for key in ['first','second']}
        projections = edges.rename(columns={'Projected':'PTS'}).assign(**{'Projected minutes':30.})
        day = date(2026,11,18)
        result = table_payload(edges, ['Rank','Player','Stat','Projected','Edge %','Play'],
            pd.DataFrame([dict(PLAYER_ID=1)]), {}, 'All games', None, 'FanDuel', day,
            projections, games)
        self.assertEqual([row['playerRowSpan'] for row in result['rows']], [1,1])
        self.assertEqual([row['projection']['stats'][0]['value'] for row in result['rows']], ['25.00','20.00'])
        expected = {edge_pick(row, games[row['Event ID']], 'FanDuel', day)['id'] for _, row in edges.iterrows()}
        self.assertEqual({row['pick']['id'] for row in result['rows']}, expected)
        self.assertEqual(len(expected), 2)

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
