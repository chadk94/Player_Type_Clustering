import unittest

import numpy as np
import pandas as pd

from edge_minutes import _minutes_projections, add_minutes_cushion
from market_points import apply_market_points
from team_assists import apply_team_assists
from team_rebounds import apply_team_rebounds
from team_threes import apply_team_threes


class EdgeMinutesTests(unittest.TestCase):
    def test_strict_rounded_cutoff_in_both_directions_and_nonplays(self):
        # At 21 and 19 minutes respectively the displayed edge is exactly 5%.
        team = pd.DataFrame({'PLAYER_ID': [1, 2], 'Team': 'DET',
                             'Projected minutes': [25., 15.], 'STL': [2.5, 1.5]})
        rows = pd.DataFrame({'PLAYER_ID': [1, 2, 1], 'Team': 'DET', 'Stat': 'STL',
                             'Play': ['Over', 'Under', ''], 'Projected': [2.5, 1.5, 2.5],
                             'Book implied total': [1.99999, 2., 2.5]})
        out = add_minutes_cushion(rows, team)
        pd.testing.assert_frame_equal(out[rows.columns], rows)
        np.testing.assert_equal(out['Good-to minutes'].to_numpy(), [21.5, 18.5, np.nan])
        np.testing.assert_equal(out['Minutes cushion'].to_numpy(), [3.5, 3.5, np.nan])

    def test_fractional_approved_minutes_zero_production_and_regulation_cap(self):
        team = pd.DataFrame({'PLAYER_ID': [1, 2], 'Team': 'DET',
                             'Projected minutes': [30.25, 20.], 'BLK': [1.06, 0.]})
        rows = pd.DataFrame({'PLAYER_ID': [1, 2], 'Team': 'DET', 'Stat': 'BLK',
                             'Play': ['Over', 'Under'], 'Projected': [1.06, 0.],
                             'Book implied total': [1.008, 1.]})
        out = add_minutes_cushion(rows, team)
        self.assertEqual(out['Good-to minutes'].tolist(), [30.25, 48.])
        self.assertEqual(out['Minutes cushion'].tolist(), [0., 28.])

    def test_scenarios_match_full_roster_reruns_including_threes_guard(self):
        rates = pd.DataFrame({
            'Assist rate': .60, 'Allowed assist rate': .60, 'Points per FGM': 2.5,
            'Allowed points per FGM': 2.5, 'Three attempt rate': .40, 'Allowed three attempt rate': .40,
            'Points per FGA': 1., 'Allowed points per FGA': 1., 'FG misses per FGM': 1.,
            'FT misses per FGM': .1, 'Player credited fraction': .8, 'Offensive rebound share': .25,
        }, index=['BOS', 'DET'])
        targets, opp = {'BOS': 100., 'DET': 110.}, {'BOS': 'DET', 'DET': 'BOS'}
        frame = pd.DataFrame({'PLAYER_ID': [1, 2], 'Team': 'BOS', 'Opponent': 'DET',
            'Projected minutes': [30., 30.], 'PTS': [25., 75.], 'OREB': [2., 3.],
            'DREB': [6., 4.], 'AST': [2., 8.], 'FG3A': [10., 10.], 'FG3M': [4., 3.],
            'STL': [1., 2.], 'BLK': [.5, 1.]})
        frame['REB'] = frame.OREB + frame.DREB
        frame['PRA'] = frame.PTS + frame.REB + frame.AST
        stats = ['PTS', 'AST', 'REB', 'OREB', 'DREB', 'FG3M', 'FG3A', 'STL', 'BLK', 'PRA']

        def recalibrate(minutes):
            raw = frame.copy()
            raw.loc[0, stats] *= minutes / 30.
            raw.loc[0, 'Projected minutes'] = minutes
            points, _, _ = apply_market_points(raw, targets)
            assists, _, _ = apply_team_assists(points, targets, rates, opp)
            threes, _, _ = apply_team_threes(assists, targets, rates, opp)
            final, _, _ = apply_team_rebounds(threes, targets, rates, rates, opp)
            return final

        minutes = np.array([.5, 12., 30., 48.])
        snapshots = [recalibrate(m) for m in minutes]
        self.assertTrue(pd.isna(snapshots[0].iloc[0]['Team 3PA target']))
        self.assertEqual(snapshots[-1].iloc[0]['Team 3PA target'], 40.)
        for start in [0, 2]:
            forecast = _minutes_projections(snapshots[start], 0, minutes)
            for stat, estimates in forecast.items():
                field = 'FG3M' if stat == '3PM' else stat
                np.testing.assert_allclose(estimates, [s.iloc[0][field] for s in snapshots])
