import unittest

import numpy as np
import pandas as pd

from slate_player_view import rank_slate_players
from player_navigation import selected_player_request


class SlatePlayerViewTests(unittest.TestCase):
    def setUp(self):
        self.now = pd.Timestamp('2026-10-28T18:00:00Z')
        self.games = {key: dict(key=key, Home=home, Away=away, GameTimeUTC=self.now+pd.Timedelta(hours=6))
                      for key,home,away in [('hou-mil','MIL','HOU'),('bos-det','DET','BOS')]}
        self.rows = pd.DataFrame([
            dict(PLAYER_ID=1, Player='A', Team='HOU', PTS=20.001, REB=3., **{'Projected minutes':30.,'Game key':'hou-mil'}),
            dict(PLAYER_ID=2, Player='B', Team='BOS', PTS=20.004, REB=9., **{'Projected minutes':25.,'Game key':'bos-det'}),
            dict(PLAYER_ID=3, Player='C', Team='MIL', PTS=np.nan, REB=5., **{'Projected minutes':np.nan,'Game key':'hou-mil'}),
            dict(PLAYER_ID=4, Player='D', Team='DET', PTS=0., REB=0., **{'Projected minutes':0.,'Game key':'bos-det'}),
        ], index=[42,56,67,98])
        self.rows['GameTimeUTC'] = self.now + pd.Timedelta(hours=6)

    def test_ranks_full_precision_keeps_unknowns_and_optionally_zero_minutes(self):
        original = self.rows.copy(deep=True)
        ranked = rank_slate_players(self.rows, 'PTS', now=self.now)
        self.assertEqual(ranked.Player.tolist(), ['B','A','C'])
        self.assertTrue(pd.isna(ranked.iloc[-1].PTS))
        self.assertEqual(ranked.Rank.tolist(), [1,2,3])
        self.assertEqual(rank_slate_players(self.rows,'REB',True,now=self.now).Player.tolist(),['B','C','A','D'])
        pd.testing.assert_frame_equal(self.rows,original)

    def test_uses_clicked_rows_game_after_sorting_and_retains_opponent(self):
        ranked = rank_slate_players(self.rows,'REB',now=self.now)
        request = selected_player_request({'cells':[(0,'Player')]},ranked,self.games,'FanDuel')
        self.assertEqual((request['player_id'],request['team'],request['opponent']),(2,'BOS','DET'))
        self.assertEqual(request['start'],self.games['bos-det']['GameTimeUTC'].isoformat())
        self.assertIsNone(selected_player_request({'rows':[0]},ranked,{},'FanDuel'))

    def test_retention_window_applies_on_sort_and_unavailable_category_stays_blank(self):
        self.rows.loc[self.rows.Team.eq('HOU'),'GameTimeUTC'] = self.now-pd.Timedelta(hours=1)
        ranked = rank_slate_players(self.rows,'PF',now=self.now)
        self.assertNotIn('A',ranked.Player.tolist())
        self.assertTrue(ranked.PF.isna().all())
