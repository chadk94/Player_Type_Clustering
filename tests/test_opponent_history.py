import unittest
from datetime import date

import numpy as np
import pandas as pd

from edge_rating_table import table_payload
from opponent_history import add_opponent_history, history_check, history_index


NOW = pd.Timestamp('2026-11-20T16:00:00Z')
MATCH = pd.Series({'ESPN_PLAYER_ID': '11'})


def prop(**changes):
    return pd.Series(dict(PLAYER_ID=1, Player='Test Player', Team='BOS', Opponent='DET',
                          Stat='PTS', Play='Over', Projected=24.,
                          **{'Book line': 20.5, 'Projected minutes': 32.}) | changes)


def boxes(values=(10, 12, 15, 16, 23), **changes):
    return pd.DataFrame([dict(ESPN_PLAYER_ID='11', GAME_ID=str(i),
                             GAME_DATE=f'2026-11-{19-i:02d}T23:00:00Z', Team='BOS',
                             Opponent='DET', MIN=30., PTS=value, REB=5., AST=3.,
                             FG3M=1., STL=0., BLK=1.) | changes
                         for i, value in enumerate(values)])


def check(row=None, logs=None):
    return history_check(prop() if row is None else row,
                         history_index(boxes() if logs is None else logs), MATCH, NOW)


class OpponentHistoryTests(unittest.TestCase):
    def test_direction_line_average_and_pushes(self):
        result = check()
        self.assertEqual(result['History flag'], 'HISTORY')
        self.assertIn('Opposes Over · 4/5 under 20.5', result['History detail'])
        self.assertIn('11/19/26 · 10 Pts', result['History detail'])
        self.assertEqual(check(prop(Play='Under'))['History flag'], 'SUPPORTS')
        self.assertEqual(check(prop(Play='Under'), boxes([30, 29, 28]))['History flag'], 'HISTORY')
        self.assertEqual(check(logs=boxes([10, 12, 25, 26, 27]))['History flag'], 'No warning')
        # A single enormous opposite result stops a flag if the average agrees.
        self.assertEqual(check(logs=boxes([19, 19, 19, 19, 50]))['History flag'], 'No warning')
        pushed = check(prop(**{'Book line': 20.}), boxes([10, 11, 12, 20]))
        self.assertEqual(pushed['History flag'], 'HISTORY')
        self.assertIn('3/4 under 20', pushed['History detail'])
        self.assertIn('20 Pts · 30 min · P', pushed['History detail'])
        self.assertEqual(check(prop(**{'Book line': 20.}), boxes([10, 10, 20, 20]))['History flag'], 'No warning')

    def test_only_current_team_opponent_and_last_five_before_tip(self):
        good = boxes([30, 30, 30, 30, 30, 0, 0, 0])
        other = boxes([0, 0, 0], Team='NYK').assign(GAME_ID=['a', 'b', 'c'])
        opponent = boxes([0, 0, 0], Opponent='MIA').assign(GAME_ID=['d', 'e', 'f'])
        future = boxes([0], GAME_DATE='2026-11-21T00:00:00Z', GAME_ID='future')
        same_day = boxes([0], GAME_DATE='2026-11-20', GAME_ID='today')
        result = check(logs=pd.concat([good, other, opponent, future, same_day]))
        self.assertEqual(result['History flag'], 'SUPPORTS')
        self.assertEqual(result['History games'], 5)
        self.assertNotIn(' · 0 Pts ·', result['History detail'])
        self.assertEqual(check(logs=boxes([1, 1, 1], MIN=9))['History flag'], 'Insufficient')

    def test_prior_season_allowed_older_season_excluded(self):
        prior = boxes([10, 10, 10]).assign(GAME_DATE=['2026-04-01', '2026-03-01', '2025-11-01'])
        old = boxes([30, 30, 30]).assign(GAME_ID=['old1','old2','old3'],
                                       GAME_DATE=['2025-04-01','2025-03-01','2024-11-01'])
        result = check(logs=pd.concat([prior, old]))
        self.assertEqual(result['History flag'], 'HISTORY')
        self.assertEqual(result['History games'], 3)
        self.assertIn('04/01/26', result['History detail'])

    def test_insufficient_missing_ambiguous_and_invalid_data_never_warn(self):
        self.assertEqual(check(logs=boxes([0, 0]))['History flag'], 'Insufficient')
        self.assertEqual(history_check(prop(), history_index(boxes()), None, NOW)['History flag'], 'Unavailable')
        for missing in [np.nan, np.inf, -1]:
            logs = boxes().astype({'PTS':float});logs.loc[0, 'PTS'] = missing
            self.assertEqual(check(logs=logs)['History flag'], 'Unavailable')
        for bad in [np.nan, np.inf, -1, 'bad']:
            self.assertEqual(check(prop(**{'Book line': bad}))['History flag'], 'Unavailable')
        self.assertEqual(check(prop(Play=''))['History flag'], 'Unavailable')
        duplicate = pd.concat([boxes([0]), boxes([0]), boxes([0])])
        self.assertEqual(check(logs=duplicate)['History games'], 1)
        conflicts = pd.concat([boxes([1, 1, 1]), boxes([2, 2, 2])])
        self.assertEqual(check(logs=conflicts)['History flag'], 'Insufficient')
        # A missing recent record cannot be replaced by a sixth, older good one.
        logs = boxes([0, 0, 0, 0, np.nan, 0])
        self.assertEqual(check(logs=logs)['History flag'], 'Unavailable')

    def test_all_categories_and_raw_makes_for_threes(self):
        for stat, line in [('PTS',20.5),('REB',6.5),('AST',4.5),('3PM',1.5),
                           ('STL',.5),('BLK',1.5),('PRA',35.5)]:
            self.assertEqual(check(prop(Stat=stat, **{'Book line':line}))['History flag'], 'HISTORY', stat)
        bad = boxes();bad.loc[0,'AST'] = np.nan
        self.assertEqual(check(prop(Stat='PRA'), bad)['History flag'], 'Unavailable')

    def test_payload_badge_and_all_model_and_filter_fields_are_unchanged(self):
        edges = pd.DataFrame([prop().to_dict(), prop(Stat='AST').to_dict()])
        edges = edges.assign(**{'Edge %':17., 'Signed edge %':17., 'Review flags':'✓', 'Robust play':True,
                                'Review group':'More robust', 'Exp. P(over)':.6,
                                'Over odds':-110., 'Under odds':-110.})
        original = edges.copy(deep=True)
        reviewed = add_opponent_history(edges, history_index(boxes()), {1:MATCH}, NOW)
        pd.testing.assert_frame_equal(edges, original)
        columns = edges.columns.drop('Review flags')
        pd.testing.assert_frame_equal(reviewed[columns], original[columns])
        self.assertTrue(reviewed['History flag'].eq('HISTORY').all())
        payload = table_payload(reviewed, ['Player','Stat','Play','Edge %','Review flags'],
                                pd.DataFrame([{'PLAYER_ID':1}]), {}, 'BOS',
                                dict(id='demo', Away='BOS', Home='DET'), 'FanDuel', date(2026,11,20))
        badge = payload['rows'][0]['reviewBadges'][0]
        self.assertEqual(badge['label'], 'History −')
        self.assertEqual(badge['color'], 'history')
        self.assertEqual(badge['popupStyle'], 'history')
        self.assertIn('Recent vs today’s line', badge['detail'])
        self.assertIn('not the historical betting lines', badge['technical'])
        sparse = add_opponent_history(edges, {}, {1:MATCH}, NOW)
        sparse_payload = table_payload(sparse, ['Player','Stat','Play','Edge %','Review flags'],
            pd.DataFrame([{'PLAYER_ID':1}]), {}, 'BOS', dict(id='demo',Away='BOS',Home='DET'), 'FanDuel',date(2026,11,20))
        self.assertFalse(any(b['label'].startswith('History') for r in sparse_payload['rows'] for b in r['reviewBadges']))
        self.assertTrue(add_opponent_history(edges.iloc[:0], {}, {}, NOW).empty)

        under = edges.assign(Play='Under', **{'Signed edge %':-17.})
        supported = add_opponent_history(under, history_index(boxes()), {1:MATCH}, NOW)
        pd.testing.assert_frame_equal(supported[columns], under[columns])
        green = table_payload(supported, ['Player','Stat','Play','Edge %','Review flags'],
            pd.DataFrame([{'PLAYER_ID':1}]), {}, 'BOS', dict(id='demo',Away='BOS',Home='DET'), 'FanDuel',date(2026,11,20))
        self.assertTrue(supported['History flag'].eq('SUPPORTS').all())
        for entry in green['rows']:
            badge = entry['reviewBadges'][0]
            self.assertEqual(badge['label'], 'History +')
            self.assertEqual(badge['color'], 'history-support')
            self.assertEqual(badge['popupStyle'], 'history')
            self.assertIn('Supports Under', badge['detail'])

    def test_support_requires_same_minimum_count_fraction_and_average(self):
        self.assertEqual(check(logs=boxes([30,29,28]))['History flag'], 'SUPPORTS')
        self.assertEqual(check(logs=boxes([30,29]))['History flag'], 'Insufficient')
        self.assertEqual(check(logs=boxes([30,29,28,10,10]))['History flag'], 'No warning')
        self.assertEqual(check(logs=boxes([21,21,21,21,0]))['History flag'], 'No warning')
        tied = prop(**{'Book line':20.})
        self.assertEqual(check(tied, boxes([30,30,30,20]))['History flag'], 'SUPPORTS')
        self.assertEqual(check(tied, boxes([30,30,30,20,20]))['History flag'], 'No warning')
        self.assertEqual(check(tied, boxes([20,20,20]))['History flag'], 'No warning')
        bad = boxes([30,30,30]);bad.loc[0,'PTS'] = np.nan
        self.assertEqual(check(logs=bad)['History flag'], 'Unavailable')


if __name__ == '__main__':
    unittest.main()
