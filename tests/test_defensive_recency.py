from copy import deepcopy
from datetime import timedelta
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from team_rate_blend import season_rates, blend_team_rates, DEFENSIVE_RATE_COLUMNS
from market_points import apply_market_points
from team_assists import apply_team_assists
from team_threes import apply_team_threes
from team_rebounds import apply_team_rebounds
import defensive_recency as trial
import defensive_recency_view as view
from tests.test_team_rate_blend import logs as box_logs


def fixture(n=15):
    data = box_logs([('A', 'B')] * n)
    data['ESPN_PLAYER_ID'] = data.Team.map({'A': '1', 'B': '2'})
    data['GAME_DATE'] = data.GAME_ID.map(lambda g: (pd.Timestamp('2026-10-01', tz='UTC') + pd.Timedelta(days=int(g))).isoformat())
    # A defense reduces opponent threes, assists and offensive rebounds recently.
    changed = data.Team.eq('B') & data.GAME_ID.astype(int).ge(8)
    data.loc[changed, ['FG3A', 'AST', 'OREB', 'REB']] = [25., 20., 8., 38.]
    prior = season_rates(box_logs([('A', 'B')] * 10))
    teams = pd.DataFrame({'TEAM_ABBREVIATION': ['A', 'B'], 'GAMES_PLAYED': [n, n]})
    current = blend_team_rates(prior, data, teams)
    rates = trial.comparison_rates(prior, current, data)
    base = pd.DataFrame([dict(PLAYER_ID=i, Player=f'Player {i}', Team=t, Opponent=o,
        **{'Projected minutes': 40.}, PTS=20., AST=4., OREB=1., DREB=5., REB=6.,
        PRA=30., FG3A=5., FG3M=2., STL=1., BLK=.5, TOV=2.)
        for i, (t, o) in enumerate([('A', 'B')] * 6 + [('B', 'A')] * 6)])
    opponents, targets = {'A': 'B', 'B': 'A'}, {'A': 110., 'B': 112.}
    final, _, errors = apply_market_points(base, targets)
    final, _, errors = apply_team_assists(final, targets, current['assists'], opponents)
    assert not errors
    final, _, errors = apply_team_threes(final, targets, current['threes'], opponents)
    assert not errors
    final, _, errors = apply_team_rebounds(final, targets, current['rebounds_own'], current['rebounds_allowed'], opponents)
    assert not errors
    return data, prior, current, rates, final


class DefensiveRecencyTests(unittest.TestCase):
    def test_only_three_allowed_rates_change_with_same_year_weight(self):
        logs, prior, current, rates, _ = fixture()
        untouched = deepcopy(current)
        n = 15
        weights = .5 ** (np.arange(n-1, -1, -1) / 10)
        allowed = logs[logs.Team.eq('B')]
        share = np.dot(weights, allowed.FG3A) / np.dot(weights, allowed.FGA)
        w = current['coverage'].loc['A', 'Defensive current weight']
        self.assertAlmostEqual(rates['threes'].loc['A', 'Allowed three attempt rate'],
                               w * share + (1-w) * prior['threes'].loc['A', 'Allowed three attempt rate'])
        for kind in prior:
            for col in current[kind]:
                if col not in DEFENSIVE_RATE_COLUMNS.get(kind, ()):
                    pd.testing.assert_series_equal(current[kind][col], rates[kind][col])
            pd.testing.assert_frame_equal(current[kind], untouched[kind])
        pd.testing.assert_frame_equal(current['coverage'], rates['coverage'].drop(columns='Recency status'))

    def test_games_not_appearances_and_input_order_does_not_change_weights(self):
        logs, prior, current, rates, _ = fixture()
        split = logs.copy()
        split[split.select_dtypes('number').columns] /= 2
        second = split.copy()
        second.ESPN_PLAYER_ID += '0'
        split = pd.concat([split, second], ignore_index=True).sample(frac=1, random_state=7)
        other = trial.comparison_rates(prior, current, split)
        for kind in rates:
            pd.testing.assert_frame_equal(rates[kind], other[kind])

    def test_eight_game_gate_and_missing_history(self):
        for n in [7, 8, 20]:
            logs, prior, current, rates, _ = fixture(n)
            self.assertTrue(rates['coverage']['Recency status'].eq('Ready' if n >= 8 else 'Waiting for eight verified games').all())
            broken = trial.comparison_rates(prior, current, logs[logs.GAME_ID.ne('0')])
            self.assertTrue(broken['coverage']['Recency status'].str.contains('incomplete').all())
            for kind in prior:
                pd.testing.assert_frame_equal(broken[kind], current[kind])
            if n == 20:
                prior['threes']['Allowed three attempt rate'] = np.nan
                without_prior = trial.comparison_rates(prior, current, logs)
                self.assertTrue(np.isfinite(without_prior['threes']['Allowed three attempt rate']).all())

    def test_malformed_dates_duplicates_and_boxes_never_produce_trial(self):
        logs, prior, current, _, _ = fixture()
        for bad in [logs.assign(GAME_DATE='bad'), pd.concat([logs, logs.iloc[:1]]), logs.assign(FG3A=1000.)]:
            result = trial.comparison_rates(prior, current, bad)
            self.assertTrue(result['coverage']['Recency status'].str.contains('Unavailable').all())

    def test_comparison_preserves_live_frame_and_player_shares(self):
        _, _, _, rates, final = fixture()
        before = final.copy(deep=True)
        totals, players = trial.compare_game(final, rates, {'A': 'B', 'B': 'A'})
        pd.testing.assert_frame_equal(final, before, check_exact=True)
        self.assertEqual(len(totals), 14)
        for team in ['A', 'B']:
            rows = players[players.Team.eq(team)]
            for stat in ['AST', '3PM', 'REB']:
                subset = rows[rows.Stat.eq(stat)]
                self.assertTrue(np.allclose(subset.Current/subset.Current.sum(),
                                           subset['Defensive recency']/subset['Defensive recency'].sum()))
            p = rows.pivot(index='Player', columns='Stat', values='Defensive recency')
            pts = final[final.Team.eq(team)].set_index('Player').PTS
            self.assertTrue(np.allclose(p.PRA, p.AST + p.REB + pts.loc[p.index]))
        t = totals.set_index(['Team', 'Stat'])
        for team in ['A', 'B']:
            self.assertAlmostEqual(t.loc[(team, '3PM'), 'Current']/t.loc[(team, '3PA'), 'Current'],
                                   t.loc[(team, '3PM'), 'Defensive recency']/t.loc[(team, '3PA'), 'Defensive recency'])
        self.assertLess(t.loc[('B', '3PA'), 'Change'], 0)

    def test_incomplete_projections_cannot_be_a_successful_trial(self):
        _, _, _, rates, final = fixture()
        for col in ['Projected minutes', *trial.TARGETS, 'FG3A']:
            bad = final.copy()
            bad.loc[0, col] = np.nan
            with self.assertRaises(ValueError):
                trial.compare_game(bad, rates, {'A': 'B', 'B': 'A'})

    def make_record(self):
        logs, _, _, rates, final = fixture()
        totals, _ = trial.compare_game(final, rates, {'A': 'B', 'B': 'A'})
        game = dict(Home='A', Away='B', GameTimeUTC='2026-11-23T02:00:00Z', key='game')
        record = trial.snapshot(totals, game, 'FanDuel', logs, final, now='2026-11-22T23:00:00Z')
        return record, logs, final, totals, game

    def test_snapshot_freezes_inputs_and_rejects_post_tip(self):
        record, logs, final, totals, game = self.make_record()
        self.assertTrue(trial.valid_record(record))
        same = trial.snapshot(totals, game, 'FanDuel', logs, final, now='2026-11-22T23:30:00Z')
        self.assertEqual(record['id'], same['id'])
        self.assertIsNone(trial.snapshot(totals, game, 'FanDuel', logs, final, now=game['GameTimeUTC']))
        self.assertIsNone(trial.snapshot(totals, game, 'FanDuel', logs.assign(GAME_DATE='2026-11-23T00:00:00Z'), final, now='2026-11-22T23:00:00Z'))
        tampered = deepcopy(record)
        tampered['totals'][0]['Current'] = 999
        self.assertFalse(trial.valid_record(tampered))

    def test_latest_snapshot_and_documented_change_survive_later_untagged_view(self):
        record, logs, final, totals, game = self.make_record()
        tag = trial.snapshot(totals, game, 'FanDuel', logs, final,
            {'A': {'change': 'New starter Nov 22', 'source': 'https://example.com/team-report'}}, now='2026-11-22T23:05:00Z')
        newer = deepcopy(record)
        newer['captured_at'] = '2026-11-22T23:15:00Z'
        newer['totals'][0]['Current'] += .1
        newer['id'] = trial.record_id(newer)
        results = logs[logs.GAME_ID.eq('14')].assign(GAME_DATE=game['GameTimeUTC'])
        history = trial.evaluate([record, tag, newer], results)
        self.assertEqual(len(history), 14)
        self.assertTrue(history.Actual.notna().all())
        self.assertTrue(history['Captured at'].eq(newer['captured_at']).all())
        self.assertTrue(history.loc[history.Team.eq('B'), 'Documented opponent change'].ne('').all())
        self.assertTrue(history.loc[history.Team.eq('A'), 'Documented opponent change'].eq('').all())
        self.assertTrue(trial.error_summary(history, documented_only=True)['Team-games'].eq(1).all())
        for broken in [results.iloc[:1], results.assign(MIN=100), pd.concat([results, results])]:
            self.assertTrue(trial.evaluate([record], broken).Actual.isna().all())

    def test_browser_restore_incremental_ack_and_failed_storage(self):
        record, *_ = self.make_record()
        state = {}
        view.accept_reply(state, 'req', dict(request='req', status='ok', records=[json.dumps(record)], full=True))
        self.assertEqual(list(state['records']), [record['id']])
        view.accept_reply(state, 'next', dict(request='next', base='req', status='ok', records=[], full=False))
        self.assertEqual(len(state['records']), 1)
        view.accept_reply(state, 'failure', dict(request='failure', status='unavailable'))
        self.assertEqual(state['status'], 'unavailable')
        self.assertEqual(len(state['records']), 1)

    def test_ui_comparison_is_lazy_and_history_ack_does_not_call_rates_again(self):
        record, logs, final, totals, game = self.make_record()
        rates = fixture()[3]
        loader = Mock(return_value=rates)
        game = {**game, 'GameTimeUTC': '2099-11-23T02:00:00Z'}
        def component(**kwargs):
            data = kwargs['data']
            return SimpleNamespace(reply={'request': data['request'], 'status': 'ok', 'full': True,
                                           'records': [r['body'] for r in data['records']], 'results': []})
        with patch.object(view, 'history_component', return_value=component), \
             patch.object(view, '_fixture', (final, game, loader, logs), create=True):
            app = AppTest.from_string('''
import streamlit as st
import defensive_recency_view as view
final, game, rates, logs = view._fixture
view.render_trial(st, final, game, 'FanDuel', rates, logs)
''').run()
            self.assertFalse(app.exception)
            loader.assert_not_called()
            app.toggle(key='show_defensive_recency_trial').set_value(True).run()
            self.assertFalse(app.exception, [e.message for e in app.exception])
            self.assertEqual(loader.call_count, 1)
            self.assertTrue(any('snapshots saved' in c.value for c in app.caption))
            app.toggle(key='defensive_recency_players').set_value(True).run()
            self.assertFalse(app.exception, [e.message for e in app.exception])


if __name__ == '__main__':
    unittest.main()
