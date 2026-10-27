"""Exercise real Streamlit fragment reruns, with all external feeds stubbed."""
from dataclasses import replace
from unittest.mock import patch
import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.local_script_runner import LocalScriptRunner

import projection_dashboard as dashboard
from live_projections import STAT_NAMES
from season_pipeline import ROOT
from approved_minutes import digest, validate_bundle
from import_minutes import TEAM_IDS
from schedule_feed import ScheduleUnavailable


class DashboardFragmentTests(unittest.TestCase):
    def setUp(self):
        self.roster = pd.read_csv(ROOT / 'preseason_archetypes.csv')
        teams = self.roster[['TEAM_ID', 'TEAM_ABBREVIATION']].drop_duplicates()
        teams['ESPN_TEAM_ID'] = teams.TEAM_ABBREVIATION
        teams['GAMES_PLAYED'] = 4  # Exercise the newly active current-season path.
        players = self.roster[['PLAYER_NAME']].copy()
        players['ESPN_PLAYER_ID'] = self.roster.PLAYER_ID.astype(str)
        players['GP'] = 4
        players['MIN'] = 24.
        for stat in STAT_NAMES:
            players[stat] = 3.
        state = dict(teams=teams, players=players, errors=[], as_of='test snapshot')
        logs = pd.read_csv(ROOT / 'data/game_logs_2026.csv.gz',
                           dtype={'ESPN_PLAYER_ID': str, 'GAME_ID': str})
        self.live = self.mock('live_state', return_value=state)
        self.logs = self.mock('current_matchup_logs', return_value=(logs, {'errors': []}))
        self.imported = self.mock('render_import', return_value=None)
        self.schedule = self.mock('today_schedule', return_value=pd.DataFrame())
        self.mock('next_team_game', side_effect=lambda season, espn, team: {
            'Opponent': 'HOU' if team != 'HOU' else 'ATL',
            'GameTimeUTC': '2026-10-27T23:00:00Z',
        })
        self.mock('get_odds_api_key', return_value=None)
        self.build = self.mock('build_projections', wraps=dashboard.build_projections)
        self.effects = self.mock('matchup_effects', wraps=dashboard.matchup_effects)
        # Catch accidental network access, rather than letting a test make live calls.
        self.mock('fetch_json', side_effect=AssertionError('Unexpected network request'))
        self.mock('read_feed', side_effect=AssertionError('Unexpected sportsbook request'))

        # AppTest 1.53 always asks for a full rerun. Retain the same fragment
        # storage and send the widget's fragment ID, just as the browser does.
        # All rendering, widget callbacks and calculations remain real.
        owner = self
        self.storage = None
        self.fragment_id = None

        class FragmentRunner(LocalScriptRunner):
            def __init__(runner, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if owner.storage is None:
                    owner.storage = runner._fragment_storage
                else:
                    runner._fragment_storage = owner.storage

            def request_rerun(runner, data):
                return super().request_rerun(replace(data, fragment_id=owner.fragment_id))

        runner_patch = patch('streamlit.testing.v1.app_test.LocalScriptRunner', FragmentRunner)
        runner_patch.start()
        self.addCleanup(runner_patch.stop)
        self.app = AppTest.from_string('''
import pandas as pd
import streamlit as st
from season_pipeline import ROOT
from projection_dashboard import render_live_dashboard
render_live_dashboard(st, pd.read_csv(ROOT / 'preseason_archetypes.csv'))
''', default_timeout=20).run()
        self.assert_no_errors()

    def mock(self, name, **kwargs):
        patcher = patch.object(dashboard, name, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def fragment_for(self, widget):
        return self.app.session_state._state._get_widget_metadata(widget.id).fragment_id

    def assert_no_errors(self):
        self.assertFalse(self.app.exception, [e.message for e in self.app.exception])

    def test_player_and_minutes_updates_skip_league_refresh_and_other_tabs(self):
        select = self.app.selectbox(key='projection_player')
        self.fragment_id = self.fragment_for(select)
        self.assertTrue(self.fragment_id)
        parent_id = self.fragment_for(next(b for b in self.app.button if b.label == 'Refresh live data'))
        self.assertNotEqual(self.fragment_id, parent_id)
        counts = [m.call_count for m in (self.live, self.logs, self.effects, self.imported, self.schedule, self.build)]
        next_player = next(name for name in select.options if name != select.value)
        select.select(next_player).run()
        self.assert_no_errors()
        self.assertEqual([m.call_count for m in (self.live, self.logs, self.effects, self.imported, self.schedule, self.build)], counts)
        self.assertIn(f'{next_player} vs HOU', [h.value for h in self.app.subheader])
        self.assertFalse(self.app.tabs)  # Only the changed tab emitted deltas.

        before = self.app.dataframe[0].value.set_index('Stat')
        self.app.slider[0].set_value(12.).run()
        self.assert_no_errors()
        after = self.app.dataframe[0].value.set_index('Stat')
        self.assertAlmostEqual(after.loc['PTS', 'Projected'], before.loc['PTS', 'Projected'] / 2)
        self.assertAlmostEqual(after.loc['PRA', 'Projected'],
                               after.loc[['PTS', 'REB', 'AST'], 'Projected'].sum())
        self.assertEqual(self.build.call_count, counts[-1] + 1)
        self.assertEqual([m.call_count for m in (self.live, self.logs, self.effects, self.imported, self.schedule)], counts[:-1])
        next(b for b in self.app.button if b.label == 'Use automatic minutes').click().run()
        self.assert_no_errors()
        self.assertEqual(self.app.slider[0].value, 24.)

    def test_other_tabs_have_distinct_reruns_and_refresh_rebuilds_snapshot(self):
        keys = ['projection_team', 'team_projection_book', 'archetype_team']
        ids = [self.fragment_for(self.app.selectbox(key=key)) for key in keys]
        ids.append(self.fragment_for(self.app.date_input(key='matchup_date')))
        refresh = next(b for b in self.app.button if b.label == 'Refresh live data')
        ids.append(self.fragment_for(refresh))
        self.assertEqual(len(set(ids)), len(ids))
        self.fragment_id = ids[-1]
        counts = (self.live.call_count, self.logs.call_count, self.imported.call_count)
        refresh.click().run()
        self.assert_no_errors()
        self.assertEqual((self.live.call_count, self.logs.call_count, self.imported.call_count), tuple(n+1 for n in counts))
        self.live.clear.assert_called_once()
        self.logs.clear.assert_called_once()
        self.assertEqual(len(self.app.tabs), 5)

    def test_schedule_failure_has_retry_not_a_false_empty_slate(self):
        self.fragment_id = self.fragment_for(self.app.date_input(key='matchup_date'))
        self.schedule.side_effect = ScheduleUnavailable('temporary feed failure')
        counts = [m.call_count for m in (self.live, self.logs, self.imported, self.build)]
        self.app.run()
        self.assert_no_errors()
        self.assertTrue(any('temporarily unavailable' in w.value for w in self.app.warning))
        self.assertFalse(any('No upcoming' in i.value for i in self.app.info))
        retry = self.app.button(key='matchup_schedule_reload')
        self.assertEqual(retry.label, 'Retry schedule')
        self.schedule.side_effect = None
        self.schedule.return_value = pd.DataFrame()
        retry.click().run()
        self.assert_no_errors()
        self.assertTrue(any('No upcoming' in i.value for i in self.app.info))
        self.assertFalse(self.app.warning)
        self.assertEqual(self.app.button(key='matchup_schedule_reload').label, 'Reload schedule')
        self.schedule.clear.assert_called_once_with(dashboard.SEASON, self.app.date_input(key='matchup_date').value)
        self.assertEqual([m.call_count for m in (self.live, self.logs, self.imported, self.build)], counts)

    def test_new_approved_import_replaces_game_cache_for_all_tabs(self):
        selected = self.app.selectbox(key='projection_player').value
        game_key = f'2026-10-27_{TEAM_IDS["ATL"]}_{TEAM_IDS["HOU"]}'
        approvals = []
        for team, opponent in [('ATL', 'HOU'), ('HOU', 'ATL')]:
            rows = self.roster[self.roster.TEAM_ABBREVIATION.eq(team)].copy()
            # Include the selected player first; no private minutes are used.
            rows['selected'] = rows.PLAYER_NAME.eq(selected)
            rows = rows.sort_values(['selected', 'PROJECTED_MIN'], ascending=False).head(6)
            approvals.append(dict(schema='nba_approved_minutes_v1', game_date='2026-10-27',
                game_key=game_key, team_id=TEAM_IDS[team], team=team,
                opponent_id=TEAM_IDS[opponent], approved_at='2026-10-26T12:00:00Z',
                approved_by='test', approval_version=1, warnings=[],
                players=[dict(player_id=int(row.PLAYER_ID), player_name=row.PLAYER_NAME,
                              minutes=[39.9, 40.1, 40., 40., 40., 40.][i],
                              starter=i<5, injury_status='Available')
                         for i, row in enumerate(rows.itertuples())]))
        bundle = dict(schema='nba_approved_slate_v1', approvals=approvals,
                      games=[dict(game_key=game_key, game_date='2026-10-27',
                                  away_team_id=TEAM_IDS['ATL'], home_team_id=TEAM_IDS['HOU'])])
        bundle['bundle_id'] = digest(bundle)
        self.imported.return_value = validate_bundle(bundle, self.roster, '2026-27')
        refresh = next(b for b in self.app.button if b.label == 'Refresh live data')
        self.fragment_id = self.fragment_for(refresh)
        self.app.run()  # An upload belongs to the parent dashboard fragment.
        self.assert_no_errors()
        before = self.app.dataframe[0].value.set_index('Stat')
        self.assertTrue(any('39.9 minutes (Approved game minutes)' in c.value for c in self.app.caption))

        bundle['approvals'][0]['players'][0]['minutes'] = 34.9
        bundle['approvals'][0]['players'][1]['minutes'] = 45.1
        bundle.pop('bundle_id')
        bundle['bundle_id'] = digest(bundle)
        self.imported.return_value = validate_bundle(bundle, self.roster, '2026-27')
        self.app.run()
        self.assert_no_errors()
        after = self.app.dataframe[0].value.set_index('Stat')
        self.assertTrue(any('34.9 minutes (Approved game minutes)' in c.value for c in self.app.caption))
        self.assertAlmostEqual(after.loc['PTS', 'Baseline'], before.loc['PTS', 'Baseline'] * 34.9 / 39.9)


if __name__ == '__main__':
    unittest.main()
