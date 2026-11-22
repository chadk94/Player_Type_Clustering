from unittest import TestCase
from unittest.mock import patch, Mock

import pandas as pd

from tabs_minutes_status import minutes_metadata, render_tabs_minutes_status


class TabsMinutesStatusTests(TestCase):
    def test_switching_games_dates_or_sources_remounts_the_marker(self):
        game = dict(Away='DET', Home='ATL', GameTimeUTC=pd.Timestamp('2026-11-22T20:40:00Z'))
        other = dict(Away='OKC', Home='MIA', GameTimeUTC=pd.Timestamp('2026-11-22T23:10:00Z'))
        component = Mock()
        state = {'tabs_minutes_reference': self.refs()}
        with patch('tabs_minutes_status.st.session_state', state), patch('tabs_minutes_status.st.runtime.get_instance'), patch('tabs_minutes_status._status_component', return_value=component):
            def render(g):
                render_tabs_minutes_status([g], 'edges_game_tabs_status')
                return component.call_args.kwargs
            first = render(game)
            second = render(other)
            self.assertNotEqual(first['key'], second['key'])
            self.assertEqual([s['team'] for s in second['data']['scopes']], ['OKC', 'MIA'])
            # A new publication updates the same marker without remounting it.
            state['tabs_minutes_reference'] = self.refs().assign(source_updated_at='2026-11-18T22:10:00Z')
            self.assertEqual(second['key'], render(other)['key'])
            self.assertNotEqual(first['key'], render(dict(game, GameTimeUTC=game['GameTimeUTC']+pd.Timedelta(days=1)))['key'])
            state['tabs_minutes_mode'] = 'CSV upload'
            csv = render(game)
            self.assertNotEqual(first['key'], csv['key'])
            self.assertFalse(csv['data']['browser'])

    def refs(self):
        return pd.DataFrame([dict(Team='CHI', Opponent='BKN', game_date='2026-11-18',
            PLAYER_ID=i, minutes=30., source_saved=False,
            source_updated_at='2026-11-18T22:00:00Z', updated_at='2026-11-18T22:30:00Z')
            for i in range(8)])

    def test_publication_time_is_not_capture_time_and_inputs_are_unchanged(self):
        refs = self.refs()
        original = refs.copy(deep=True)
        row, = minutes_metadata(refs)
        self.assertEqual(row['published'], '2026-11-18T22:00:00+00:00')
        self.assertEqual(row['captured'], '2026-11-18T22:30:00+00:00')
        self.assertTrue(row['complete'])
        self.assertFalse(row['saved'])
        pd.testing.assert_frame_equal(refs, original)

    def test_mixed_times_use_oldest_and_incomplete_or_saved_coverage_stays_visible(self):
        refs = self.refs()
        refs.loc[0, 'source_updated_at'] = '2026-11-18T21:00:00Z'
        refs.loc[0, 'source_saved'] = True
        refs.attrs['incomplete_teams'] = ['CHI']
        row, = minutes_metadata(refs)
        self.assertEqual(row['published'], '2026-11-18T21:00:00+00:00')
        self.assertTrue(row['saved'])
        self.assertFalse(row['complete'])
        refs.attrs.clear()
        self.assertFalse(minutes_metadata(refs.iloc[:3])[0]['complete'])

    def test_missing_dates_are_not_fabricated_and_rematches_stay_separate(self):
        refs = self.refs()
        refs.loc[0, 'source_updated_at'] = None
        self.assertIsNone(minutes_metadata(refs)[0]['published'])
        other = self.refs().assign(game_date='2026-11-20')
        self.assertEqual(len(minutes_metadata(pd.concat([refs, other]))), 2)
        self.assertEqual(minutes_metadata(None), [])

