from unittest import TestCase

import pandas as pd

from tabs_minutes_status import minutes_metadata


class TabsMinutesStatusTests(TestCase):
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

