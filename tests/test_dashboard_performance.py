from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

import pandas as pd

import matchup_data
import projection_dashboard as dashboard
from live_projections import CompletedTeamCache
from tests.test_team_box_coverage import box


class DashboardPerformanceTests(TestCase):
    def test_changed_counts_incomplete_responses_and_expiry_refetch(self):
        cache = CompletedTeamCache()
        read = Mock(side_effect=[set(), {'a'}, {'a', 'b'}, {'b', 'c'}])
        def get(games):
            return cache.get(('schedule', 2027, 'ATL'), games, read, lambda ids: len(ids) == games)
        with patch('live_projections.monotonic', return_value=100):
            self.assertEqual(get(1), set())
            self.assertEqual(get(1), {'a'})
            self.assertEqual(get(1), {'a'})
            self.assertEqual(get(2), {'a', 'b'})
        with patch('live_projections.monotonic', return_value=1901):
            self.assertEqual(get(2), {'b', 'c'})
        self.assertEqual(read.call_count, 4)

    def test_bad_archived_box_is_replaced_then_reused(self):
        schedule = {'team': {'id': '1'}, 'events': [{'id':'g', 'season':{'year':2027},
                    'seasonType':{'type':2}, 'competitions':[{'status':{'type':{'completed':True}}}]}]}
        fetch = Mock(side_effect=[schedule, box()])
        with TemporaryDirectory() as tmp, patch('matchup_data.fetch_json', fetch):
            folder = Path(tmp)
            (folder / 'g.json').write_text('{}')
            cache = CompletedTeamCache()
            for _ in range(3):
                logs, meta = matchup_data.load_game_logs(2027, ['1'], folder,
                    game_counts={'1':1}, team_cache=cache)
                self.assertEqual(meta['errors'], [])
                self.assertEqual(set(logs.Team), {'ATL','BOS'})
        self.assertEqual(fetch.call_count, 2)

    def test_checkpoint_skips_only_currently_verified_completed_games(self):
        archive = matchup_data.bundled_box_rows(2027)
        self.assertIsNotNone(archive)
        archived_id = archive.GAME_ID.iloc[0]
        def event(game_id):
            return {'id':game_id, 'season':{'year':2027}, 'seasonType':{'type':2},
                    'competitions':[{'status':{'type':{'completed':True}}}]}
        schedule = {'team':{'id':'1'}, 'events':[event(archived_id), event('g')]}
        with patch('matchup_data.fetch_json', side_effect=[schedule, box()]) as fetch:
            rows, meta = matchup_data.load_game_logs(2027, ['1'], archive=archive)
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(set(rows.GAME_ID), {archived_id, 'g'})
        self.assertEqual(meta['errors'], [])
        expected = archive.loc[archive.GAME_ID.eq(archived_id)].reset_index(drop=True)
        pd.testing.assert_frame_equal(rows.loc[rows.GAME_ID.eq(archived_id)].reset_index(drop=True), expected)

    def test_slate_market_reuse_checks_freshness_and_changed_minutes(self):
        now = datetime.now(timezone.utc)
        event = dict(id='game', home_team='Boston Celtics', away_team='Atlanta Hawks',
                     commence_time=(now + timedelta(hours=3)).isoformat())
        markets = [dict(key='spreads', outcomes=[dict(name=n, point=p, price=-110) for n,p in
                   [('Boston Celtics',-2),('Atlanta Hawks',2)]]),
                   dict(key='totals', outcomes=[dict(name=n, point=225, price=-110) for n in ['Over','Under']])]
        payload = {**event,'bookmakers':[dict(key='fanduel', last_update=now.isoformat(), markets=markets)]}
        rates = {}
        frame = pd.DataFrame({'Projected minutes':[20.]})
        with patch.object(dashboard,'sportsbook_slate_lines',return_value=[payload]), \
             patch.object(dashboard,'sportsbook_game_lines',return_value={}) as individual, \
             patch.object(dashboard.st,'session_state',{}), \
             patch.object(dashboard,'calculate_team_adjustments',side_effect=lambda ui,f,*a:f.assign(PTS=10)) as calc:
            def render(f):
                return dashboard.team_projection_preview(Mock(),f,event,'fanduel','key',rates)
            first = render(frame)
            first.loc[0,'PTS'] = 999
            self.assertEqual(render(frame).PTS.tolist(),[10])
            self.assertEqual(calc.call_count,1)
            render(frame * 2)
            self.assertEqual(calc.call_count,2)
            individual.assert_not_called()
            payload['bookmakers'][0]['last_update'] = (now-timedelta(minutes=16)).isoformat()
            self.assertNotIn('PTS',render(frame))
            individual.assert_called_once()
