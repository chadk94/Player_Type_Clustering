from unittest import TestCase
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

from edge_calculation_cache import GameCalculationCache, source_deadlines
from edge_review import assess_edges, scenario_projection
from minutes_blend import minutes_plan
from test_edge_review import raw_team, finalized, reference, edge, healthy, NOW, GAME
from test_minutes_blend import references


class GameCacheTests(TestCase):
    def test_rotations_reused_and_copied_but_source_values_and_coverage_invalidate(self):
        base = raw_team()
        refs = references(base.set_index('PLAYER_ID')['Projected minutes'].to_dict())
        cache = GameCalculationCache()
        calculate = Mock(side_effect=lambda: minutes_plan(base, refs, GAME, NOW))
        def get():
            return cache.get((base, refs, GAME), NOW, calculate, lambda: source_deadlines(refs))
        first = get()
        first[0]['blended'][1] = 999
        first[1].loc[0, 'Your minutes'] = 999
        repeated = get()
        self.assertEqual(repeated[0]['blended'][1], 30.)
        self.assertEqual(repeated[1].loc[0, 'Your minutes'], 30.)
        self.assertEqual(calculate.call_count, 1)
        refs.loc[:1, 'minutes'] = [20., 40.]
        self.assertEqual(get()[0]['blended'][1], 25.)
        refs.attrs['incomplete_teams'] = ['BOS']
        self.assertFalse(get()[0]['blended'])
        self.assertEqual(calculate.call_count, 3)

    def test_clock_boundaries_never_extend_minutes_or_injury_freshness(self):
        cache = GameCalculationCache()
        refs = references({1:30.})
        frame = finalized(raw_team())
        def get(now):
            return cache.get((refs, frame), now,
                lambda: assess_edges(edge(), frame, GAME, pd.DataFrame(), {}, refs, healthy(), NOW, now),
                lambda: source_deadlines(refs, NOW))
        for now in (NOW-pd.Timedelta(nanoseconds=1), NOW,
                    NOW+pd.Timedelta(minutes=30), NOW+pd.Timedelta(minutes=30,nanoseconds=1),
                    NOW+pd.Timedelta(hours=2), NOW+pd.Timedelta(hours=2,nanoseconds=1)):
            actual = get(now)
            expected = assess_edges(edge(), frame, GAME, pd.DataFrame(), {}, refs, healthy(), NOW, now)
            cols = [c for c in expected if c != 'Review captured UTC']
            pd.testing.assert_frame_equal(actual[cols], expected[cols])

    def test_cache_invalidates_price_injury_passing_and_source_status(self):
        cache = GameCalculationCache()
        e, info, refs = edge(), healthy(), reference()
        passing = pd.DataFrame({'POTENTIAL_AST':[5.]})
        calc = Mock(return_value={'value':1})
        def get():return cache.get((e,info,refs,passing), NOW, calc)
        get();get();self.assertEqual(calc.call_count,1)
        e.loc[0,'Book implied total'] = 20.;get()
        info['BOS'].statuses['Player 2'] = 'Questionable';get()
        passing.loc[0,'POTENTIAL_AST'] = 10.;get()
        refs['source_saved'] = True;get()
        self.assertEqual(calc.call_count,5)
        cache.clear();get();self.assertEqual(calc.call_count,6)

    def test_role_reuse_matches_independent_stat_checks_and_keeps_means(self):
        frame = finalized(raw_team())
        fields = {'PTS':'PTS','AST':'AST','PRA':'PRA','3PM':'FG3M'}
        edges = pd.concat([edge(float(frame.iloc[0][field]),10.,stat=stat) for stat,field in fields.items()],ignore_index=True)
        signal = dict(ratios={'PTS':.8,'AST':.7,'FG3A':1.2}, trends={'PTS':-.25,'AST':-.3,'FG3A':.3},detail='Recent volume')
        args = (frame,GAME,pd.DataFrame(),{},reference(),healthy(),NOW,NOW)
        cases = dict(yours=frame.copy(),tabs=frame.copy())
        cases['tabs'].loc[0,'PTS'] = 14.
        with patch('edge_review.role_evidence', return_value=signal):
            expected = pd.concat([assess_edges(row.to_frame().T,*args,minutes_cases=cases) for _,row in edges.iterrows()],ignore_index=True)
            with patch('edge_review.scenario_projection',wraps=scenario_projection) as scenario:
                actual = assess_edges(edges,*args,minutes_cases=cases)
                self.assertEqual(scenario.call_count,3)  # blended + each source, not each stat
        for col in ['Recent test projection','Recent test edge %','Combined test projection','Combined test edge %']:
            np.testing.assert_allclose(actual[col],expected[col])
        for col in ['Minutes flag','Role flag','Injury flag','Robust play']:
            self.assertEqual(actual[col].tolist(),expected[col].tolist())
        pd.testing.assert_frame_equal(actual[edges.columns],edges)
