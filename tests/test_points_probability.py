from datetime import datetime, timedelta, timezone
import json
from unittest.mock import patch
import unittest

import numpy as np
import pandas as pd

from points_probability import (points_pmf, line_probabilities, points_quote,
    scoring_components, model_profile, PROBABILITY_COLUMNS, game_points_probabilities)
from points_tracking import snapshot, valid_record, decode_records, grade_records, history_table, evaluation_rows
from sportsbook_edges import game_edges
from tests import test_sportsbook_edges as edge_tests


class PointsProbabilityTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 11, 3, 18, tzinfo=timezone.utc)
        self.event = dict(id='book-event', commence_time=(self.now+timedelta(hours=4)).isoformat())
        self.row = {'PLAYER_ID':1629011, 'Player':'Mitchell Robinson', 'Team':'NYK', 'Opponent':'PHI',
                    'PTS':6.959201608024548, 'FG3M':0., 'Projected minutes':24.,
                    'Actual sample minutes':0., 'Stat baseline':'2025–26 full regular season',
                    'Minutes source':'Approved game minutes'}
        self.quote = {'Stat':'PTS', 'Projected':self.row['PTS'], 'Market status':'Current', 'Book line':6.5,
                      'Over odds':-110, 'Under odds':-110, 'Updated UTC':self.now.isoformat()}

    def record(self, **kwargs):
        row = {**self.row, **self.quote, **points_quote(self.row, self.quote)}
        return snapshot(row, self.event, 'fanduel', {}, 'test', now=self.now, **kwargs)

    def test_pmf_preserves_means_and_component_scoring_support(self):
        for lam in [(0.,3.,0.), (1.,2.,0.), (.2,.5,3.), (5.,12.,5.)]:
            for phi in [0., .1, .5]:
                p = points_pmf(lam, phi)
                self.assertAlmostEqual(p.sum(), 1., places=9)
                self.assertAlmostEqual(p @ np.arange(len(p)), np.dot(lam,[1,2,3]), places=7)
                self.assertTrue((p>=0).all())
                probabilities = [line_probabilities(lam,phi,k+.5)[PROBABILITY_COLUMNS[0]] for k in range(35)]
                self.assertTrue(all(a>=b for a,b in zip(probabilities,probabilities[1:])))
        self.assertTrue((points_pmf((0.,3.,0.), .1)[1::2] == 0).all())
        # Independent simulation of the selected low-volume + gamma mixture.
        rng = np.random.default_rng(1132026)
        n = 300000
        volume = rng.gamma(1/.05, .05, n) * np.where(rng.random(n)<.05,.25,(1-.05*.25)/.95)
        actual = rng.poisson(volume)+2*rng.poisson(2*volume)+3*rng.poisson(.4*volume)
        predicted = line_probabilities((1.,2.,.4),.1,6.)
        for event, key in [(actual>6,0),(actual<6,1),(actual==6,2)]:
            self.assertAlmostEqual(event.mean(),predicted[PROBABILITY_COLUMNS[key]],delta=.004)

    def test_probabilities_do_not_mutate_projection_or_invent_inconsistent_means(self):
        original = self.row.copy()
        p = points_quote(self.row,self.quote)
        self.assertEqual(self.row,original)
        self.assertAlmostEqual(p['Expected FTM']+2*p['Expected 2PM']+3*p['Expected 3PM'],self.row['PTS'])
        self.assertEqual(p['Expected 3PM'],self.row['FG3M'])
        self.assertAlmostEqual(sum(p[c] for c in PROBABILITY_COLUMNS),1.)
        integer = points_quote(self.row,{**self.quote,'Book line':6.})
        self.assertGreater(integer['Exp. P(push)'],0.)
        for row,quote in [({**self.row,'FG3M':3.},self.quote),
                          ({**self.row,'Projected minutes':0.},self.quote),
                          ({**self.row,'PLAYER_ID':999999999},self.quote),
                          (self.row,{**self.quote,'Market status':'Stale quote'})]:
            self.assertTrue(all(np.isnan(points_quote(row,quote)[c]) for c in PROBABILITY_COLUMNS))

    def test_records_are_pregame_immutable_deduplicable_and_keep_manual_variants(self):
        first = self.record()
        self.assertTrue(valid_record(first))
        # The browser transports immutable forecast JSON as text (0.0 stays 0.0).
        self.assertEqual(decode_records([json.loads(json.dumps(json.dumps(first)))]),[first])
        self.assertEqual(decode_records(['{broken',json.dumps({**first,'mean':100}),None]),[])
        self.now += timedelta(minutes=1)
        second = self.record()
        self.assertEqual(first['id'],second['id'])
        manual = self.record(mode='Manual minutes')
        self.assertNotEqual(first['id'],manual['id'])
        self.assertFalse(valid_record({**first,'mean':100}))
        self.event['commence_time'] = self.now.isoformat()
        self.assertIsNone(self.record())
        self.event['commence_time'] = (self.now+timedelta(hours=1)).isoformat()
        self.now += timedelta(minutes=20)
        self.assertIsNone(self.record())

    def test_results_match_exact_player_game_and_exclude_missing_appearances(self):
        r = self.record()
        log = dict(ESPN_PLAYER_ID=r['espn_id'], GAME_ID='123', GAME_DATE=r['start'],
                   Team='NYK', Opponent='PHI', PTS=6., MIN=23.)
        logs = pd.DataFrame([log])
        results = grade_records([r],logs)
        self.assertEqual(results[0]['outcome'],'Under')
        for changed in [dict(ESPN_PLAYER_ID='9999'),dict(Team='BOS'),dict(Opponent='DET'),
                        dict(GAME_DATE='2026-11-04T22:00:00Z'),dict(PTS=np.nan)]:
            self.assertFalse(grade_records([r],pd.DataFrame([{**log,**changed}])))
        self.assertFalse(grade_records([r],pd.DataFrame([log,log])))
        self.quote['Book line']=6.
        integer=self.record()
        self.assertEqual(grade_records([integer],logs)[0]['outcome'],'Push')

    def test_evaluation_selects_latest_before_checking_results_and_keeps_manual_separate(self):
        old = self.record()
        self.quote['Book line'] = 7.5
        self.now += timedelta(minutes=1)
        new = self.record()
        manual = self.record(mode='Manual minutes')
        frame = history_table([old,new,manual],[{'id':old['id'],'points':6.,'game':'123'}])
        selected = evaluation_rows(frame)
        self.assertEqual(selected.id.tolist(),[new['id']])
        self.assertTrue(selected.actual_points.isna().all())

    def test_adding_probabilities_does_not_change_edges_or_order(self):
        fixture = edge_tests.GameEdgeTests(); fixture.setUp()
        fixture.players['PTS'] = 15.
        fixture.players['FGM'] = 6.
        fixture.players['FTM'] = 2.
        before,count = game_edges(fixture.players,fixture.payload,fixture.event,'fanduel',now=fixture.now)
        probs = game_points_probabilities(fixture.players,fixture.payload,'fanduel',{},now=fixture.now)
        after = before.merge(probs[['Player','Team','Stat',*PROBABILITY_COLUMNS]],
                             on=['Player','Team','Stat'],how='left',validate='many_to_one',sort=False)
        pd.testing.assert_frame_equal(before.reset_index(drop=True),after[before.columns])
