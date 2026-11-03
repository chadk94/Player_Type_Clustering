"""The approved hybrid STL/BLK prior changes exposure weighting, not matchups."""
import unittest

import numpy as np
import pandas as pd

from import_minutes import TEAM_IDS
from live_projections import STAT_NAMES, build_projections
from player_rate_priors import load_prior_season_minutes, stocks_prior_minutes


class PlayerRatePriorTests(unittest.TestCase):
    def setUp(self):
        self.roster = pd.DataFrame([dict(PLAYER_ID=1, PLAYER_NAME='Example Player',
            TEAM_ID=TEAM_IDS['ATL'], TEAM_ABBREVIATION='ATL', PROJECTED_MIN=30.,
            OffArchetype='Secondary Playmaker', DefArchetype='POA Defender')])
        self.history = pd.DataFrame([dict(PLAYER_ID=1, MIN=20.,
            **{s:1. for s in STAT_NAMES})])
        self.current = pd.DataFrame([dict(ESPN_PLAYER_ID='1', PLAYER_NAME='Example Player',
            GP=10., MIN=20., **{s:3. for s in STAT_NAMES})])
        self.teams = pd.DataFrame([dict(TEAM_ID=TEAM_IDS['ATL'], GAMES_PLAYED=10)])

    def project(self, exposure=500., **kwargs):
        args = dict(roster=self.roster, history=self.history, current=self.current,
                    teams=self.teams, career_profiles={}, prior_season_minutes={1:exposure})
        args.update(kwargs)
        return build_projections(**args)

    def test_inclusive_cutoff_and_unknown_exposure(self):
        for minutes in [None, np.nan, np.inf, -1., 0., 499., 499.999, 'unknown']:
            with self.subTest(minutes=minutes):
                self.assertEqual(stocks_prior_minutes(minutes),200.)
        for minutes in [500.,500.001,2000.]:
            self.assertEqual(stocks_prior_minutes(minutes),1000.)

    def test_only_steals_and_blocks_change_at_500(self):
        old = self.project(499.999)
        new = self.project(500.)
        for stat in ['STL','BLK']:
            self.assertAlmostEqual(old.iloc[0][stat],2.)  # 200 actual / (200 + 200)
            self.assertAlmostEqual(new.iloc[0][stat],4/3)  # 200 actual / (200 + 1000)
        self.assertAlmostEqual(new.iloc[0]['STL/BLK current weight'],1/6)
        self.assertEqual(new.iloc[0]['STL/BLK prior minutes'],1000.)
        changed = ['STL','BLK','Prior season minutes','STL/BLK prior minutes','STL/BLK current weight']
        pd.testing.assert_frame_equal(old.drop(columns=changed),new.drop(columns=changed))
        for minutes in [0.,499.,np.nan]:
            result = self.project(minutes)
            pd.testing.assert_frame_equal(old.drop(columns=['Prior season minutes']),
                                          result.drop(columns=['Prior season minutes']))

    def test_actual_exposure_determines_weight_not_projected_minutes(self):
        for overrides, expected_minutes in [({'approved_minutes':{1:12.25}},12.25),
            ({'approved_minutes':{1:12.25},'minutes_overrides':{1:30.}},30.),
            ({'approved_minutes':{1:0.}},0.)]:
            row = self.project(**overrides).iloc[0]
            self.assertEqual(row['Projected minutes'],expected_minutes)
            self.assertAlmostEqual(row['STL/BLK current weight'],1/6)
            self.assertAlmostEqual(row.STL,expected_minutes/15)
            self.assertAlmostEqual(row.BLK,expected_minutes/15)
        # Increasing projected minutes cannot qualify a player for the stronger prior.
        self.assertEqual(self.project(499.,minutes_overrides={1:48.}).iloc[0]['STL/BLK prior minutes'],200.)

    def test_existing_game_five_gate_and_missing_current_preserved(self):
        for games in [0,1,2,3]:
            old = self.project(499.,teams=self.teams.assign(GAMES_PLAYED=games)).iloc[0]
            new = self.project(500.,teams=self.teams.assign(GAMES_PLAYED=games)).iloc[0]
            self.assertEqual(old.STL,new.STL)
            self.assertEqual(new['STL/BLK current weight'],0.)
        row = self.project(current=self.current.iloc[:0],approved_minutes={1:30.}).iloc[0]
        self.assertEqual(row.STL,1.5)
        self.assertEqual(row['STL/BLK current weight'],0.)

    def test_rookies_estimates_and_real_zero_stay_actual_first(self):
        for fallback, priors in [(self.history,{}), (None,{1:self.history.iloc[0].to_dict()}), (None,{})]:
            row = self.project(history=self.history.iloc[:0],fallback=fallback,
                               archetype_priors=priors,current=self.current.assign(STL=0.)).iloc[0]
            self.assertEqual(row.STL,0.)
            self.assertEqual(row.BLK,3.)
            self.assertEqual(row['STL/BLK prior minutes'],0.)
            self.assertEqual(row['STL/BLK current weight'],1.)
        # A missing observed category still uses its labeled estimate, not a new blend.
        row = self.project(history=self.history.iloc[:0],fallback=self.history,
                           current=self.current.assign(BLK=np.nan)).iloc[0]
        self.assertEqual(row.BLK,1.)
        self.assertEqual(row['Estimate fallback stats'],'BLK')

    def test_explicit_actual_first_approval_takes_precedence(self):
        # Niang has historical NBA exposure but an approved current-only baseline.
        roster = self.roster.assign(PLAYER_ID=1627777,SEASON_ID=22026)
        history = self.history.assign(PLAYER_ID=1627777)
        row = self.project(roster=roster,history=history,
                           prior_season_minutes={1627777:1500.}).iloc[0]
        self.assertEqual(row['Stat baseline'],'Current season')
        self.assertEqual(row.STL,3.)
        self.assertEqual(row.BLK,3.)

    def test_verified_totals_do_not_round_players_across_cutoff(self):
        totals = load_prior_season_minutes('2026-27')
        self.assertEqual(len(totals),582)
        self.assertAlmostEqual(totals[1642854],499+55/60)  # Asa Newell
        self.assertEqual(totals[1630574],498.)  # Ariel Hukporti
        for pid in [1642854,1630574]:
            self.assertEqual(stocks_prior_minutes(totals[pid]),200.)
        with self.assertRaisesRegex(ValueError,'source season'):
            load_prior_season_minutes('2027-28')


if __name__ == '__main__':
    unittest.main()
