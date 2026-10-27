"""Actual regular-season rates replace synthetic profiles without a prior blend."""
import unittest

import numpy as np
import pandas as pd

from archetype_fallback import ARCHETYPE_BASELINE
from import_minutes import TEAM_IDS
from live_projections import STAT_NAMES, build_projections


class RookieActualTests(unittest.TestCase):
    def setUp(self):
        self.roster = pd.DataFrame([dict(PLAYER_ID=1, PLAYER_NAME='Example Rookie',
            TEAM_ID=TEAM_IDS['BOS'], TEAM_ABBREVIATION='BOS', PROJECTED_MIN=32.,
            OffArchetype='Secondary Playmaker', DefArchetype='POA Defender')])
        self.history = pd.DataFrame(columns=['PLAYER_ID','MIN',*STAT_NAMES])
        self.fallback = pd.DataFrame([dict(PLAYER_ID=1, MIN=20., **{s:10. for s in STAT_NAMES})])
        self.current = pd.DataFrame([dict(PLAYER_NAME='Example Rookie', ESPN_PLAYER_ID='1',
            GP=1., MIN=16., **{s:2. for s in STAT_NAMES})])
        self.teams = pd.DataFrame([dict(TEAM_ID=TEAM_IDS['BOS'], GAMES_PLAYED=1)])
        self.archetypes = {1:dict(MIN=1., **{s:.25 for s in STAT_NAMES},
            BaselineLabel=ARCHETYPE_BASELINE, **{'Fallback details':'Temporary donors'})}

    def project(self, **kwargs):
        params = dict(roster=self.roster, history=self.history, current=self.current,
            teams=self.teams, fallback=self.fallback, archetype_priors=self.archetypes)
        params.update(kwargs)
        return build_projections(**params).iloc[0]

    def test_first_appearance_replaces_roto_archetype_or_blank_profile(self):
        for fallback, priors in [(self.fallback,self.archetypes), (None,self.archetypes), (None,{})]:
            with self.subTest(roto=fallback is not None, archetype=bool(priors)):
                row = self.project(fallback=fallback, archetype_priors=priors)
                for stat in STAT_NAMES:
                    self.assertEqual(row[stat],4.)
                self.assertEqual(row.PRA,12.)
                self.assertEqual(row['Stat baseline'],'Current season')
                self.assertEqual(row.Status,'Available')
                self.assertFalse(row['Estimated stat profile'])
                self.assertEqual(row['Estimate fallback stats'],'')
                self.assertEqual(row['Actual sample minutes'],16.)
                self.assertEqual(row['Offensive archetype'],'Secondary Playmaker')
                self.assertEqual(row['Defensive archetype'],'POA Defender')

    def test_no_synthetic_weight_after_game_five_and_minutes_keep_precedence(self):
        original = self.roster.copy(deep=True)
        for completed in [1,3,4,10]:
            for minute_args, minutes in [({'approved_minutes':{1:12.25}},12.25),
                ({'approved_minutes':{1:12.25},'minutes_overrides':{1:30.}},30.),
                ({'approved_minutes':{1:0.}},0.)]:
                row = self.project(teams=self.teams.assign(GAMES_PLAYED=completed), **minute_args)
                self.assertEqual(row['Projected minutes'],minutes)
                self.assertEqual(row.PTS,2/16*minutes)
                self.assertEqual(row.PRA,row.PTS+row.REB+row.AST)
        pd.testing.assert_frame_equal(self.roster,original)

    def test_missing_actual_categories_use_labeled_fallback_but_zeros_do_not(self):
        current = self.current.copy()
        current.loc[0,'STL'] = 0.
        for value in [np.nan, -1., np.inf]:
            current.loc[0,'BLK'] = value
            row = self.project(current=current)
            self.assertEqual(row.STL,0.)
            self.assertEqual(row.BLK,16.)
            self.assertEqual(row.PTS,4.)
            self.assertEqual(row['Stat baseline'],'Current season + RotoWire fallback')
            self.assertEqual(row['Estimate fallback stats'],'BLK')
        row = self.project(current=current,fallback=None)
        self.assertEqual(row.BLK,8.)
        self.assertTrue(row['Estimated stat profile'])
        self.assertEqual(row.Status,ARCHETYPE_BASELINE)
        self.assertIn('archetype fallback',row['Stat baseline'])
        row = self.project(current=current,fallback=None,archetype_priors={})
        self.assertTrue(np.isnan(row.BLK))
        self.assertEqual(row.Status,'Some stat rates unavailable')

    def test_no_games_bad_sample_or_ambiguous_name_retains_fallback(self):
        for current in [self.current.iloc[:0], self.current.assign(GP=0.),
                        self.current.assign(MIN=0.), self.current.assign(MIN=np.nan),
                        pd.concat([self.current,self.current.assign(ESPN_PLAYER_ID='2')])]:
            row = self.project(current=current)
            self.assertEqual(row['Stat baseline'],'RotoWire 2026–27 projection')
            self.assertEqual(row.PTS,16.)
        self.assertEqual(self.project(teams=self.teams.assign(GAMES_PLAYED=0))['Stat baseline'],
                         'RotoWire 2026–27 projection')

    def test_existing_individual_history_keeps_original_veteran_transition(self):
        history = self.fallback.copy()
        self.assertEqual(self.project(history=history).PTS,16.)
        row = self.project(history=history,teams=self.teams.assign(GAMES_PLAYED=4))
        self.assertEqual(row['Stat baseline'],'Prior season + current season')
        self.assertAlmostEqual(row.PTS, ((1-16/216)*.5 + (16/216)*.125)*16)


if __name__ == '__main__':
    unittest.main()
