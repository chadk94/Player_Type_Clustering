import unittest

import numpy as np
import pandas as pd

from archetype_fallback import ARCHETYPE_BASELINE, build_archetype_priors, OFF_STATS, DEF_STATS
from import_minutes import TEAM_IDS
from live_projections import STAT_NAMES, build_projections
from matchup_projections import apply_matchup, assign_game_archetypes, matchup_effects, prior_season_profiles
from season_pipeline import ROOT, apply_roster_rules
from team_projection_view import TEAM_STATS, team_projection_tables
from market_points import apply_market_points
from team_assists import historical_assist_rates, apply_team_assists
from team_threes import historical_three_rates, apply_team_threes
from team_rebounds import historical_rebound_rates, apply_team_rebounds


class ArchetypeFallbackTests(unittest.TestCase):
    def setUp(self):
        self.roster = pd.DataFrame([dict(PLAYER_ID=i, PLAYER_NAME=f'Player {i}',
            TEAM_ID=TEAM_IDS['CLE'], TEAM_ABBREVIATION='CLE', PROJECTED_MIN=0.,
            OffArchetype='Wing' if i < 4 else 'Big', DefArchetype='Wing' if i == 0 or i > 3 else 'Guard')
            for i in range(7)])
        self.history = pd.DataFrame([dict(PLAYER_ID=i, MIN=10.*i,
            **{stat: i*(1. if i < 4 else 2.) for stat in STAT_NAMES}) for i in range(1,7)])
        self.teams = pd.DataFrame([dict(TEAM_ID=TEAM_IDS['CLE'], GAMES_PLAYED=0)])
        self.priors = build_archetype_priors(self.roster, self.history)

    def project(self, history=None, current=None, teams=None, **kwargs):
        return build_projections(self.roster.iloc[:1], self.history if history is None else history,
            self.teams if teams is None else teams, pd.DataFrame() if current is None else current,
            archetype_priors=self.priors, approved_minutes={0:12.1}, **kwargs).iloc[0]

    def test_roles_supply_distinct_rates_and_exact_minutes(self):
        original = self.history.copy(deep=True)
        row = self.project()
        self.assertEqual(row['Projected minutes'], 12.1)
        for stat in OFF_STATS:
            self.assertAlmostEqual(row[stat], 1.21)
        for stat in DEF_STATS:
            self.assertAlmostEqual(row[stat], 2.42)
        self.assertAlmostEqual(row.REB, row.OREB + row.DREB)
        self.assertAlmostEqual(row.PRA, row.PTS + row.REB + row.AST)
        self.assertEqual(row.Status, ARCHETYPE_BASELINE)
        self.assertEqual(row['Stat baseline'], ARCHETYPE_BASELINE)
        self.assertTrue(row['Estimated stat profile'])
        self.assertIn('3 offensive donors', row['Fallback details'])
        self.assertIn('3 defensive donors', row['Fallback details'])
        pd.testing.assert_frame_equal(self.history, original)

    def test_no_self_donors_or_estimates_without_both_supported_roles(self):
        self.assertEqual(set(self.priors), {0})
        self.assertEqual(build_archetype_priors(self.roster, self.history.iloc[1:]), {})
        invalid = self.history.copy()
        invalid.loc[0,'PTS'] = -1.
        self.assertEqual(build_archetype_priors(self.roster, invalid), {})
        invalid = self.history.copy()
        invalid.loc[0,'FGA'] = np.nan
        self.assertEqual(build_archetype_priors(self.roster, invalid), {})
        for label in ['RotoWire projection', ARCHETYPE_BASELINE]:
            self.assertEqual(build_archetype_priors(self.roster, self.history.assign(BaselineLabel=label)), {})
        self.assertEqual(build_archetype_priors(self.roster.assign(DefArchetype=None), self.history), {})
        self.assertEqual(build_archetype_priors(self.roster.iloc[:1], self.history), {})

    def test_individual_history_and_rotowire_replace_temporary_estimate(self):
        individual = pd.DataFrame([dict(PLAYER_ID=0, MIN=20., **{s: 10. for s in STAT_NAMES})])
        for row in [self.project(history=pd.concat([self.history, individual])),
                    self.project(fallback=individual)]:
            self.assertFalse(row['Estimated stat profile'])
            self.assertEqual(row['Fallback details'], '')
            self.assertEqual(row.Status, 'Available')
            self.assertAlmostEqual(row.PTS, 6.05)
        invalid = individual.assign(MIN=0.)
        self.assertEqual(self.project(fallback=invalid)['Stat baseline'], ARCHETYPE_BASELINE)

    def test_eligible_current_stats_replace_estimate_without_synthetic_blending(self):
        current = pd.DataFrame([dict(PLAYER_NAME='Player 0', ESPN_PLAYER_ID='0', GP=4., MIN=10.,
                                    **{s: 5. for s in STAT_NAMES})])
        self.assertEqual(self.project(current=current)['Stat baseline'], ARCHETYPE_BASELINE)
        row = self.project(current=current, teams=self.teams.assign(GAMES_PLAYED=4))
        self.assertEqual(row['Stat baseline'], 'Current season')
        self.assertFalse(row['Estimated stat profile'])
        self.assertAlmostEqual(row.PTS, 6.05)

    def test_known_players_outputs_are_unchanged(self):
        before = build_projections(self.roster.iloc[1:], self.history, self.teams, pd.DataFrame())
        after = build_projections(self.roster.iloc[1:], self.history, self.teams, pd.DataFrame(), archetype_priors=self.priors)
        pd.testing.assert_frame_equal(before, after)


class ClevelandCoverageRegression(unittest.TestCase):
    def test_hezonja_approved_rotation_restores_all_totals_with_visible_estimate(self):
        roster = apply_roster_rules(pd.read_csv(ROOT / 'preseason_archetypes.csv'))
        logs = pd.read_csv(ROOT / 'data/game_logs_2026.csv.gz', dtype={'ESPN_PLAYER_ID':str, 'GAME_ID':str})
        history = pd.read_csv(ROOT / 'historical_player_data.csv')
        supplement = pd.read_csv(ROOT / 'data/prior_stat_supplements.csv')
        history = pd.concat([history, supplement[~supplement.PLAYER_ID.isin(history.PLAYER_ID)]])
        full = prior_season_profiles(logs, roster)
        history = pd.concat([full, history[~history.PLAYER_ID.isin(full.PLAYER_ID)]], ignore_index=True)
        fallback = pd.read_csv(ROOT / 'data/rotowire_fallback.csv')
        teams = roster[['TEAM_ID']].drop_duplicates().assign(GAMES_PLAYED=0)
        cle = roster[roster.TEAM_ABBREVIATION.eq('CLE')]
        # Synthetic 240-minute fixture; never publish a user's imported rotation.
        allocation = {name:27.5 for name in ['James Harden', 'Donovan Mitchell', 'Jarrett Allen',
            'Sam Merrill', 'Evan Mobley', 'Peyton Watson', 'Craig Porter Jr.', 'Jaylon Tyson']}
        allocation['Mario Hezonja'] = 20.
        minutes = {p.PLAYER_ID:allocation.get(p.PLAYER_NAME,0.) for p in cle.itertuples()}
        before = build_projections(cle,history,teams,pd.DataFrame(),fallback=fallback,approved_minutes=minutes)
        after = build_projections(cle,history,teams,pd.DataFrame(),fallback=fallback,approved_minutes=minutes,
                                  archetype_priors=build_archetype_priors(roster,history))
        # Newly classified zero-minute reserves can now receive an estimate too.
        # Existing individual profiles must remain untouched, and Hezonja must
        # still be the only active estimate in this approved rotation.
        mask = before.Player.ne('Mario Hezonja') & before.PTS.notna()
        pd.testing.assert_frame_equal(before[mask], after[mask])
        self.assertEqual(after['Projected minutes'].sum(), 240.)
        active_estimate = after['Estimated stat profile'] & after['Projected minutes'].gt(0)
        self.assertEqual(after.loc[active_estimate, 'Player'].tolist(), ['Mario Hezonja'])
        effects = matchup_effects(assign_game_archetypes(logs, roster))
        for opponent in ['PHI', 'BOS', 'NYK', 'BKN']:
            matchup = apply_matchup(after,roster,teams,effects,{},opponent)
            totals, _, issues = team_projection_tables(after,matchup,matchup,['CLE'])
            self.assertTrue(np.isfinite(totals[list(TEAM_STATS)]).all().all())
            self.assertEqual(totals.iloc[0]['Stat coverage'], 'Provisional — archetype estimates')
            self.assertEqual(totals.iloc[0]['Estimated players'], 'Mario Hezonja')
            self.assertEqual(len(issues),1)
            self.assertIn('Mario Hezonja (20 minutes)', issues[0])
        # The complete lineup also unlocks each existing team calibration.
        targets = {'CLE':120., 'BKN':108.5}  # Fixed test quotes, not live prices.
        opponents = {'CLE':'BKN', 'BKN':'CLE'}
        adjusted, _, issues = apply_market_points(matchup, targets)
        self.assertEqual(issues, [])
        adjusted, _, issues = apply_team_assists(adjusted, targets, historical_assist_rates(logs, 82), opponents)
        self.assertEqual(issues, [])
        adjusted, _, issues = apply_team_threes(adjusted, targets, historical_three_rates(logs, 82), opponents)
        self.assertEqual(issues, [])
        own, allowed = historical_rebound_rates(logs, 82)
        adjusted, _, issues = apply_team_rebounds(adjusted, targets, own, allowed, opponents)
        self.assertEqual(issues, [])
        totals, breakdown, issues = team_projection_tables(after, matchup, adjusted, ['CLE'])
        self.assertTrue(np.isfinite(totals[list(TEAM_STATS)]).all().all())
        np.testing.assert_allclose(breakdown['Final projection'], breakdown['Team target'])
        np.testing.assert_allclose(adjusted.PRA, adjusted.PTS + adjusted.REB + adjusted.AST)
        pd.testing.assert_series_equal(after['Projected minutes'], adjusted['Projected minutes'])
        self.assertEqual(totals.iloc[0]['Stat coverage'], 'Provisional — archetype estimates')
