"""Regression checks for the user's October 27 archetype approvals."""
import json
import tempfile
import unittest

import pandas as pd

from live_projections import STAT_NAMES, build_projections
from preseason_bootstrap import bootstrap
from season_pipeline import ROOT, apply_roster_rules, load_rules


class ApprovedRookieRolesTests(unittest.TestCase):
    def test_42_approvals_and_untouched_pickett_override_reach_all_exports(self):
        approved = pd.read_csv(ROOT/'data/audit/rookie_roles_approved_2026-10-27.csv')
        self.assertEqual(len(approved),42)
        self.assertEqual(approved.Group.value_counts().to_dict(),{'Low sample':23,'Provisional':19})
        self.assertEqual(approved.OffWasManual.sum(),0)
        self.assertEqual(approved.DefWasManual.sum(),1)
        self.assertEqual(approved.loc[approved.DefWasManual,'PLAYER_NAME'].tolist(),['Jalen Pickett'])
        rules = load_rules()
        roster = pd.read_csv(ROOT/'preseason_archetypes.csv').set_index('PLAYER_ID')
        for side,filename in [('Off','player_clusters_detailed.csv'),('Def','def_player_clusters_detailed.csv')]:
            detailed = pd.read_csv(ROOT/filename).set_index('PLAYER_ID')
            for row in approved.itertuples():
                role = getattr(row,side+'Cluster')
                self.assertEqual(rules[side.upper()+'_CLUSTER_OVERRIDES'][row.PLAYER_NAME],role)
                self.assertEqual(roster.loc[row.PLAYER_ID,side+'Cluster'],role)
                self.assertEqual(detailed.loc[row.PLAYER_ID,'Cluster'],role)
                self.assertEqual(roster.loc[row.PLAYER_ID,side+'AssignmentSource'],'manual_override')

    def test_explicit_user_corrections_and_low_sample_defaults(self):
        roster = pd.read_csv(ROOT/'preseason_archetypes.csv').set_index('PLAYER_NAME')
        corrections = {'Nate Ament':(4,5),'Carter Bryant':(4,11),'Alpha Diallo':(4,11),
                       'Jonathan Mogbo':(4,5),'Jamison Battle':(0,12),
                       'Christian Anderson':(6,0),'Trevon Brazile':(9,8)}
        for name,roles in corrections.items():
            self.assertEqual(tuple(roster.loc[name,['OffCluster','DefCluster']]),roles)
        exceptions = {'Baba Miller':1,'Koa Peat':1,'Zuby Ejiofor':1,'Hannes Steinbach':8,
                      'Tarris Reed Jr.':8,'Nick Martinelli':8,'Chris Cenac Jr.':9,'Ebuka Okorie':2}
        approved = pd.read_csv(ROOT/'data/audit/rookie_roles_approved_2026-10-27.csv')
        for row in approved[approved.Group.eq('Low sample')].itertuples():
            self.assertEqual(roster.loc[row.PLAYER_NAME,'OffCluster'],exceptions.get(row.PLAYER_NAME,4))
            self.assertEqual(roster.loc[row.PLAYER_NAME,'DefCluster'],{'G':0,'F':12,'C':7}[row.Position])

    def test_stale_exports_and_later_bootstrap_cannot_remove_approved_roles(self):
        approved = pd.concat([pd.read_csv(ROOT/'data/audit'/filename) for filename in
            ['rookie_roles_approved_2026-10-27.csv','reserve_roles_approved_2026-10-27.csv']])
        stale = pd.read_csv(ROOT/'preseason_archetypes.csv')
        for side in ['Off','Def']:
            stale.loc[stale.PLAYER_ID.isin(approved.PLAYER_ID),side+'Cluster'] = 3
        repaired = apply_roster_rules(stale).set_index('PLAYER_ID')
        with tempfile.TemporaryDirectory() as directory:
            bootstrap(ROOT/'projected_minutes.csv',directory)
            rebuilt = pd.read_csv(f'{directory}/preseason_archetypes.csv').set_index('PLAYER_ID')
        for row in approved.itertuples():
            for side in ['Off','Def']:
                for result in [repaired,rebuilt]:
                    self.assertEqual(result.loc[row.PLAYER_ID,side+'Cluster'],getattr(row,side+'Cluster'))

    def test_requested_reserves_keep_roles_with_zero_minutes(self):
        expected = {'Cam Whitmore':(5,12),'DaRon Holmes II':(9,7),'KJ Simpson':(6,11),
                    'Ryan Nembhard':(2,0),'Zeke Nnaji':(1,7)}
        roster = pd.read_csv(ROOT/'preseason_archetypes.csv').set_index('PLAYER_NAME')
        rules = load_rules()
        for side,filename in [('Off','player_clusters_detailed.csv'),('Def','def_player_clusters_detailed.csv')]:
            export = pd.read_csv(ROOT/filename).set_index('PLAYER_NAME')
            for name,roles in expected.items():
                role = roles[0 if side == 'Off' else 1]
                self.assertEqual(roster.loc[name,side+'Cluster'],role)
                self.assertEqual(export.loc[name,'Cluster'],role)
                self.assertEqual(rules[side.upper()+'_CLUSTER_OVERRIDES'][name],role)
        zero_minutes = roster.loc[list(expected)].reset_index().assign(PROJECTED_MIN=0.)
        rebuilt = apply_roster_rules(zero_minutes)
        self.assertTrue(rebuilt.PROJECTED_MIN.eq(0.).all())
        for side in ['Off','Def']:
            self.assertTrue(rebuilt[side+'Archetype'].notna().all())
            self.assertTrue(rebuilt[side+'AssignmentSource'].eq('manual_override').all())

    def test_role_assignment_cannot_revert_an_approved_actual_stat_baseline(self):
        # Niang has an individual historical profile, but the user already
        # switched his estimated baseline to actual rates before assigning roles.
        roster = pd.read_csv(ROOT/'preseason_archetypes.csv')
        roster = roster[roster.PLAYER_NAME.eq('Georges Niang')]
        player = roster.iloc[0]
        teams = pd.DataFrame([dict(TEAM_ID=player.TEAM_ID,GAMES_PLAYED=2)])
        history = pd.DataFrame([dict(PLAYER_ID=player.PLAYER_ID,MIN=20.,**{s:10. for s in STAT_NAMES})])
        current = pd.DataFrame([dict(PLAYER_NAME=player.PLAYER_NAME,GP=2.,MIN=10.,
                                     ESPN_PLAYER_ID='test',**{s:1. for s in STAT_NAMES})])
        def project(roster=roster):
            return build_projections(roster,history,teams,current,
                                     approved_minutes={player.PLAYER_ID:20.}).iloc[0]
        row = project()
        self.assertEqual(row['Stat baseline'],'Current season')
        self.assertEqual(row.PTS,2.)
        self.assertEqual(row.PRA,6.)
        self.assertEqual(row['Offensive archetype'],'Catch & Shoot Wing')
        self.assertEqual(row['Defensive archetype'],'Non-POA Wing')
        self.assertEqual(project(roster.assign(SEASON_ID=22027)).PTS,10.)
        current.loc[0,'BLK'] = float('nan')
        row = project()
        self.assertEqual(row.PTS,2.)
        self.assertEqual(row.BLK,10.)
        self.assertEqual(row['Stat baseline'],'Current season + historical fallback')
        self.assertEqual(row['Estimate fallback stats'],'BLK')
        current.loc[0,'GP'] = 0.
        self.assertEqual(project().PTS,10.)


if __name__ == '__main__':
    unittest.main()
