import unittest
from unittest.mock import patch

import pandas as pd

from roster_membership import add_roster_profiles
from season_pipeline import ROOT, apply_roster_rules, load_rules
from tabs_minutes import parse_tabs_browser_snapshot
from live_projections import match_player_stats


class RosterAdditionTests(unittest.TestCase):
    def setUp(self):
        self.original = pd.read_csv(ROOT/'preseason_archetypes.csv')
        self.roster = add_roster_profiles(self.original, '2026-27')
        self.ids = {1641778, 1641807, 1643633, 1628396}

    def test_additions_preserve_existing_profiles_and_manual_precedence(self):
        pd.testing.assert_frame_equal(self.original, self.roster.iloc[:len(self.original)])
        self.assertEqual(set(self.roster.PLAYER_ID)-set(self.original.PLAYER_ID), self.ids)
        pd.testing.assert_frame_equal(add_roster_profiles(self.roster, '2026-27'), self.roster)
        pd.testing.assert_frame_equal(add_roster_profiles(self.original, '2027-28'), self.original)
        new = self.roster[self.roster.PLAYER_ID.isin(self.ids)]
        self.assertTrue(new.PROJECTED_MIN.eq(0).all())
        rules = load_rules()
        rules['OFF_CLUSTER_OVERRIDES']['Norchad Omier'] = 9
        with patch('season_pipeline.load_rules', return_value=rules):
            changed = apply_roster_rules(self.roster).set_index('PLAYER_ID')
        self.assertEqual(changed.loc[1641807, 'OffArchetype'], 'Stretch Big')
        self.assertEqual(changed.loc[1641807, 'OffAssignmentSource'], 'manual_override')

    def test_all_four_match_tabs_and_current_stat_identities(self):
        new = self.roster[self.roster.PLAYER_ID.isin(self.ids)]
        opponents = {'SAC':'PHX','CLE':'MEM','GSW':'DAL','UTA':'LAL'}
        now = pd.Timestamp('2026-11-18T20:00:00Z')
        games = [dict(Home=team, Away=opp, GameTimeUTC=now+pd.Timedelta(hours=4))
                 for team,opp in opponents.items()]
        snapshot = dict(schema=1, source='TabsNBA', source_url='https://app.tabsnba.com/projections',
                        slate_date='2026-11-18', scenario='Default', source_updated_at=now.isoformat(),
                        observed_at=now.isoformat(), manual_edits=False, scope='All games',
                        rows=[dict(Player=p.PLAYER_NAME, Team=p.TEAM_ABBREVIATION,
                                   Opp=opponents[p.TEAM_ABBREVIATION], minutes=10, status='Projected')
                              for p in new.itertuples()])
        refs,details = parse_tabs_browser_snapshot(snapshot,self.roster,games,now)
        self.assertEqual(set(refs.PLAYER_ID), self.ids)
        self.assertEqual(details['unmatched'], [])
        current = pd.DataFrame([dict(PLAYER_NAME=p.PLAYER_NAME,ESPN_PLAYER_ID=str(i),GP=4,MIN=10)
                                for i,p in enumerate(new.itertuples())])
        self.assertEqual(set(match_player_stats(new,current)),self.ids)


if __name__ == '__main__':
    unittest.main()
