import unittest
from unittest.mock import patch

import pandas as pd
import projection_dashboard as dashboard


class MatchupCacheTests(unittest.TestCase):
    def test_cache_reuses_public_inputs_and_invalidates_changed_stats_roles_or_rules(self):
        dashboard.current_matchup_effects.clear()
        self.addCleanup(dashboard.current_matchup_effects.clear)
        logs = pd.DataFrame({'PTS': [10.]})
        roster = pd.DataFrame({'OffCluster': [1.]})
        with patch.object(dashboard, 'assign_game_archetypes', side_effect=lambda l,r:l), \
             patch.object(dashboard, 'matchup_effects', side_effect=lambda l:{'points':float(l.PTS.sum())}) as calculate:
            first = dashboard.current_matchup_effects(logs, roster, 'rules-v1')
            first['points'] = 999.  # Cached return values must be independent copies.
            self.assertEqual(dashboard.current_matchup_effects(logs, roster, 'rules-v1')['points'], 10.)
            self.assertEqual(calculate.call_count, 1)
            self.assertEqual(dashboard.current_matchup_effects(logs*2, roster, 'rules-v1')['points'], 20.)
            dashboard.current_matchup_effects(logs, roster*2, 'rules-v1')
            dashboard.current_matchup_effects(logs, roster, 'rules-v2')
            self.assertEqual(calculate.call_count, 4)
