import copy
import json
import unittest

import pandas as pd

from matchup_evidence import recent_peer_games, peer_panels
from matchup_projections import matchup_effects, apply_matchup
from edge_rating_table import ROLE_STATS


class MatchupEvidenceTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame([
            dict(ESPN_PLAYER_ID='11', PLAYER_NAME='Peer One', GAME_ID=str(day),
                 GAME_DATE=f'2026-11-{day:02d}', MIN=20., Opponent='MIA',
                 OffCluster=1, DefCluster=2, PTS=10., FG3A=float(day), FG3M=0.,
                 FGA=8., AST=3., OREB=1., DREB=5., STL=0., BLK=1.)
            for day in range(1, 13)])

    def effects(self, logs):
        return {key: dict(performances=rows) for key, rows in recent_peer_games(logs).items()}

    def player(self):
        return dict(PLAYER_NAME='Selected Player', OffCluster=1, DefCluster=2,
                    OffArchetype='Stretch Big', DefArchetype='Rim Protector')

    def test_latest_ten_and_pregame_usual_rate_without_future_leakage(self):
        logs = self.fixture()
        original = logs.copy(deep=True)
        effects = self.effects(logs)
        result = peer_panels('3PM', self.player(), 'MIA', effects, ROLE_STATS['3PM'])
        panel = result['panels'][0]
        self.assertEqual(panel['columns'], ['Player / date', 'MIN', '3PA', '3PM', 'Usual 3PA'])
        self.assertEqual(len(panel['rows']), 10)
        self.assertEqual(panel['rows'][0]['date'], 'Nov 12')
        self.assertEqual(panel['rows'][0]['values'], ['20.0', '12', '0', '6.0'])
        # Game four compares against the first three, not itself or later games.
        fourth = effects[('Off', 1, 'MIA')]['performances'][8]
        self.assertEqual(fourth['usual']['FG3A'], 2.)
        self.assertIsNone(effects[('Off', 1, 'MIA')]['performances'][-1]['usual']['FG3A'])
        json.dumps(result, allow_nan=False)
        pd.testing.assert_frame_equal(logs, original)

    def test_target_player_wrong_roles_opponents_low_minutes_and_missing_stats(self):
        logs = self.fixture()
        latest = logs.iloc[-1].to_dict()
        logs = pd.concat([logs, pd.DataFrame([
            dict(latest, GAME_ID='a', ESPN_PLAYER_ID='12', PLAYER_NAME='Selected Player'),
            dict(latest, GAME_ID='b', ESPN_PLAYER_ID='13', PLAYER_NAME='Wrong Team', Opponent='BOS'),
            dict(latest, GAME_ID='c', ESPN_PLAYER_ID='14', PLAYER_NAME='Wrong Role', OffCluster=3),
            dict(latest, GAME_ID='d', ESPN_PLAYER_ID='15', PLAYER_NAME='Brief Cameo', MIN=4.),
            dict(latest, GAME_ID='e', ESPN_PLAYER_ID='16', PLAYER_NAME='Missing Attempts', FG3A=float('nan')),
        ])], ignore_index=True)
        panels = peer_panels('3PM', self.player(), 'MIA', self.effects(logs), ROLE_STATS['3PM'])['panels']
        self.assertEqual({r['player'] for r in panels[0]['rows']}, {'Peer One'})
        prior = peer_panels('3PM', self.player(), 'MIA', self.effects(logs), ROLE_STATS['3PM'], '2026-11-05T00:00:00Z')
        self.assertEqual([r['date'] for r in prior['panels'][0]['rows']], ['Nov 04','Nov 03','Nov 02','Nov 01'])

    def test_rebounds_keep_populations_separate_and_missing_values_are_not_zero(self):
        logs = self.fixture()
        logs.loc[11, ['FG3M', 'DREB']] = float('nan')
        latest = logs.iloc[-1].to_dict()
        logs = pd.concat([logs, pd.DataFrame([dict(latest, GAME_ID='other', ESPN_PLAYER_ID='12',
                          PLAYER_NAME='Other Defender', OffCluster=4, DREB=9.)])], ignore_index=True)
        effects = self.effects(logs)
        panels = peer_panels('REB', self.player(), 'MIA', effects, ROLE_STATS['REB'])['panels']
        self.assertEqual(panels[0]['columns'][-2:], ['OREB', 'Usual OREB'])
        self.assertEqual(panels[1]['columns'][-2:], ['DREB', 'Usual DREB'])
        self.assertNotIn('Other Defender', [r['player'] for r in panels[0]['rows']])
        self.assertIn('Other Defender', [r['player'] for r in panels[1]['rows']])
        threes = peer_panels('3PM', self.player(), 'MIA', effects, ROLE_STATS['3PM'])
        self.assertEqual(threes['panels'][0]['rows'][0]['values'][2], '—')
        self.assertFalse(peer_panels('BLK', self.player(), 'NYK', effects, ROLE_STATS['BLK'])['panels'][0]['rows'])

    def test_display_metadata_leaves_matchup_projections_identical(self):
        logs = self.fixture()
        effects = matchup_effects(logs, include_recent=True)
        unchanged = copy.deepcopy(effects)
        for key, games in recent_peer_games(logs).items():
            effects[key]['performances'] = games
        roster = pd.DataFrame([dict(self.player(), PLAYER_ID=1)])
        base = pd.DataFrame([dict(PLAYER_ID=1, Opponent='MIA', PTS=20., REB=8., AST=3.,
                                 OREB=2., DREB=6., FG3M=2., FG3A=6., FGM=7., FGA=15.,
                                 FTM=4., FTA=5., STL=1., BLK=1., TOV=2., PF=2.)])
        teams = pd.DataFrame([dict(TEAM_ABBREVIATION='MIA', GAMES_PLAYED=12)])
        before = apply_matchup(base, roster, teams, {}, unchanged)
        after = apply_matchup(base, roster, teams, {}, effects)
        pd.testing.assert_frame_equal(before, after, check_exact=True)

    def test_eastern_dates_duplicates_and_empty_history(self):
        logs = self.fixture().iloc[:4].copy()
        logs.loc[3, 'GAME_DATE'] = '2026-11-05T01:00:00Z'
        panels = peer_panels('3PM', self.player(), 'MIA', self.effects(logs), ROLE_STATS['3PM'])
        self.assertEqual(panels['panels'][0]['rows'][0]['date'], 'Nov 04')
        duplicate = pd.concat([logs, logs.iloc[-1:]], ignore_index=True)
        panels = peer_panels('3PM', self.player(), 'MIA', self.effects(duplicate), ROLE_STATS['3PM'])
        self.assertEqual(len(panels['panels'][0]['rows']), 3)
        self.assertEqual(recent_peer_games(logs.assign(MIN=0)), {})
        self.assertEqual(recent_peer_games(pd.DataFrame()), {})


if __name__ == '__main__':
    unittest.main()
