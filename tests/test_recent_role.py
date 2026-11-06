import unittest

import numpy as np
import pandas as pd

from recent_role import recent_role_comparison, add_recent_columns, LOG_STATS, DISPLAY_COLUMNS


class RecentRoleTests(unittest.TestCase):
    def setUp(self):
        self.roster = pd.DataFrame({'PLAYER_ID':[1,2], 'PLAYER_NAME':['Rising Player','Stable Player']})
        rows = []
        for player in self.roster.itertuples():
            for i in range(20):
                scale = 2 if player.PLAYER_ID == 1 and i >= 15 else 1
                rows.append(dict(ESPN_PLAYER_ID=str(player.PLAYER_ID), PLAYER_NAME=player.PLAYER_NAME,
                    GAME_ID=str(i), GAME_DATE=(pd.Timestamp('2026-10-01',tz='UTC')+pd.Timedelta(days=i)).isoformat(),
                    MIN=30., FGA=10.*scale, FGM=6.*scale, FG3A=4.*scale,
                    FG3M=1.*scale, FTA=2., FTM=1., AST=3.*scale))
        self.logs = pd.DataFrame(rows)
        self.current = self.logs.groupby(['ESPN_PLAYER_ID','PLAYER_NAME'])[LOG_STATS].mean().reset_index()
        self.current['GP'] = 20
        self.final = pd.DataFrame([dict(PLAYER_ID=p.PLAYER_ID, Player=p.PLAYER_NAME, Team='BOS',
            **{'Season appearances':20., 'Projected minutes':30., 'Expected 3P%':.4, 'Career 3P weight':.7},
            PTS=20., AST=5., FG3A=4., FG3M=1.6, REB=6., PRA=31., STL=1., BLK=.5,
            **{'PTS %':10. if p.PLAYER_ID==1 else -5., 'Edge %':25., 'Play':'Over'}) for p in self.roster.itertuples()])

    def run_comparison(self, **kwargs):
        return recent_role_comparison(kwargs.get('final',self.final), self.roster,
            kwargs.get('current',self.current), kwargs.get('logs',self.logs), '2026-27', as_of='2026-10-25T00:00:00Z')

    def test_normalized_full_roster_matches_direct_weighted_allocation(self):
        original = self.final.copy(deep=True)
        recent = self.run_comparison()
        w = .5**(np.arange(19,-1,-1)/10)
        sample = self.logs[self.logs.ESPN_PLAYER_ID.eq('1')]
        weighted = sample[LOG_STATS].mul(w,axis=0).sum()/np.dot(w,sample.MIN)
        base = sample[LOG_STATS].sum()/sample.MIN.sum()
        pct2 = (base.FGM-base.FG3M)/(base.FGA-base.FG3A)
        pctft = base.FTM/base.FTA
        def opportunity(r): return 2*(r.FGA-r.FG3A)*pct2+3*r.FG3A*.4+r.FTA*pctft
        ratios = [opportunity(weighted)/opportunity(base), weighted.AST/base.AST, weighted.FG3A/base.FG3A]
        for stat, ratio in zip(['PTS','AST','FG3A'], ratios):
            expected = self.final[stat].to_numpy()*[ratio,1]
            expected *= self.final[stat].sum()/expected.sum()
            np.testing.assert_allclose(recent[f'Recent {stat}'],expected)
            self.assertAlmostEqual(recent[f'Recent {stat}'].sum(),self.final[stat].sum())
        np.testing.assert_allclose(recent['Recent FG3M']/recent['Recent FG3A'],.4)
        np.testing.assert_allclose(recent['Recent PRA'],recent['Recent PTS']+recent.REB+recent['Recent AST'])
        self.assertGreater(recent.iloc[0]['Recent PTS'],20)
        self.assertLess(recent.iloc[1]['Recent PTS'],20)
        pd.testing.assert_frame_equal(recent[original.columns],original)
        pd.testing.assert_frame_equal(self.final,original)

    def test_player_below_cutoff_keeps_rate_but_shares_rebalance(self):
        frame = self.final.copy()
        frame.loc[1,'Season appearances'] = 19
        result = self.run_comparison(final=frame)
        self.assertIn('Current rates',result.iloc[1]['Recent role status'])
        self.assertIn('team shares rebalanced',result.iloc[1]['Recent role status'])
        self.assertLess(result.iloc[1]['Recent AST'],frame.iloc[1].AST)
        frame['Season appearances'] = 19
        result = self.run_comparison(final=frame)
        self.assertTrue(result['Recent PTS'].isna().all())
        self.assertTrue(result['Recent role status'].str.startswith('Not active yet').all())

    def test_incomplete_log_of_one_teammate_disables_team_comparison(self):
        result = self.run_comparison(logs=self.logs.drop(index=0))
        self.assertTrue(result['Recent PTS'].isna().all())
        self.assertTrue(result['Recent role status'].str.contains('Rising Player: Incomplete').all())
        pd.testing.assert_frame_equal(result[self.final.columns],self.final)

    def test_duplicate_wrong_season_and_inconsistent_summary_fail_closed(self):
        duplicate = pd.concat([self.logs,self.logs.iloc[[0]]],ignore_index=True)
        old = self.logs.copy();old['GAME_DATE'] = '2025-10-01T00:00:00Z'
        wrong = self.current.copy();wrong['AST'] += 2
        for args in [dict(logs=duplicate),dict(logs=old),dict(current=wrong),dict(logs=pd.DataFrame())]:
            with self.subTest(args=list(args)):
                result = self.run_comparison(**args)
                self.assertTrue(result['Recent PTS'].isna().all())
                self.assertTrue(result['Recent role status'].str.startswith('Unavailable').all())

    def test_future_rows_are_ignored(self):
        future = self.logs.iloc[[0]].copy()
        future['GAME_DATE'] = '2026-10-26T00:00:00Z';future['GAME_ID'] = 'future'
        future[LOG_STATS] *= 3
        pd.testing.assert_frame_equal(self.run_comparison(), self.run_comparison(logs=pd.concat([self.logs,future],ignore_index=True)))

    def test_zero_minute_players_do_not_break_coverage_or_enter_allocation(self):
        inactive = self.final.iloc[[0]].copy()
        inactive['PLAYER_ID'] = 3;inactive['Player'] = 'Out player';inactive['Projected minutes'] = 0
        inactive[['PTS','AST','FG3A','FG3M','REB']] = np.nan
        result = self.run_comparison(final=pd.concat([self.final,inactive],ignore_index=True))
        pd.testing.assert_frame_equal(result.iloc[:2],self.run_comparison())
        self.assertEqual(result.iloc[2]['Recent role status'],'No projected minutes')

    def test_incomplete_active_team_and_invalid_scoring_are_explicit(self):
        missing = self.final.copy();missing.loc[1,'Projected minutes'] = np.nan
        result = self.run_comparison(final=missing)
        self.assertTrue(result['Recent PTS'].isna().all())
        invalid = self.final.copy();invalid['PTS'] = 1.
        result = self.run_comparison(final=invalid)
        self.assertTrue(result['Recent FG3M'].isna().all())
        self.assertTrue(result['Recent 3PM status'].str.contains('exceed projected points').all())
        self.assertTrue(result['Recent PTS'].notna().all())

    def test_presentation_does_not_change_main_projection_edges_order_or_probabilities(self):
        table = pd.DataFrame({'Player':['Stable Player','Rising Player','Rising Player'], 'Team':['BOS']*3,
            'Stat':['PTS','AST','STL'],'Projected':[20.,5.,1.], 'Rank':[1,2,3], 'Edge %':[30.,20.,10.],
            'Play':['Under','Over','Over'],'Exp. P(over)':[.4,np.nan,np.nan]})
        original = table.copy(deep=True)
        result = add_recent_columns(table,self.run_comparison())
        pd.testing.assert_frame_equal(result.drop(columns=DISPLAY_COLUMNS),original)
        pd.testing.assert_frame_equal(table,original)
        self.assertLess(result.iloc[0]['Recent change %'],0)
        self.assertGreater(result.iloc[1]['Recent change %'],0)
        self.assertTrue(pd.isna(result.iloc[2]['Recent role']))
        self.assertEqual(result.iloc[2]['Recent role status'],'Unchanged category')
        single = add_recent_columns(table.iloc[[1]].drop(columns=['Player','Team']),self.run_comparison(),1)
        self.assertEqual(single.iloc[0]['Recent role'],result.iloc[1]['Recent role'])


if __name__ == '__main__':
    unittest.main()
