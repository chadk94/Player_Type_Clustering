import unittest
import numpy as np
import pandas as pd
from team_rate_blend import season_rates, blend_team_rates


def logs(games, ast=25., attempts=35.):
    rows=[]
    for game, (one,two) in enumerate(games):
        for team, opponent in [(one,two),(two,one)]:
            rows.append(dict(GAME_ID=str(game),Team=team,Opponent=opponent,MIN=240.,
                             AST=ast,FGM=40.,FGA=90.,FG3M=14.,FG3A=attempts,
                             FTM=16.,FTA=20.,PTS=110.,OREB=10.,DREB=30.,REB=40.))
    return pd.DataFrame(rows)


class TeamRateBlendTests(unittest.TestCase):
    def setUp(self):
        self.prior=season_rates(logs([('A','B'),('A','C'),('B','C')]),2)
        self.current=logs([('A','B'),('A','C')],ast=30.,attempts=45.)
        self.teams=pd.DataFrame({'TEAM_ABBREVIATION':['A','B','C'],'GAMES_PLAYED':[2,1,1]})

    def test_each_team_has_its_own_ten_game_prior(self):
        result=blend_team_rates(self.prior,self.current,self.teams)
        observed=season_rates(self.current)
        for team,games in [('A',2),('B',1),('C',1)]:
            self.assertAlmostEqual(result['coverage'].loc[team,'Current weight'],games/(10+games))
            for kind,old in self.prior.items():
                np.testing.assert_allclose(result[kind].loc[team],(10*old.loc[team]+games*observed[kind].loc[team])/(10+games))

    def test_weights_grow_continuously_to_fifty_percent_at_ten_games(self):
        for games in [0,1,3,4,10,20,82]:
            current=logs([('A','B')]*games) if games else pd.DataFrame()
            teams=pd.DataFrame({'TEAM_ABBREVIATION':['A','B','C'],'GAMES_PLAYED':[games,games,0]})
            result=blend_team_rates(self.prior,current,teams)
            self.assertAlmostEqual(result['coverage'].loc['A','Current weight'],games/(10+games))
            for kind,old in self.prior.items():
                pd.testing.assert_series_equal(result[kind].loc['C'],old.loc['C'])

    def test_missing_game_falls_back_for_affected_teams_only(self):
        result=blend_team_rates(self.prior,self.current[self.current.GAME_ID.eq('0')],self.teams)
        coverage=result['coverage']
        self.assertEqual(coverage.loc['A','Current weight'],0)
        self.assertEqual(coverage.loc['C','Current weight'],0)
        self.assertAlmostEqual(coverage.loc['B','Current weight'],1/11)
        for kind,old in self.prior.items():
            pd.testing.assert_series_equal(result[kind].loc['A'],old.loc['A'])
        self.assertIn('incomplete',coverage.loc['A','Status'])

    def test_unknown_records_and_invalid_box_scores_cannot_be_current_rates(self):
        for frame,teams in [(self.current,pd.DataFrame()),
                            (self.current.assign(FG3A=100.),self.teams),
                            (self.current.iloc[:-1],self.teams)]:
            result=blend_team_rates(self.prior,frame,teams)
            self.assertTrue(result['coverage']['Current weight'].eq(0).all())
            for kind,old in self.prior.items():pd.testing.assert_frame_equal(result[kind],old)

    def test_player_appearances_are_not_game_counts(self):
        # Split each team box into two appearances: rates and weights unchanged.
        split=self.current.copy()
        numeric=split.select_dtypes('number').columns
        split[numeric]/=2
        split=pd.concat([split,split],ignore_index=True)
        original=blend_team_rates(self.prior,self.current,self.teams)
        result=blend_team_rates(self.prior,split,self.teams)
        for key in original:pd.testing.assert_frame_equal(original[key],result[key])


if __name__=='__main__':unittest.main()
