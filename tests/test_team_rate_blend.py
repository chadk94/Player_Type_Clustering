import unittest
import numpy as np
import pandas as pd
from team_rate_blend import (season_rates, blend_team_rates, defensive_prior_games,
                             defensive_current_weight, DEFENSIVE_RATE_COLUMNS)


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
        self.current[['OREB','DREB']] = [14.,26.]
        self.teams=pd.DataFrame({'TEAM_ABBREVIATION':['A','B','C'],'GAMES_PLAYED':[2,1,1]})

    def test_each_team_uses_its_own_count_and_only_three_rates_get_five(self):
        result=blend_team_rates(self.prior,self.current,self.teams)
        observed=season_rates(self.current)
        selected = {('assists','Allowed assist rate'),('threes','Allowed three attempt rate'),
                    ('rebounds_allowed','Offensive rebound share')}
        self.assertEqual({(kind,col) for kind,cols in DEFENSIVE_RATE_COLUMNS.items() for col in cols},selected)
        for team,games in [('A',2),('B',1),('C',1)]:
            self.assertAlmostEqual(result['coverage'].loc[team,'Current weight'],games/(10+games))
            self.assertAlmostEqual(result['coverage'].loc[team,'Defensive current weight'],games/(5+games))
            for kind,old in self.prior.items():
                for col in old.columns:
                    prior = 5 if (kind,col) in selected else 10
                    self.assertAlmostEqual(result[kind].loc[team,col],
                        (prior*old.loc[team,col]+games*observed[kind].loc[team,col])/(prior+games))

    def test_other_weights_keep_ten_game_prior(self):
        for games in [0,1,3,4,10,20,82]:
            current=logs([('A','B')]*games) if games else pd.DataFrame()
            teams=pd.DataFrame({'TEAM_ABBREVIATION':['A','B','C'],'GAMES_PLAYED':[games,games,0]})
            result=blend_team_rates(self.prior,current,teams)
            self.assertAlmostEqual(result['coverage'].loc['A','Current weight'],games/(10+games))
            for kind,old in self.prior.items():
                pd.testing.assert_series_equal(result[kind].loc['C'],old.loc['C'])

    def test_preview_preserved_then_prior_tapers_to_zero_at_twenty(self):
        for games in range(9):
            self.assertEqual(defensive_prior_games(games),5.)
            self.assertEqual(defensive_current_weight(games),games/(games+5.))
        self.assertAlmostEqual(defensive_prior_games(10),25/6)
        self.assertAlmostEqual(defensive_current_weight(10),12/17)
        self.assertAlmostEqual(defensive_prior_games(15),25/12)
        self.assertAlmostEqual(defensive_current_weight(15),36/41)
        weights = [defensive_current_weight(g) for g in range(83)]
        self.assertTrue(all(0<=w<=1 for w in weights))
        self.assertTrue(all(a<=b for a,b in zip(weights,weights[1:])))
        self.assertLess(weights[19],1.)
        for games in [20,21,50,82]:
            self.assertEqual(defensive_prior_games(games),0.)
            self.assertEqual(defensive_current_weight(games),1.)
        for invalid in [-1,np.nan,np.inf]:
            with self.assertRaises(ValueError):defensive_prior_games(invalid)

    def test_twenty_games_removes_history_only_from_selected_defensive_rates(self):
        for games in [8,10,15,19,20,21,82]:
            current = logs([('A','B')]*games,ast=30.,attempts=45.)
            current[['OREB','DREB']] = [14.,26.]
            teams = self.teams.assign(GAMES_PLAYED=[games,games,0])
            result = blend_team_rates(self.prior,current,teams)
            observed = season_rates(current)
            for kind,cols in DEFENSIVE_RATE_COLUMNS.items():
                for col in cols:
                    if games>=20:
                        self.assertEqual(result[kind].loc['A',col],observed[kind].loc['A',col])
                    else:
                        self.assertNotEqual(result[kind].loc['A',col],observed[kind].loc['A',col])
            self.assertAlmostEqual(result['assists'].loc['A','Assist rate'],
                (10*self.prior['assists'].loc['A','Assist rate']+games*.75)/(10+games))
            self.assertEqual(result['coverage'].loc['A','Defensive status'],
                             'Current season only' if games>=20 else 'Blended')
            if games>=20:
                changed_prior = {kind:frame.copy(deep=True) for kind,frame in self.prior.items()}
                for kind,cols in DEFENSIVE_RATE_COLUMNS.items():
                    for col in cols:changed_prior[kind][col] = np.nan
                current_only = blend_team_rates(changed_prior,current,teams)
                for kind,cols in DEFENSIVE_RATE_COLUMNS.items():
                    for col in cols:
                        self.assertEqual(current_only[kind].loc['A',col],observed[kind].loc['A',col])

    def test_missing_twentieth_game_never_masquerades_as_complete_current_data(self):
        current = logs([('A','B')]*19,ast=30.,attempts=45.)
        teams = self.teams.assign(GAMES_PLAYED=[20,20,0])
        result = blend_team_rates(self.prior,current,teams)
        self.assertTrue(result['coverage']['Defensive current weight'].eq(0.).all())
        self.assertIn('incomplete',result['coverage'].loc['A','Defensive status'])
        for kind,old in self.prior.items():
            pd.testing.assert_frame_equal(result[kind],old,check_exact=True)

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
