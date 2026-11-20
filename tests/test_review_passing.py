from datetime import date
from unittest import TestCase
from unittest.mock import Mock, patch

import pandas as pd

from edge_review import assess_edges, role_evidence, passing_role_evidence
from edge_review_sources import archived_review_passing, review_passing
from test_edge_review import NOW, GAME, logs, raw_team, finalized, edge, reference, healthy


class ReviewPassingTests(TestCase):
    def test_checkpoint_survives_cold_start_and_only_missing_dates_are_requested(self):
        games=pd.DataFrame(dict(ESPN_PLAYER_ID=['11','11'],GAME_ID=['old','new'],
            GAME_DATE=['2026-11-15T01:00Z','2026-11-17T01:00Z'],MIN=[30.,31.],AST=[3.,4.]))
        archive=pd.DataFrame([dict(PLAYER_ID=1,GAME_ID='old',GAME_DATE='2026-11-14',GP=1,
            MIN=30.,AST=3.,POTENTIAL_AST=8.,PASSES_MADE=45.)])
        fresh=pd.DataFrame([dict(GP=1,MIN=31.,AST=4.,POTENTIAL_AST=10.,PASSES_MADE=52.)],index=[1])
        read=Mock(return_value={date(2026,11,16):fresh})
        with patch('edge_review_sources.archived_review_passing',return_value=archive):
            result=review_passing('2026-27',games,{1:pd.Series(dict(ESPN_PLAYER_ID='11'))},read)
        read.assert_called_once_with('2026-27',{date(2026,11,16)})
        self.assertEqual(result.GAME_ID.tolist(),['old','new'])
        self.assertEqual(result.POTENTIAL_AST.tolist(),[8.,10.])

    def test_archive_must_still_match_current_box_and_exact_game(self):
        games=pd.DataFrame([dict(ESPN_PLAYER_ID='11',GAME_ID='a',GAME_DATE='2026-11-17T01:00Z',MIN=30.,AST=3.)])
        for change in [dict(AST=4),dict(MIN=32),dict(GAME_ID='b'),dict(GAME_DATE='2026-11-15')]:
            archived=dict(PLAYER_ID=1,GAME_ID='a',GAME_DATE='2026-11-16',GP=1,MIN=30.,AST=3.,POTENTIAL_AST=8.,PASSES_MADE=45.)
            archived.update(change)
            read=Mock(return_value={})
            with patch('edge_review_sources.archived_review_passing',return_value=pd.DataFrame([archived])):
                result=review_passing('2026-27',games,{1:pd.Series(dict(ESPN_PLAYER_ID='11'))},read)
            self.assertTrue(result.empty)
            read.assert_called_once_with('2026-27',{date(2026,11,16)})

    def test_assist_check_uses_opportunities_even_without_shooting_and_reports_missing_games(self):
        games=logs().drop(columns=['FGA','FGM','FG3A','FG3M','FTA','FTM'])
        passing=pd.DataFrame(dict(PLAYER_ID=1,GAME_ID=games.GAME_ID,
                                 POTENTIAL_AST=[8]*5+[4]*5,PASSES_MADE=[40]*10))
        args=(raw_team().iloc[0],pd.Series(dict(ESPN_PLAYER_ID='11',GP=10)))
        evidence=role_evidence(*args,games,NOW,passing)
        self.assertAlmostEqual(evidence['ratios']['AST'],(2/3+1)/2)
        self.assertAlmostEqual(evidence['trends']['AST'],-.25)
        self.assertIn('Potential Assists per 36 minutes',evidence['passing_detail'])
        self.assertNotIn('PTS',evidence['ratios'])
        # Actual conversion noise does not change the passing trend.
        self.assertEqual(evidence,role_evidence(*args,games.assign(AST=30),NOW,passing))
        partial=role_evidence(*args,games,NOW,passing.iloc[:-1])
        self.assertAlmostEqual(partial['ratios']['AST'],(9/14+1)/2)
        self.assertAlmostEqual(partial['trends']['AST'],-.25)
        self.assertIn('4 of 5 recent games',partial['passing_detail'])
        self.assertNotIn('pending',partial['passing_detail'])

    def test_up_to_three_missing_games_use_only_verified_counts_and_minutes(self):
        games=pd.DataFrame(dict(GAME_ID=[str(i) for i in range(15)],MIN=[20.]*10+[40.]*5))
        passing=pd.DataFrame(dict(PLAYER_ID=1,GAME_ID=games.GAME_ID,
                                 POTENTIAL_AST=[4.]*10+[16.]*5,PASSES_MADE=[20.]*10+[80.]*5))
        original=games.copy(deep=True)
        for missing in ([14],[0,14],[0,13,14]):
            with self.subTest(missing=missing):
                available=passing.drop(missing)
                result=passing_role_evidence(games,1,5,available)
                verified=games.merge(available,on='GAME_ID')
                season_rate=verified.POTENTIAL_AST.sum()/verified.MIN.sum()
                self.assertAlmostEqual(result['ratios']['AST'],.4/season_rate)
                self.assertAlmostEqual(result['trends']['AST'],1.)
                self.assertNotIn('pending',result['passing_detail'])
        pd.testing.assert_frame_equal(games,original)
        # Four missing records still require review even with enough games left.
        result=passing_role_evidence(games,1,5,passing.drop([0,1,2,14]))
        self.assertNotIn('AST',result['ratios'])
        self.assertIn('4 games are missing',result['passing_detail'])

    def test_small_gaps_still_require_enough_verified_games_and_minutes(self):
        games=logs()
        passing=pd.DataFrame(dict(PLAYER_ID=1,GAME_ID=games.GAME_ID,POTENTIAL_AST=4.,PASSES_MADE=40.))
        for missing,minutes in [([7,8,9],30.),([0],30.),([9],10.)]:
            with self.subTest(missing=missing,minutes=minutes):
                result=passing_role_evidence(games.assign(MIN=minutes),1,5,passing.drop(missing))
                self.assertNotIn('AST',result['ratios'])
                self.assertIn('three verified recent and five verified preceding',result['passing_detail'])

    def test_invalid_tracking_is_excluded_and_real_zeroes_stay_verified(self):
        games=logs()
        passing=pd.DataFrame(dict(PLAYER_ID=1,GAME_ID=games.GAME_ID,POTENTIAL_AST=4.,PASSES_MADE=40.))
        passing.loc[9,'POTENTIAL_AST']=float('nan')
        partial=passing_role_evidence(games,1,5,passing)
        self.assertEqual(partial['ratios']['AST'],1.)
        self.assertIn('4 of 5 recent games',partial['passing_detail'])
        passing.loc[9,['POTENTIAL_AST','PASSES_MADE']]=0.
        zero=passing_role_evidence(games,1,5,passing)
        self.assertAlmostEqual(zero['ratios']['AST'],8/9)
        self.assertIn('5 of 5 recent games',zero['passing_detail'])
        duplicate=passing_role_evidence(games,1,5,pd.concat([passing,passing.iloc[:1]]))
        self.assertNotIn('AST',duplicate['ratios'])
        self.assertIn('Duplicate',duplicate['passing_detail'])

    def test_partial_coverage_completes_assist_and_pra_checks_without_changing_projections(self):
        games=logs()
        passing=pd.DataFrame(dict(PLAYER_ID=1,GAME_ID=games.GAME_ID,POTENTIAL_AST=4.,PASSES_MADE=40.)).iloc[:-1].copy()
        team=finalized(raw_team())
        original=team.copy(deep=True)
        matches={1:pd.Series(dict(ESPN_PLAYER_ID='11',GP=10))}
        for stat in ('AST','PRA'):
            with self.subTest(stat=stat):
                mean=team.iloc[0][stat]
                plays=edge(mean,mean/1.5,stat=stat)
                result=assess_edges(plays,team,GAME,games,matches,reference(),healthy(),NOW,NOW,passing=passing)
                self.assertEqual(result.iloc[0]['Role flag'],'Clear')
                self.assertTrue(result.iloc[0]['Robust play'])
                pd.testing.assert_frame_equal(result[plays.columns],plays)
        pd.testing.assert_frame_equal(team,original)
        # Tolerating a gap must not suppress a real adverse passing trend.
        passing.loc[passing.index<5,['POTENTIAL_AST','PASSES_MADE']]*=2
        mean=team.iloc[0].AST
        result=assess_edges(edge(mean,mean/1.5,stat='AST'),team,GAME,games,matches,
                            reference(),healthy(),NOW,NOW,passing=passing)
        self.assertEqual(result.iloc[0]['Role flag'],'ROLE')

    def test_bundled_public_checkpoint_is_valid_and_has_boozer_games(self):
        archive=archived_review_passing('2026-27')
        self.assertEqual(len(archive),4500)
        self.assertEqual(len(archive[archive.PLAYER_ID.eq(1643409)]),14)
        self.assertFalse(archive.duplicated(['PLAYER_ID','GAME_ID']).any())
        self.assertTrue(archived_review_passing('2025-26').empty)
