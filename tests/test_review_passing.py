from datetime import date
from unittest import TestCase
from unittest.mock import Mock, patch

import pandas as pd

from edge_review import role_evidence
from edge_review_sources import archived_review_passing, review_passing
from test_edge_review import NOW, logs, raw_team


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
        self.assertNotIn('AST',partial['ratios'])
        self.assertIn('4 of 5 recent games',partial['passing_detail'])
        self.assertIn('pending',partial['passing_detail'])

    def test_bundled_public_checkpoint_is_valid_and_has_boozer_games(self):
        archive=archived_review_passing('2026-27')
        self.assertEqual(len(archive),4500)
        self.assertEqual(len(archive[archive.PLAYER_ID.eq(1643409)]),14)
        self.assertFalse(archive.duplicated(['PLAYER_ID','GAME_ID']).any())
        self.assertTrue(archived_review_passing('2025-26').empty)
