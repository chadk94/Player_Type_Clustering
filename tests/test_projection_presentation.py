import unittest
from projection_presentation import over_line_label, matchup_rating


class ProjectionPresentationTests(unittest.TestCase):
    def test_line_price_signed_and_numeric_inputs_unchanged(self):
        self.assertEqual(over_line_label(1.5,-150),'1.5 -150')
        self.assertEqual(over_line_label(3.,120),'3 +120')
        self.assertEqual(over_line_label(float('nan'),-110),'—')
        self.assertEqual(over_line_label(2.5,float('nan')),'—')

    def test_rating_boundaries_average_and_missing_data(self):
        for projected,rating in [(9,1),(9.5,2),(9.7,3),(10,3),(10.3,3),(10.5,4),(11,5)]:
            self.assertTrue(matchup_rating(10,projected,'PTS').startswith(str(rating)))
        self.assertEqual(matchup_rating(10,11,'PTS',False),'—')
        self.assertEqual(matchup_rating(0,0,'PTS'),'—')
        self.assertEqual(matchup_rating(10,float('nan'),'PTS'),'—')

    def test_turnovers_invert_and_minutes_scaling_does_not_change_rating(self):
        for stat in ['TOV','PF']:
            self.assertEqual(matchup_rating(2,1.8,stat),'5 ⬆️⬆️')
            self.assertEqual(matchup_rating(2,2.2,stat),'1 ⬇️⬇️')
        self.assertEqual(matchup_rating(10,10.5,'PTS'),matchup_rating(20,21,'PTS'))
