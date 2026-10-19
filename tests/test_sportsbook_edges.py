from datetime import datetime, timezone, timedelta
import unittest
import pandas as pd
from sportsbook_edges import game_edges, scheduled_book_games
from sportsbook_lines import OddsFeedError
from import_minutes import TEAM_IDS


class GameEdgeTests(unittest.TestCase):
    def setUp(self):
        self.now=datetime(2026,10,19,19,tzinfo=timezone.utc)
        self.event={'id':'game1','Home':'NYK','Away':'PHI','home_team':'New York Knicks',
                    'away_team':'Philadelphia 76ers','commence_time':(self.now+timedelta(days=1)).isoformat()}
        self.payload={**self.event,'bookmakers':[{'key':'fanduel','markets':[
            {'key':'player_threes','last_update':self.now.isoformat(),'outcomes':[
                {'description':f'Player {i}','name':side,'point':2.5,'price':-110}
                for i in range(25) for side in ['Over','Under']]}]}]}
        self.players=pd.DataFrame([{'PLAYER_ID':i,'Player':f'Player {i}','Team':'NYK' if i%2 else 'PHI',
            'Opponent':'PHI' if i%2 else 'NYK','Projected minutes':30.,'FG3M':0.2+i*.3,
            'Status':'Available','Matchup status':'Available','Off samples':50,'Def samples':60,
            'Stat baseline':'2025–26','Matchup data':'2025–26'} for i in range(25)])

    def test_top_twenty_absolute_edge_both_teams_no_duplicates(self):
        ranked,total=game_edges(pd.concat([self.players,self.players]),self.payload,self.event,'fanduel',now=self.now)
        self.assertEqual(total,24)  # One player is within the 5% buffer.
        self.assertEqual(len(ranked),20)
        self.assertEqual(ranked.Rank.tolist(),list(range(1,21)))
        self.assertTrue(ranked['Edge %'].abs().is_monotonic_decreasing)
        self.assertTrue(ranked['Edge %'].abs().gt(5).all())
        self.assertEqual(set(ranked.Team),{'NYK','PHI'})
        self.assertEqual(set(ranked.Play),{'Over','Under'})
        self.assertEqual(set(ranked.Stat),{'3PM'})

    def test_matchup_ratings_use_opponent_lift_not_sportsbook_edge(self):
        self.players['Baseline FG3M']=self.players.FG3M
        ranked,_=game_edges(self.players,self.payload,self.event,'fanduel',now=self.now)
        self.assertEqual(set(ranked['Matchup rating']),{'3 ↔️'})
        self.assertTrue(ranked['Edge %'].abs().gt(5).all())
        self.players['Baseline FG3M']=self.players.FG3M/1.15
        ranked,_=game_edges(self.players,self.payload,self.event,'fanduel',now=self.now)
        self.assertEqual(set(ranked['Matchup rating']),{'5 ⬆️⬆️'})

    def test_points_market_adjustment_does_not_change_matchup_rating(self):
        import copy
        points=copy.deepcopy(self.payload['bookmakers'][0]['markets'][0])
        points['key']='player_points'
        self.payload['bookmakers'][0]['markets'].append(points)
        self.players['PTS']=20.
        self.players['Baseline PTS']=10.
        self.players['Pre-market PTS']=10.
        ranked,_=game_edges(self.players,self.payload,self.event,'fanduel',now=self.now,stats=['PTS'])
        self.assertTrue(ranked.Projected.eq(20.).all())
        self.assertEqual(set(ranked['Matchup rating']),{'3 ↔️'})

    def test_direction_and_optional_subthreshold_rows(self):
        under,_=game_edges(self.players,self.payload,self.event,'fanduel',now=self.now,direction='Under')
        self.assertTrue(under['Edge %'].lt(-5).all())
        all_edges,total=game_edges(self.players,self.payload,self.event,'fanduel',now=self.now,plays_only=False)
        self.assertEqual(total,25)

    def test_stat_selection_filters_before_top_twenty_and_download_rows(self):
        import copy
        points=copy.deepcopy(self.payload['bookmakers'][0]['markets'][0])
        points['key']='player_points'
        self.payload['bookmakers'][0]['markets'].append(points)
        self.players['PTS']=50.
        ranked,total=game_edges(self.players,self.payload,self.event,'fanduel',now=self.now,stats=['3PM'])
        self.assertEqual(total,24)
        self.assertEqual(len(ranked),20)
        self.assertEqual(set(ranked.Stat),{'3PM'})
        empty,count=game_edges(self.players,self.payload,self.event,'fanduel',now=self.now,stats=[])
        self.assertTrue(empty.empty)
        self.assertEqual(count,0)
        with self.assertRaises(ValueError):
            game_edges(self.players,self.payload,self.event,'fanduel',now=self.now,stats=['INVALID'])

    def test_missing_stale_wrong_book_wrong_event_and_ineligible_projections(self):
        empty,count=game_edges(self.players,self.payload,self.event,'draftkings',now=self.now)
        self.assertTrue(empty.empty)
        self.assertEqual(count,0)
        stale,_=game_edges(self.players,self.payload,self.event,'fanduel',now=self.now+timedelta(minutes=16))
        self.assertTrue(stale.empty)
        with self.assertRaises(OddsFeedError):
            game_edges(self.players,{**self.payload,'id':'wrong'},self.event,'fanduel',now=self.now)
        invalid=self.players.copy()
        invalid.loc[:,'Opponent']='BOS'
        bad,_=game_edges(invalid,self.payload,self.event,'fanduel',now=self.now)
        self.assertTrue(bad.empty)
        invalid=self.players.copy()
        invalid.loc[:,'Projected minutes']=0
        bad,_=game_edges(invalid,self.payload,self.event,'fanduel',now=self.now)
        self.assertTrue(bad.empty)
        invalid=self.players.copy()
        invalid.loc[:,'Matchup status']='Insufficient archetype matchup history'
        bad,_=game_edges(invalid,self.payload,self.event,'fanduel',now=self.now)
        self.assertTrue(bad.empty)

    def test_game_selector_requires_exact_future_scheduled_regular_season_event(self):
        schedule=pd.DataFrame([{'TEAM_ID':TEAM_IDS['NYK'],'Opponent':'PHI','Venue':'home',
            'GameTimeUTC':self.event['commence_time'],'GameStatus':'Scheduled'}])
        games=scheduled_book_games([self.event,self.event],schedule,self.now)
        self.assertEqual(len(games),1)
        self.assertEqual(games[0]['id'],'game1')
        self.assertEqual(scheduled_book_games([self.event],schedule.iloc[:0],self.now),[])
        schedule.GameTimeUTC=(self.now+timedelta(days=2)).isoformat()
        self.assertEqual(scheduled_book_games([self.event],schedule,self.now),[])
        schedule.GameTimeUTC=self.event['commence_time']
        schedule.GameStatus='Postponed'
        self.assertEqual(scheduled_book_games([self.event],schedule,self.now),[])
        schedule.GameStatus='Scheduled'
        self.assertEqual(scheduled_book_games([self.event],schedule,self.now+timedelta(days=2)),[])
