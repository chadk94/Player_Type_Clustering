import copy
import unittest
from matchup_data import parse_boxscore, validate_team_boxscores


def box():
    values={'MIN':'240','PTS':'110','AST':'25','REB':'40','OREB':'10','DREB':'30',
            'STL':'4','BLK':'4','TO':'10','PF':'15','FG':'40-90','3PT':'14-35','FT':'16-20'}
    payload={'header':{'id':'g','season':{'year':2027,'type':2},'competitions':[{
        'date':'2026-10-27T00:00Z','status':{'type':{'completed':True}},
        'competitors':[{'team':{'id':id,'abbreviation':team},'score':'110'} for id,team in [('1','ATL'),('2','BOS')]]}]},
        'boxscore':{'players':[], 'teams':[]}}
    for id,team in [('1','ATL'),('2','BOS')]:
        payload['boxscore']['players'].append({'team':{'id':id},'statistics':[{'labels':list(values),
            'athletes':[{'athlete':{'id':id,'displayName':team},'stats':list(values.values())}]}]})
        stats={'fieldGoalsMade-fieldGoalsAttempted':'40-90','threePointFieldGoalsMade-threePointFieldGoalsAttempted':'14-35',
               'freeThrowsMade-freeThrowsAttempted':'16-20','assists':'25','offensiveRebounds':'10','defensiveRebounds':'30','totalRebounds':'40'}
        payload['boxscore']['teams'].append({'team':{'abbreviation':team},'statistics':[
            {'name':k,'displayValue':v} for k,v in stats.items()]})
    return payload


class TeamBoxCoverageTests(unittest.TestCase):
    def test_rounded_zero_minutes_preserve_real_shots(self):
        payload=box()
        category=payload['boxscore']['players'][0]['statistics'][0]
        shot={k:'0' for k in category['labels']}
        shot.update(FG='0-1',FT='0-0',**{'3PT':'0-1'})
        category['athletes'].append({'athlete':{'id':'zero','displayName':'Brief appearance'},'stats':list(shot.values())})
        stats=payload['boxscore']['teams'][0]['statistics']
        stats[0]['displayValue']='40-91';stats[1]['displayValue']='14-36'
        rows=parse_boxscore(payload,2027)
        self.assertEqual(rows[1]['MIN'],0.)
        self.assertEqual(rows[1]['FG3A'],1.)
        validate_team_boxscores(payload,rows)
        with self.assertRaises(ValueError):validate_team_boxscores(payload,[rows[0],rows[2]])

    def test_missing_category_or_duplicate_appearance_rejected(self):
        payload=box();rows=parse_boxscore(payload,2027)
        validate_team_boxscores(payload,rows)
        changed=copy.deepcopy(rows);changed[0]['AST']=24.
        with self.assertRaises(ValueError):validate_team_boxscores(payload,changed)
        with self.assertRaises(ValueError):validate_team_boxscores(payload,rows+[rows[0]])


if __name__=='__main__':unittest.main()
