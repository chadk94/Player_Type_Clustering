import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from minutes_blend import minutes_plan, rotation_240
from edge_review import assess_edges
from edge_rating_table import minutes_popup
from test_edge_review import raw_team, finalized, reference, edge, healthy, NOW, GAME


def references(values):
    return pd.DataFrame([dict(PLAYER_ID=pid,Team='BOS',Opponent='DET',game_date='2026-11-17',
        minutes=float(value),source='TabsNBA',updated_at=NOW.isoformat(),independent=True,
        requires_source_timestamp=True,source_updated_at=NOW.isoformat(),source_max_age_hours=2)
        for pid,value in values.items()])


class MinutesBlendTests(unittest.TestCase):
    def test_average_preserves_sources_and_total(self):
        base=raw_team();original=base.copy(deep=True)
        tabs=base.set_index('PLAYER_ID')['Projected minutes'].to_dict();tabs.update({1:24.,2:36.})
        refs=references(tabs);saved=refs.copy(deep=True)
        plans,meta=minutes_plan(base,refs,GAME,NOW)
        self.assertEqual(plans['blended'][1],27.)
        self.assertEqual(plans['blended'][2],33.)
        self.assertEqual(plans['yours'][1],30.)
        self.assertEqual(plans['tabs'][1],24.)
        self.assertAlmostEqual(sum(plans['blended'].values()),240.)
        self.assertTrue(meta['Minutes blend status'].str.startswith('50/50').all())
        pd.testing.assert_frame_equal(base,original);pd.testing.assert_frame_equal(refs,saved)

    def test_tabs_decides_in_out_and_preserves_240_with_caps(self):
        base=raw_team()
        tabs=base.set_index('PLAYER_ID')['Projected minutes'].to_dict();tabs.update({1:0.,9:30.})
        plans,_=minutes_plan(base,references(tabs),GAME,NOW)
        self.assertEqual(plans['blended'][1],0.)
        self.assertEqual(plans['blended'][9],30.)
        self.assertEqual(plans['yours'][1],30.)
        self.assertEqual(plans['yours'][9],0.)
        self.assertAlmostEqual(sum(plans['blended'].values()),240.)
        capped=rotation_240(pd.Series([48.,20.,20.,20.,20.,20.]))
        self.assertEqual(capped.iloc[0],48.)
        self.assertAlmostEqual(capped.sum(),240.)
        self.assertTrue(capped.le(48).all())

    def test_missing_stale_wrong_game_or_incomplete_rotation_never_blends(self):
        base=raw_team();values=base.set_index('PLAYER_ID')['Projected minutes'].to_dict()
        valid=references(values)
        for invalid in [None,valid[valid.PLAYER_ID.ne(1)],valid.assign(Opponent='NYK'),
                        valid.assign(source_updated_at=(NOW-pd.Timedelta(hours=3)).isoformat()),
                        valid.assign(minutes=10.)]:
            plans,meta=minutes_plan(base,invalid,GAME,NOW)
            self.assertFalse(plans['blended'])
            self.assertTrue(meta['Minutes blend status'].str.startswith('Pending').all())
        valid.attrs['incomplete_teams']=['BOS']
        self.assertFalse(minutes_plan(base,valid,GAME,NOW)[0]['blended'])

    def test_endpoint_flags_in_both_directions_and_no_extra_three_minute_stress(self):
        main=finalized(raw_team())
        yours=main.copy();tabs=main.copy()
        signal=dict(ratios={'PTS':1.},trends={'PTS':0.},detail='Stable')
        with patch('edge_review.role_evidence',return_value=signal):
            same=assess_edges(edge(15,14),main,GAME,pd.DataFrame(),{},reference(),healthy(),NOW,NOW,
                              minutes_cases={'yours':yours,'tabs':tabs}).iloc[0]
            self.assertTrue(same['Robust play'])
            self.assertEqual(same['Minutes test minutes'],30.)
            tabs.loc[0,'PTS']=14.
            over=assess_edges(edge(15,14),main,GAME,pd.DataFrame(),{},reference(),healthy(),NOW,NOW,
                              minutes_cases={'yours':yours,'tabs':tabs}).iloc[0]
            self.assertEqual(over['Minutes flag'],'MIN');self.assertFalse(over['Robust play'])
            self.assertEqual(over['Tabs minutes edge %'],0.)
            self.assertIn('TabsNBA minutes:',minutes_popup(over)['detail'])
            self.assertIn('Blended model:',minutes_popup(over)['detail'])
            tabs.loc[0,'PTS']=14.7  # Exactly 5% does not qualify.
            boundary=assess_edges(edge(15,14),main,GAME,pd.DataFrame(),{},reference(),healthy(),NOW,NOW,
                                  minutes_cases={'yours':yours,'tabs':tabs}).iloc[0]
            self.assertEqual(boundary['Minutes flag'],'MIN')
            tabs.loc[0,'PTS']=22.
            under=assess_edges(edge(15,20,'Under'),main,GAME,pd.DataFrame(),{},reference(),healthy(),NOW,NOW,
                               minutes_cases={'yours':yours,'tabs':tabs}).iloc[0]
            self.assertEqual(under['Minutes flag'],'MIN');self.assertLess(under['Tabs minutes edge %'],0.)
            missing=assess_edges(edge(),main,GAME,pd.DataFrame(),{},reference(),healthy(),NOW,NOW,
                                 minutes_cases={}).iloc[0]
            self.assertEqual(missing['Minutes flag'],'Unchecked');self.assertFalse(missing['Robust play'])
        self.assertEqual(main.iloc[0].PTS,15.)

    def test_role_is_combined_with_each_source_and_out_is_not_a_safe_under(self):
        main=finalized(raw_team());yours=main.copy();tabs=main.copy()
        signal=dict(ratios={'PTS':.9},trends={'PTS':-.05},detail='Small adverse shift')
        tabs.loc[0,'Pre-market PTS']*=.9
        tabs['PTS']=120*tabs['Pre-market PTS']/tabs['Pre-market PTS'].sum()
        with patch('edge_review.role_evidence',return_value=signal):
            row=assess_edges(edge(15,12.6),main,GAME,pd.DataFrame(),{},reference(),healthy(),NOW,NOW,
                             minutes_cases={'yours':yours,'tabs':tabs}).iloc[0]
        self.assertGreater(row['Minutes test edge %'],5)
        self.assertGreater(row['Recent test edge %'],5)
        self.assertLess(row['Combined test edge %'],5)
        self.assertFalse(row['Robust play'])
        tabs.loc[0,['Projected minutes','PTS']]=0.
        tabs.loc[8,'Projected minutes']=30.
        with patch('edge_review.role_evidence',return_value=dict(ratios={'PTS':1.},trends={'PTS':0.},detail='Stable')):
            under=assess_edges(edge(15,20,'Under'),main,GAME,pd.DataFrame(),{},reference(),healthy(),NOW,NOW,
                               minutes_cases={'yours':yours,'tabs':tabs}).iloc[0]
        self.assertEqual(under['Minutes flag'],'MIN')
        self.assertFalse(under['Robust play'])


if __name__=='__main__':unittest.main()
