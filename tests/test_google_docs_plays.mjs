import {test} from 'node:test';
import assert from 'node:assert/strict';
import {googleDocsText} from '../todays_plays.mjs';

test('Docs snapshot contains only the selected slate, saved terms and review status', () => {
    const day = '2026-11-10';
    const pick = {day, player: 'Demo Wing', game: 'A @ B', book: 'FanDuel',
        side: 'Under', line: 5.5, stat: 'REB', projected: 4.6, odds: -115,
        edge: 12.345, status: 'Review', id: 'private-id'};
    const confirmed = {...pick, player: 'Demo Guard', odds: 120, status: 'Confirmed'};
    const other = {...pick, day: '2026-11-11', player: 'Other slate'};
    assert.equal(googleDocsText(day, [confirmed, other, pick]), [
        "Today's Plays · 2026-11-10", '', 'In review',
        'Demo Wing · Under 5.5 REB (-115) · FanDuel · A @ B · Projected 4.60 · Edge 12.35%',
        '', 'Confirmed',
        'Demo Guard · Under 5.5 REB (+120) · FanDuel · A @ B · Projected 4.60 · Edge 12.35%',
    ].join('\n'));
    assert.doesNotMatch(googleDocsText(day, [confirmed]), /In review/);
});
