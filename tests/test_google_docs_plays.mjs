import {test} from 'node:test';
import assert from 'node:assert/strict';
import {googleDocsTable} from '../todays_plays.mjs';

test('Docs snapshot contains only the selected slate, saved terms and review status', () => {
    const day = '2026-11-10';
    const pick = {day, player: 'Demo <Wing> & Co.', game: 'A @ B', book: 'FanDuel',
        side: 'Under', line: 5.5, stat: 'REB', projected: 4.6, odds: -115,
        edge: 12.345, status: 'Review', id: 'private-id'};
    const confirmed = {...pick, player: 'Demo Guard', odds: 120, status: 'Confirmed'};
    const other = {...pick, day: '2026-11-11', player: 'Other slate'};
    const {html, text} = googleDocsTable(day, [confirmed, other, pick]);
    assert.equal((html.match(/<tr>/g) || []).length, 3);
    assert.equal((html.match(/<th /g) || []).length, 6);
    assert.equal((html.match(/<td /g) || []).length, 12);
    assert.match(html, /Partner feedback<\/th>/);
    assert.equal((html.match(/><br><\/td><\/tr>/g) || []).length, 2);
    assert.match(html, /Demo &lt;Wing&gt; &amp; Co.<br>A @ B/);
    assert.doesNotMatch(html, /Other slate|private-id|Demo <Wing>/);
    const [title, headers, review, confirmedRow] = text.split('\n');
    assert.equal(title, "Today's Plays · 2026-11-10");
    assert.equal(headers.split('\t').at(-1), 'Partner feedback');
    assert.deepEqual(review.split('\t'), ['Demo <Wing> & Co. · A @ B', 'Under 5.5 REB (-115) · FanDuel', '4.60', '12.35%', 'In review', '']);
    assert.match(confirmedRow, /Under 5.5 REB \(\+120\).*Confirmed\t$/);
});
