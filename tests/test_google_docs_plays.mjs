import {test} from 'node:test';
import assert from 'node:assert/strict';
import {googleDocsTable} from '../todays_plays.mjs';
import {CONFIDENCE_TIERS} from '../play_storage.mjs';

test('Docs snapshot contains only the selected slate, saved terms and review status', () => {
    const day = '2026-11-10';
    const pick = {day, player: 'Demo <Wing> & Co.', game: 'A @ B', book: 'FanDuel',
        side: 'Under', line: 5.5, stat: 'REB', projected: 4.6, odds: -115,
        edge: 12.345, status: 'Review', id: 'private-id'};
    const confirmed = {...pick, player: 'Demo Guard', odds: 120, status: 'Confirmed',confidence_cents:15};
    const other = {...pick, day: '2026-11-11', player: 'Other slate'};
    const {html, text} = googleDocsTable(day, [confirmed, other, pick], CONFIDENCE_TIERS);
    assert.equal((html.match(/<tr>/g) || []).length, 3);
    assert.equal((html.match(/<th /g) || []).length, 7);
    assert.equal((html.match(/<td /g) || []).length, 14);
    assert.match(html, /Partner feedback<\/th>/);
    assert.match(html, /background-color:#166449;color:#ffffff;[^>]*>High · 15¢/);
    assert.equal((html.match(/><br><\/td><\/tr>/g) || []).length, 2);
    assert.match(html, /Demo &lt;Wing&gt; &amp; Co.<br>A @ B/);
    assert.doesNotMatch(html, /Other slate|private-id|Demo <Wing>/);
    const [title, headers, review, confirmedRow] = text.split('\n');
    assert.equal(title, "Today's Plays · 2026-11-10");
    assert.equal(headers.split('\t').at(-1), 'Partner feedback');
    assert.deepEqual(review.split('\t'), ['Demo <Wing> & Co. · A @ B', 'Under 5.5 REB (-115) · FanDuel', '4.60', '12.35%', 'In review', 'Low · 5¢', '']);
    assert.match(confirmedRow, /Under 5.5 REB \(\+120\).*Confirmed\tHigh · 15¢\t$/);
});
