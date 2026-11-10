import assert from 'node:assert/strict';
import {test} from 'node:test';
import render from '../injury_alert_voice.mjs';

function fixture() {
    const spoken = [];
    const synth = {
        cancellations: 0,
        speak(line) { spoken.push(line); line.onstart(); },
        cancel() { this.cancellations++; },
    };
    globalThis.window = {
        speechSynthesis: synth,
        SpeechSynthesisUtterance: class { constructor(text) { this.text = text; } },
    };
    const preview = {parentElement: {}}, mute = {}, status = {};
    const elements = {'.preview': preview, '.mute': mute, '[role=status]': status};
    const update = (report = null, messages = [], games = ['a', 'b']) =>
        render({parentElement: {querySelector: key => elements[key]}, data: {games, report, messages}});
    return {spoken, synth, preview, mute, status, update};
}

test('spoken alerts name each affected player once, across repeated fragment renders', () => {
    const {spoken, preview, status, update} = fixture();
    const cle = 'Cleveland Cavaliers. First is now out.';
    const nyk = 'New York Knicks. Second is now off the injury report.';
    update('initial', [cle]);
    assert.equal(spoken.length, 0);
    preview.onclick();
    assert.match(spoken[0].text, /^Voice test/);
    spoken[0].onend();
    assert.match(status.textContent, /on for this tab/);
    update('initial', [cle]);
    update('new', [cle, nyk])();
    update('new', [cle, nyk]);
    assert.equal(spoken.length, 2);
    spoken[1].onend();
    spoken[2].onend();
    update('new', [cle, nyk]);
    assert.deepEqual(spoken.slice(1).map(s => s.text), [cle, nyk]);
});

test('changing scope or muting stops queued speech, and later updates respect mute', () => {
    const {spoken, synth, preview, mute, update} = fixture();
    update();
    preview.onclick();
    spoken[0].onend();
    update('one', ['First', 'Second']);
    update(null, [], []);
    assert.equal(synth.cancellations, 1);
    assert.equal(preview.parentElement.hidden, true);
    spoken.at(-1).onend();
    assert.equal(spoken.length, 2);
    update('two', ['Third'], ['b']);
    mute.onclick();
    update('three', ['Fourth'], ['b']);
    assert.equal(synth.cancellations, 2);
    assert.equal(spoken.length, 3);
});

test('unavailable or blocked browser speech gives visible feedback and can be retried', () => {
    const {spoken, preview, status, update} = fixture();
    update();
    preview.onclick();
    spoken[0].onerror({error: 'not-allowed'});
    assert.match(status.textContent, /could not start/);
    assert.equal(preview.textContent, 'Enable spoken alerts');
    preview.onclick();
    spoken[1].onend();
    assert.match(status.textContent, /on for this tab/);
    delete window.speechSynthesis;
    update();
    assert.equal(preview.disabled, true);
});
