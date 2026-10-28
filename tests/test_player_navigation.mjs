import test from 'node:test';
import assert from 'node:assert/strict';
import render from '../player_navigation.mjs';

test('one navigation per request; returning to the table does not reopen the player', () => {
    let clicks = 0, replies = 0;
    const tab = {textContent: 'Player vs Opponent', click: () => clicks++, focus: () => {}};
    const heading = {textContent: 'Ryan Rollins vs HOU', scrollIntoView: () => {},
        closest: () => ({querySelectorAll: () => [tab]})};
    globalThis.document = {getElementById: () => heading};
    const parentElement = {};
    const call = data => render({data, parentElement, setStateValue: () => replies++});
    call({id: 1, heading: heading.textContent});
    call({id: 1, heading: heading.textContent});
    call(null);
    assert.equal(clicks, 1);
    assert.equal(replies, 1);
    call({id: 2, heading: heading.textContent}); // Same player can be clicked again.
    assert.equal(clicks, 2);
});

test('waits for the selected player heading, then disconnects its observer', () => {
    let observe, disconnected = 0, clicks = 0;
    globalThis.MutationObserver = class {
        constructor(callback) { observe = callback; }
        observe() {}
        disconnect() { disconnected++; }
    };
    const tab = {textContent: 'Player vs Opponent', click: () => clicks++, focus: () => {}};
    const heading = {textContent: 'Old player vs MIL', scrollIntoView: () => {},
        closest: () => ({querySelectorAll: () => [tab]})};
    globalThis.document = {body: {}, getElementById: () => heading};
    render({data: {id: 3, heading: 'Amen Thompson vs MIL'}, parentElement: {}, setStateValue: () => {}});
    assert.equal(clicks, 0);
    heading.textContent = 'Amen Thompson vs MIL';
    observe();
    assert.equal(clicks, 1);
    assert.equal(disconnected, 1);
});
