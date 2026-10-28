import test from 'node:test';
import assert from 'node:assert/strict';
import render from '../player_navigation.mjs';

function fixture() {
    globalThis.window = new EventTarget();
    let callback;
    let disconnected = 0, clicks = 0, requests = [];
    globalThis.MutationObserver = class {
        constructor(fn) { callback = fn; }
        observe() {}
        disconnect() { disconnected++; }
    };
    const tab = {textContent: 'Player vs Opponent', click: () => clicks++, focus: () => {}};
    const heading = {textContent: 'Old player vs MIL', scrollIntoView: () => {},
        closest: () => ({querySelectorAll: () => [tab]})};
    globalThis.document = {body: {}, getElementById: () => heading};
    const receiver = {}, sender = {};
    const receive = () => render({data: {mode:'receive'}, parentElement:receiver, setStateValue: (key,value) => requests.push(value)});
    const send = request => render({data: {mode:'send',request}, parentElement:sender});
    const request = {request_id:'1', player_name:'Ryan Rollins', opponent:'HOU'};
    return {receive,send,request,heading,receiver,sender,requests,
        mutate: () => callback(), clicks: () => clicks, disconnected: () => disconnected,
        cleanup: () => {receiver.navigationCleanup?.();sender.navigationCleanup?.();}};
}

test('click requests only the player fragment and waits for its new heading', () => {
    const f=fixture(); f.receive(); f.send(f.request);
    assert.equal(f.requests.length,1); assert.equal(f.clicks(),0);
    f.heading.textContent='Ryan Rollins vs HOU'; f.mutate();
    assert.equal(f.clicks(),1); assert.equal(f.disconnected(),1);
    f.cleanup();
});

test('fragment rerenders do not replay the request or reopen the tab', () => {
    const f=fixture(); f.receive(); f.send(f.request);
    f.heading.textContent='Ryan Rollins vs HOU'; f.mutate();
    f.receive(); f.send(f.request); f.receive();
    assert.equal(f.requests.length,1); assert.equal(f.clicks(),1);
    f.send({...f.request,request_id:'2'}); // Same player selected again intentionally.
    assert.equal(f.requests.length,2); assert.equal(f.clicks(),2);
    f.cleanup();
});

test('sender before receiver still delivers once when receiver becomes ready', () => {
    const f=fixture(); f.send(f.request); f.receive();
    assert.equal(f.requests.length,1); f.receive(); assert.equal(f.requests.length,1);
    f.cleanup();
});
