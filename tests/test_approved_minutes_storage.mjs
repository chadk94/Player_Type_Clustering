import test from 'node:test';
import assert from 'node:assert/strict';
import render from '../approved_minutes_storage.mjs';

globalThis.window = new EventTarget();
globalThis.document = new EventTarget();
document.visibilityState = 'visible';

test('save, fresh-page restore, replace, and clear without rerun loops', () => {
    const values = new Map();
    globalThis.localStorage = {
        setItem: (k, v) => values.set(k, v),
        getItem: k => values.get(k) ?? null,
        removeItem: k => values.delete(k),
    };
    const key = 'nba-props.approved-minutes.v1.2026-27';
    let count = 0;
    const call = (request, action, bundle, parentElement = {}) => {
        let result;
        const cleanup = render({data: {request, action, bundle, storage_key: key}, parentElement,
                setStateValue: (_, value) => { count++; result = value; }});
        cleanup();
        return result;
    };
    const parent = {};
    const exact = '{"minutes": 18.0, "fractional": 26.1387832623}';
    assert.equal(call('save-1', 'save', exact, parent).status, 'ok');
    assert.equal(call('save-1', 'save', exact, parent), undefined);
    assert.equal(count, 1);
    assert.equal(call('new-session', 'restore').bundle, exact);
    call('replacement', 'save', '{"minutes":20.0}');
    assert.equal(call('refresh', 'restore').bundle, '{"minutes":20.0}');
    call('remove', 'clear');
    assert.equal(call('reopen', 'restore').bundle, null);
    values.set(key, '{broken');
    assert.equal(call('corrupt', 'restore').status, 'invalid');
});

test('blocked browser storage reports failure rather than successful persistence', () => {
    globalThis.localStorage = {
        setItem: () => { throw Error('blocked'); },
        getItem: () => { throw Error('blocked'); },
        removeItem: () => { throw Error('blocked'); },
    };
    for (const action of ['save', 'restore', 'clear']) {
        let result;
        const cleanup = render({data: {request: action, action, storage_key: 'test', bundle: {}}, parentElement: {},
                setStateValue: (_, value) => { result = value; }});
        cleanup();
        assert.equal(result.status, 'unavailable');
    }
});

test('open tabs adopt replacements and clears once without replaying old uploads', () => {
    const values = new Map();
    globalThis.localStorage = {
        setItem: (k,v) => values.set(k,v), getItem: k => values.get(k) ?? null,
        removeItem: k => values.delete(k),
    };
    const key = 'cross-tab';
    const old = '{"teams":2}', full = '{"teams":24,"minutes":18.0}';
    const host = {}, replies = [];
    const data = {request:'old-save',action:'save',storage_key:key,bundle:old,pending:true};
    const props = {data,parentElement:host,setStateValue:(_,r)=>replies.push(r)};
    render(props);
    data.pending = false;
    render(props);
    assert.equal(replies.length,1);
    values.set(key,full); // Another app tab saved its new approved file.
    const change = new Event('storage'); change.key = key;
    window.dispatchEvent(change);
    assert.equal(replies.at(-1).source,'storage');
    assert.equal(replies.at(-1).bundle,full);
    render(props);
    window.dispatchEvent(change);
    window.dispatchEvent(new Event('focus'));
    assert.equal(replies.length,2, 'unchanged data must not cause refresh loops');
    assert.equal(values.get(key),full, 'old upload must not overwrite new storage');
    // Even a newly mounted component cannot replay an acknowledged save.
    const remounted = { ...props, parentElement:{} };
    const cleanup = render(remounted);
    assert.equal(values.get(key),full);
    assert.equal(replies.at(-1).bundle,full);
    cleanup();
    values.delete(key);
    window.dispatchEvent(change);
    assert.equal(replies.at(-1).bundle,null);
    const count = replies.length;
    host.minutesCleanup();
    values.set(key,old);
    window.dispatchEvent(change);
    assert.equal(replies.length,count,'unmount removes listeners');
});

test('focus catches missed changes, corruption and recovery without a timer', () => {
    let raw = '{"teams":2}';
    globalThis.localStorage = {getItem:()=>raw};
    const replies = [], host = {};
    const data = {request:'watch',action:'restore',storage_key:'focus-test',pending:false};
    const cleanup = render({data,parentElement:host,setStateValue:(_,r)=>replies.push(r)});
    raw = '{broken';
    window.dispatchEvent(new Event('focus'));
    assert.equal(replies.at(-1).status,'invalid');
    raw = '{"teams":24}';
    document.dispatchEvent(new Event('visibilitychange'));
    assert.equal(replies.at(-1).status,'ok');
    assert.equal(replies.at(-1).bundle,raw);
    const count = replies.length;
    window.dispatchEvent(new Event('focus'));
    assert.equal(replies.length,count);
    cleanup();
});
