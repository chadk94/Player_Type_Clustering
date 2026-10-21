import test from 'node:test';
import assert from 'node:assert/strict';
import render from '../approved_minutes_storage.mjs';

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
        render({data: {request, action, bundle, storage_key: key}, parentElement,
                setStateValue: (_, value) => { count++; result = value; }});
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
        render({data: {request: action, action, storage_key: 'test', bundle: {}}, parentElement: {},
                setStateValue: (_, value) => { result = value; }});
        assert.equal(result.status, 'unavailable');
    }
});
