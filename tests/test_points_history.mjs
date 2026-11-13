import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readHistory} from '../points_tracking.mjs';

// Run IndexedDB callbacks in request order, including reads queued from onsuccess.
function historyDB(rows, results) {
    const reads = [];
    return {reads, transaction() {
        const queue = [];
        function read(kind, value) {
            reads.push(kind);
            const request = {result:value};
            queue.push(() => request.onsuccess?.());
            return request;
        }
        const tx = {objectStore: name => name === 'results'
            ? {getAll: () => read('results', results)}
            : {getAllKeys: () => read('keys', [...rows.keys()]),
               getAll: () => read('all forecasts', [...rows.values()]),
               get: key => read(key, rows.get(key))}};
        queueMicrotask(() => {
            while (queue.length) queue.shift()();
            tx.oncomplete();
        });
        return tx;
    }};
}

test('restore once, read only new forecast bodies including other tabs, and recover from deletion', async () => {
    const season = '2026-27';
    const rows = new Map([['a', {id:'a', season, body:'{"minutes":30.0}'}],
                          ['old', {id:'old',season:'2025-26',body:'old season'}]]);
    const results = [{id:'a',actual:5}];
    const db = historyDB(rows,results);
    const first = await readHistory(db,season,null);
    assert.deepEqual(first.reply,{full:true,records:['{"minutes":30.0}'],results});
    db.reads.length = 0;
    rows.set('b', {id:'b',season,body:'new from another tab'});
    const next = await readHistory(db,season,first.index);
    assert.deepEqual(next.reply,{full:false,records:['new from another tab'],results:[]});
    assert.deepEqual(db.reads,['keys','results','b']);
    results[0] = {id:'a',actual:6};
    const corrected = await readHistory(db,season,next.index);
    assert.deepEqual(corrected.reply,{full:false,records:[],results});
    rows.delete('a');
    const recovered = await readHistory(db,season,corrected.index);
    assert.equal(recovered.reply.full,true);
    assert.deepEqual(recovered.reply.records,['new from another tab']);
});
