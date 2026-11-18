import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readHistory,packHistoryReply} from '../points_tracking.mjs';
import {gunzipSync} from 'node:zlib';

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

test('a 30 MB history restores losslessly without closing the app connection',async()=>{
    const record=JSON.stringify({id:'test',mean:1.23456789123456789,stat:'PTS',detail:'x'.repeat(1100)});
    const reply={request:'current',status:'ok',full:true,records:Array(27000).fill(record),results:[]};
    assert.ok(Buffer.byteLength(JSON.stringify(reply))>30000000);
    const packed=await packHistoryReply(reply);
    assert.equal(packed.encoding,'gzip-base64');assert.ok(JSON.stringify(packed).length<1000000);
    assert.deepEqual(JSON.parse(gunzipSync(Buffer.from(packed.payload,'base64'))),reply);
    const small={request:'small',status:'ok',records:[record]};
    assert.deepEqual(await packHistoryReply(small),small);
    const native=globalThis.CompressionStream;
    try {globalThis.CompressionStream=undefined;await assert.rejects(packHistoryReply(reply));}
    finally {globalThis.CompressionStream=native;}
});

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
