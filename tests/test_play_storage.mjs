import {test} from 'node:test';
import assert from 'node:assert/strict';
import {setPick,savedPlay,setPickStatus,allPicks} from '../play_storage.mjs';

class Storage {
    getItem(k) { return this[k] ?? null; }
    setItem(k,v) { this[k]=String(v); }
    removeItem(k) { delete this[k]; }
}
test('quotes survive refreshes, other games, a repeat selection and status changes',()=>{
    const storage=new Storage();
    const a={id:'a',day:'2026-11-10',status:'Review',odds:-110};
    const b={...a,id:'b',day:'2026-11-11'};
    setPick(storage,a,true);setPick(storage,b,true);
    setPickStatus(storage,a.id,'Confirmed');
    setPick(storage,{...a,odds:-150},true);
    assert.deepEqual(savedPlay(storage,a.id),{...a,status:'Confirmed'});
    assert.equal(allPicks(storage).length,2);
    setPickStatus(storage,a.id,'Review');
    assert.deepEqual(savedPlay(storage,a.id),a);
    setPick(storage,a,false);setPickStatus(storage,a.id,'Confirmed');
    assert.deepEqual(allPicks(storage),[b]);
});
