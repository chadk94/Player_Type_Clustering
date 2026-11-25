import {test} from 'node:test';
import assert from 'node:assert/strict';
import {setPick,savedPlay,setPickStatus,allPicks,saveConfidence,acknowledgeConfidence,saveAlternate,acknowledgeAlternates} from '../play_storage.mjs';

class Storage {
    getItem(k) { return this[k] ?? null; }
    setItem(k,v) { this[k]=String(v); }
    removeItem(k) { delete this[k]; }
}
test('alternate choices persist and an older send cannot clear a newer edit',()=>{
    const s=new Storage(),p={id:'p',odds:-110,status:'Confirmed',confidence_cents:10};
    setPick(s,p,true);saveAlternate(s,p,'dd');
    assert.deepEqual(savedPlay(s,'p'),{...p,alt_choice:'dd',alt_edited:true});
    saveAlternate(s,p,'line:14.5');acknowledgeAlternates(s,{p:'dd'});
    assert.equal(savedPlay(s,'p').alt_edited,true);
    acknowledgeAlternates(s,{p:'line:14.5'});
    assert.equal(savedPlay(s,'p').alt_edited,false);
    assert.equal(savedPlay(s,'p').odds,-110);
});
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
test('changing confidence on either page preserves the original terms and confirmation',()=>{
    const storage=new Storage();
    const p={id:'p',odds:-110,status:'Review'};
    saveConfidence(storage,p,15);
    setPickStatus(storage,p.id,'Confirmed');
    saveConfidence(storage,{...p,odds:-130},10);
    acknowledgeConfidence(storage,{p:15});
    assert.deepEqual(allPicks(storage),[{...p,status:'Confirmed',confidence_cents:10,confidence_edited:true}]);
    acknowledgeConfidence(storage,{p:10});
    assert.equal(savedPlay(storage,'p').confidence_edited,false);
    assert.throws(()=>saveConfidence(storage,p,9));
    setPick(storage,p,false);
    saveConfidence(storage,p,5);
    assert.deepEqual(allPicks(storage),[{...p,confidence_cents:5,confidence_edited:true}]);
});
