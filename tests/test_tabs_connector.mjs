import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import render from '../tabs_browser_bridge.mjs';

const source=readFileSync(new URL('../integrations/tabs_browser_connector/reader.js',import.meta.url),'utf8');
const context=vm.createContext({Date,Error});vm.runInContext(source,context);
const capture=(doc,now='2026-11-18T01:05:00Z')=>context.tabsSnapshot(doc,new Date(now));
function doc({edited=false,scenario='POR: Avdija OUT (Q) ×',published='5:57 PM MST',date='NOV 17 2026 6:05 PM',slate='TUE, 11/17',missingChips=false,empty=false,minutes='28.0'}={}) {
    const head=['PLAYER','POS','TEAM','OPP','MIN ▾'].map(innerText=>({innerText}));
    const row=(name,min,hidden=false)=>({getClientRects:()=>hidden?[]:[{}],querySelectorAll:()=>[
        {innerText:name+(min==null?' OUT\n8:00 pm ET':' START\n8:00 pm ET')},{innerText:'PG'},{innerText:'POR'},{innerText:'MIN'},{}],
        querySelector:()=>min==null?null:{value:min,getAttribute:()=>`Minutes for ${name}`}});
    const rows=[{querySelectorAll:()=>head},...empty?[]:[row('Jrue Holiday',minutes),row('Deni Avdija',null),row('Hidden player','48',true)]];
    const table={querySelectorAll:()=>rows};
    return {body:{innerText:`${date}\n${slate} · 5 GAMES\nLast updated ${published}\n${edited?'Your changes':'No changes yet: these are Tabs numbers'}`},
        querySelectorAll:()=>[table],querySelector:key=>key==='#scenChips'?(missingChips?null:{innerText:scenario}):key==='#game'?{selectedIndex:0,options:[{textContent:'POR @ MIN'}]}:null};
}

test('rendered inputs and explicit OUT rows retain chosen scenario and source timestamp',()=>{
    const snapshot=capture(doc());
    assert.equal(snapshot.rows.length,2);assert.equal(snapshot.rows[0].Player,'Jrue Holiday');
    assert.equal(snapshot.rows[0].minutes,28);assert.equal(snapshot.rows[1].minutes,0);
    assert.equal(snapshot.scenario,'POR: Avdija OUT (Q)');assert.equal(snapshot.slate_date,'2026-11-17');
    assert.equal(snapshot.source_updated_at,'2026-11-18T00:57:00.000Z');
});
test('manual edits, stale source, changed controls, empty tables and invalid minutes do not clear checks',()=>{
    for (const options of [{edited:true},{published:'2:00 PM MST'},{missingChips:true},{empty:true},{minutes:''},{minutes:'NaN'},{minutes:'49'}])
        assert.throws(()=>capture(doc(options)));
    const unpublished=doc();unpublished.body.innerText='No projections yet';
    assert.throws(()=>capture(unpublished),/not published this slate/);
});
test('default after news, midnight and year boundaries retain correct date',()=>{
    assert.equal(capture(doc({scenario:''})).scenario,'Default');
    const s=capture(doc({date:'JAN 1 2027 12:05 AM',slate:'THU, 12/31',published:'11:57 PM MST'}),'2027-01-01T07:05:00Z');
    assert.equal(s.slate_date,'2026-12-31');assert.equal(s.source_updated_at,'2027-01-01T06:57:00.000Z');
});

test('only changed snapshots update Python; disconnect clears; cleanup stops polling',()=>{
    const handlers={},requests=[],saved=[],timers=new Map();let tid=0;
    globalThis.location={origin:'https://cktest2.streamlit.app'};
    globalThis.window={addEventListener:(name,cb)=>handlers[name]=cb,removeEventListener:(name,cb)=>{if(handlers[name]===cb)delete handlers[name];},postMessage:msg=>requests.push(msg)};
    const real={setTimeout,clearTimeout,setInterval,clearInterval};
    globalThis.setTimeout=globalThis.setInterval=cb=>{timers.set(++tid,cb);return tid;};
    globalThis.clearTimeout=globalThis.clearInterval=id=>timers.delete(id);
    const status={},button={},snapshot=JSON.parse(JSON.stringify(capture(doc())));
    const reply=result=>handlers.message({source:window,origin:location.origin,data:{channel:'nba-tabs-minutes-v1',type:'response',id:requests.at(-1).id,result}});
    try {
        const cleanup=render({data:{previous:{ok:true,snapshot}},parentElement:{querySelector:k=>k==='button'?button:status},setStateValue:(_key,value)=>saved.push(value)});
        reply({ok:true,snapshot:{...snapshot,observed_at:'2026-11-18T01:05:30Z'}});
        assert.equal(saved.length,0);assert.match(status.textContent,/Connected/);
        button.onclick();reply({ok:true,snapshot:{...snapshot,scenario:'Default'}});assert.equal(saved.length,1);
        button.onclick();reply({ok:false,error:'Source tab closed'});assert.equal(saved.length,2);assert.equal(saved.at(-1).ok,false);
        cleanup();assert.equal(timers.size,0);assert.equal(handlers.message,undefined);
    } finally {Object.assign(globalThis,real);}
});

test('extension permissions are limited to the two sites and session storage',()=>{
    const manifest=JSON.parse(readFileSync(new URL('../integrations/tabs_browser_connector/manifest.json',import.meta.url)));
    assert.deepEqual(manifest.permissions,['storage']);
    assert.deepEqual(manifest.host_permissions,['https://app.tabsnba.com/*','https://cktest2.streamlit.app/*']);
    assert.ok(!manifest.content_scripts[0].all_frames);
    assert.equal(manifest.content_scripts[1].all_frames,true);
    for (const name of ['reader.js','relay.js','background.js']) {
        const code=readFileSync(new URL('../integrations/tabs_browser_connector/'+name,import.meta.url),'utf8');
        assert.doesNotMatch(code,/fetch\(|XMLHttpRequest|document\.cookie|localStorage|webRequest/);
    }
});

test('relay only reads the explicitly selected Tabs tab for the prop app',async()=>{
    let selected,click,removed,message;const calls=[];
    const chrome={storage:{session:{get:async()=>({sourceTab:selected}),set:async v=>{selected=v.sourceTab;},remove:async()=>{selected=undefined;}}},
        action:{onClicked:{addListener:cb=>click=cb},setBadgeText:async()=>{},setBadgeBackgroundColor:async()=>{},setTitle:async()=>{}},
        tabs:{onRemoved:{addListener:cb=>removed=cb},get:async id=>({id,url:'https://app.tabsnba.com/projections'}),
              sendMessage:async id=>{calls.push(id);return {ok:true,snapshot:{scenario:'Selected tab only'}};}},
        runtime:{onMessage:{addListener:cb=>message=cb}}};
    const sandbox=vm.createContext({chrome,URL,JSON,Error});
    vm.runInContext(readFileSync(new URL('../integrations/tabs_browser_connector/background.js',import.meta.url),'utf8'),sandbox);
    await click({id:10,url:'https://app.tabsnba.com/projections'});
    const ask=sender=>new Promise(resolve=>{const keep=message({type:'tabs-minutes-read-v1'},sender,resolve);if(!keep)resolve(null);});
    assert.equal(await ask({frameId:0,url:'https://other.example'}),null);
    assert.equal(await ask({frameId:1,url:'https://other.example'}),null);
    assert.equal((await ask({frameId:1,url:'https://cktest2.streamlit.app/~/+/'})).ok,true);
    assert.deepEqual(calls,[10]);
    await click({id:20,url:'https://app.tabsnba.com/projections'});
    await ask({frameId:0,url:'https://cktest2.streamlit.app/'});assert.deepEqual(calls,[10,20]);
    await removed(20);
    assert.equal((await ask({frameId:0,url:'https://cktest2.streamlit.app/'})).ok,false);
    assert.deepEqual(calls,[10,20]);
});
