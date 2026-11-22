import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import render, {readSavedTabs, SAVED_TABS_KEY} from '../tabs_browser_bridge.mjs';

const source=readFileSync(new URL('../integrations/tabs_browser_connector/reader.js',import.meta.url),'utf8');
const context=vm.createContext({Date,Error});vm.runInContext(source,context);
const capture=(doc,now='2026-11-18T01:05:00Z')=>context.tabsSnapshot(doc,new Date(now));
function doc({edited=false,scenario='POR: Avdija OUT (Q) ×',published='5:57 PM MST',date='NOV 17 2026 6:05 PM',slate='TUE, 11/17',missingChips=false,empty=false,minutes='28.0',joinedBadge=false,missingOut=false,stats={}}={}) {
    const head=['PLAYER','POS','TEAM','OPP','MIN ▾',...Object.keys(stats)].map(innerText=>({innerText}));
    const row=(name,min,hidden=false)=>({getClientRects:()=>hidden?[]:[{}],querySelectorAll:()=>[
        {innerText:(joinedBadge?'DA\n':'')+name+(min==null?(missingOut?'':joinedBadge?'OUT':' OUT')+'\n8:00 pm ET':' START\n8:00 pm ET'),
         querySelector:key=>joinedBadge?(key==='[data-pn]'?{innerText:name}:key==='.tag.out'&&!missingOut&&min==null?{innerText:'OUT'}:null):null},
        {innerText:'PG'},{innerText:'POR'},{innerText:'MIN'},{},...Object.values(stats).map(innerText=>({innerText}))],
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
test('joined OUT badges preserve the player name without photo initials; missing status is not zero',()=>{
    const out=capture(doc({joinedBadge:true})).rows[1];
    assert.equal(out.Player,'Deni Avdija');
    assert.equal(out.status,'Out');assert.equal(out.minutes,0);
    assert.throws(()=>capture(doc({joinedBadge:true,missingOut:true})),/no verifiable minutes/);
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

test('raw stat totals travel with minutes; missing, percent, invalid and OUT cells stay absent',()=>{
    const rows=capture(doc({stats:{'PTS ↓':'18.2','REB':'5.1','AST':'4.0','3PM':'2.2',STL:'0.0',BLK:'—',TOV:'-1',FPTS:'40.0','DD%':'10%'}})).rows;
    assert.equal(rows[0].PTS,18.2);assert.equal(rows[0].REB,5.1);assert.equal(rows[0].AST,4);
    assert.equal(rows[0]['3PM'],2.2);assert.equal(rows[0].STL,0);
    for (const key of ['BLK','TOV','FPTS','DD%']) assert.equal(rows[0][key],undefined);
    assert.equal(rows[1].PTS,undefined);assert.equal(rows[1].status,'Out');
    for (const value of ['', 'NaN', '2.2%', '2.2 10%', 'Infinity'])
        assert.equal(capture(doc({stats:{'3PM':value}})).rows[0]['3PM'],undefined);
});

function propsDoc(options={}) {
    let view='props', current=doc(options);
    const clicks=[], scroll={isConnected:true,scrollLeft:420,scrollTop:160};
    const focus={isConnected:true,focus:()=>{page.activeElement=focus;}};
    const buttons=Object.fromEntries(['proj','props'].map(name=>[name,{
        getAttribute:()=>String(view===name),
        click:()=>{clicks.push(name);view=name;page.activeElement=null;
            scroll.scrollLeft=0;scroll.scrollTop=0;page.defaultView.scrollY=0;}
    }]));
    const page={get body(){return view==='proj'?current.body:{innerText:'Props table has no full rotation'};},
        activeElement:focus,defaultView:{scrollX:0,scrollY:700,scrollTo:({left,top})=>{
            page.defaultView.scrollX=left;page.defaultView.scrollY=top;}},
        querySelector:key=>key.includes('data-v="props"')?buttons.props:key.includes('data-v="proj"')?buttons.proj:current.querySelector(key),
        querySelectorAll:key=>key==='.scroll'?[scroll]:view==='proj'?current.querySelectorAll(key):[]};
    return {page,clicks,scroll,focus,view:()=>view,update:next=>{current=doc(next);}};
}

test('Props reads the full fresh rotation including OUT rows and restores view, focus and scroll',()=>{
    const source=propsDoc();
    const captureProps=()=>context.tabsCapture(source.page,new Date('2026-11-18T01:05:00Z'));
    const first=captureProps();
    assert.deepEqual(JSON.parse(JSON.stringify(first)),JSON.parse(JSON.stringify(capture(doc()))));
    assert.equal(first.rows[1].status,'Out');
    assert.deepEqual(source.clicks,['proj','props']);
    assert.equal(source.view(),'props');assert.equal(source.page.activeElement,source.focus);
    assert.equal(source.scroll.scrollLeft,420);assert.equal(source.scroll.scrollTop,160);
    assert.equal(source.page.defaultView.scrollY,700);
    // A newer publication / scenario is captured, not a cached prior rotation.
    source.update({minutes:'32.0',scenario:'Default',published:'6:04 PM MST'});
    const second=captureProps();
    assert.equal(second.rows[0].minutes,32);assert.equal(second.scenario,'Default');
    assert.equal(second.source_updated_at,'2026-11-18T01:04:00.000Z');
    assert.equal(source.view(),'props');
});

test('Props restores the selected view even when edited, stale or incomplete data is rejected',()=>{
    for (const options of [{edited:true},{published:'2:00 PM MST'},{missingChips:true},{empty:true},{minutes:'NaN'}]) {
        const source=propsDoc(options);
        assert.throws(()=>context.tabsCapture(source.page,new Date('2026-11-18T01:05:00Z')));
        assert.equal(source.view(),'props');assert.deepEqual(source.clicks,['proj','props']);
        assert.equal(source.scroll.scrollLeft,420);assert.equal(source.page.defaultView.scrollY,700);
    }
    assert.deepEqual(JSON.parse(JSON.stringify(context.tabsCapture(doc(),new Date('2026-11-18T01:05:00Z')))),
        JSON.parse(JSON.stringify(capture(doc()))));
});

test('Props captures the same stat projection and restores the view',()=>{
    const source=propsDoc({stats:{PTS:'18.2','3PM':'2.2'}});
    const snapshot=context.tabsCapture(source.page,new Date('2026-11-18T01:05:00Z'));
    assert.equal(snapshot.rows[0].PTS,18.2);assert.equal(snapshot.rows[0]['3PM'],2.2);
    assert.equal(source.view(),'props');assert.deepEqual(source.clicks,['proj','props']);
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
        button.onclick();reply({ok:true,snapshot:{...snapshot,rows:[...snapshot.rows].reverse()}});
        assert.equal(saved.length,0); // Sorting Tabs does not rerun the dashboard.
        button.onclick();reply({ok:true,snapshot:{...snapshot,scenario:'Default'}});assert.equal(saved.length,1);
        button.onclick();reply({ok:true,snapshot:{...snapshot,scenario:'Default',rows:snapshot.rows.map(r=>({...r,PTS:18.2}))}});
        assert.equal(saved.length,2); // A stat-only update refreshes flags even with unchanged minutes.
        button.onclick();reply({ok:false,error:'Source tab closed'});assert.equal(saved.length,3);assert.equal(saved.at(-1).ok,false);
        cleanup();assert.equal(timers.size,0);assert.equal(handlers.message,undefined);
    } finally {Object.assign(globalThis,real);}
});

test('validated minutes persist through disconnect, timeout and reload; raw replies never replace backup',()=>{
    const handlers={},requests=[],saved=[],timers=new Map(),items=new Map();let tid=0;
    const real={setTimeout,clearTimeout,setInterval,clearInterval,localStorage:globalThis.localStorage};
    globalThis.localStorage={getItem:key=>items.get(key),setItem:(key,value)=>items.set(key,value)};
    globalThis.location={origin:'https://cktest2.streamlit.app'};
    globalThis.window={addEventListener:(name,cb)=>handlers[name]=cb,removeEventListener:(name,cb)=>{if(handlers[name]===cb)delete handlers[name];},postMessage:msg=>requests.push(msg)};
    globalThis.setTimeout=globalThis.setInterval=cb=>{timers.set(++tid,cb);return tid;};
    globalThis.clearTimeout=globalThis.clearInterval=id=>timers.delete(id);
    const status={},button={},snapshot=JSON.parse(JSON.stringify(capture(doc())));
    const mount=data=>render({data,parentElement:{querySelector:k=>k==='button'?button:status},setStateValue:(_key,value)=>saved.push(value)});
    const reply=result=>handlers.message({source:window,origin:location.origin,data:{channel:'nba-tabs-minutes-v1',type:'response',id:requests.at(-1).id,result}});
    try {
        let cleanup=mount({previous:{ok:true,snapshot},saved_snapshot:snapshot});
        reply({ok:false,error:'Source tab closed'});
        assert.deepEqual(saved.at(-1).saved_snapshot,snapshot);assert.equal(saved.at(-1).ok,false);
        assert.match(status.textContent,/Using saved Tabs minutes/);
        assert.deepEqual(readSavedTabs(localStorage),snapshot);
        cleanup();saved.length=0;
        cleanup=mount({}); // A completely new Streamlit/browser page restores its private backup.
        assert.deepEqual(saved[0].saved_snapshot,snapshot);
        const raw={...snapshot,manual_edits:true,rows:[]};
        reply({ok:true,snapshot:raw});
        assert.deepEqual(readSavedTabs(localStorage),snapshot);
        assert.deepEqual(saved.at(-1).saved_snapshot,snapshot); // Fresh reply cannot race away restored teams.
        button.onclick();timers.get(tid)();
        assert.deepEqual(saved.at(-1).saved_snapshot,snapshot);
        assert.equal(saved.at(-1).saved_snapshot.source_updated_at,snapshot.source_updated_at);
        cleanup();
        const newer={...snapshot,scenario:'Default'};
        cleanup=mount({previous:{ok:true,snapshot:newer},saved_snapshot:newer});
        assert.deepEqual(readSavedTabs(localStorage),newer);cleanup();
        const teams={schema:2,snapshots:[snapshot,newer]};
        cleanup=mount({saved_snapshot:teams});cleanup();saved.length=0;
        cleanup=mount({});
        assert.deepEqual(saved[0].saved_snapshot,teams);
        reply({ok:true,snapshot:newer});
        assert.deepEqual(saved.at(-1).saved_snapshot,teams);
        assert.deepEqual(readSavedTabs(localStorage),teams);cleanup();
        items.set(SAVED_TABS_KEY,'broken');assert.equal(readSavedTabs(localStorage),null);
        assert.equal(readSavedTabs({getItem:()=>{throw Error('Blocked storage');}}),null);
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
