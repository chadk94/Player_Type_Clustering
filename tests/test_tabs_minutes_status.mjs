import assert from 'node:assert/strict';
import {test} from 'node:test';
import render, {summarize} from '../tabs_minutes_status.mjs';
import bridge from '../tabs_browser_bridge.mjs';

const now=Date.parse('2026-11-18T22:30:00Z');
const scope={team:'CHI',opponent:'BKN',day:'2026-11-18'};
const row={...scope,published:'2026-11-18T22:00:00Z',captured:'2026-11-18T22:25:00Z',saved:false,complete:true};

test('marker uses game-scoped publication time, not read time or another game',()=>{
    const result=summarize([scope],[row,{...row,day:'2026-11-19',published:'2026-11-19T22:00:00Z'}],now);
    assert.match(result.summary,/CHI: Nov 18, 3:00 PM MST/);
    assert.doesNotMatch(result.summary,/3:25|Nov 19/);
    assert.match(summarize([{...scope,opponent:'BOS'}],[row],now).summary,/no saved minutes/);
});

test('saved, stale, partial and missing sources are distinguished, including the full slate',()=>{
    const rows=[{...row,saved:true},{...row,team:'BKN',opponent:'CHI',published:'2026-11-18T19:00:00Z'},
        {...row,team:'BOS',opponent:'DET',complete:false}];
    const scopes=[...rows.map(({team,opponent,day})=>({team,opponent,day})),{...scope,team:'DET',opponent:'BOS'}];
    const result=summarize(scopes,rows,now);
    assert.match(result.summary,/4 teams · 1 saved · 1 stale · 1 partial · 1 missing/);
    assert.match(result.teams[0].text,/3:00 PM MST \(saved\)/);
    assert.match(summarize([scope],[{...row,published:null}],now).summary,/time unverified/);
});

test('refresh marker invokes the existing bridge; failure preserves timestamp and cleanup removes listeners',()=>{
    const real={window:globalThis.window,location:globalThis.location,setInterval,clearInterval,setTimeout,clearTimeout};
    const window=new EventTarget(),requests=[],updates=[],timers=new Map();let id=0;
    window.postMessage=message=>requests.push(message);
    Object.assign(globalThis,{window,location:{origin:'https://cktest2.streamlit.app'},
        setInterval:cb=>{timers.set(++id,cb);return id;},setTimeout:cb=>{timers.set(++id,cb);return id;},
        clearInterval:i=>timers.delete(i),clearTimeout:i=>timers.delete(i)});
    const parts=Object.fromEntries(['[data-summary]','[data-feedback]','details','[data-teams]','button'].map(k=>[k,{}]));
    const snapshot={rows:[],scope:'All games',scenario:'Default',source_updated_at:row.published};
    const reply=result=>{
        const event=new Event('message');
        Object.assign(event,{source:window,origin:location.origin,data:{channel:'nba-tabs-minutes-v1',type:'response',id:requests.at(-1).id,result}});
        window.dispatchEvent(event);
    };
    try {
        const cleanupBridge=bridge({data:{previous:{ok:true,snapshot,saved_snapshot:snapshot},saved_snapshot:snapshot,metadata:[row]},
            parentElement:{querySelector:()=>({})},setStateValue:(_k,value)=>updates.push(value)});
        const cleanupMarker=render({data:{scopes:[scope],rows:[row],browser:true},parentElement:{querySelector:k=>parts[k]}});
        reply({ok:true,snapshot});
        const stamp=parts['[data-summary]'].textContent;
        parts.button.onclick();
        assert.equal(requests.length,2);
        assert.equal(parts.button.disabled,true);
        parts.button.onclick(); // A repeated click cannot start a competing request.
        assert.equal(requests.length,2);
        reply({ok:false,error:'Source closed'});
        assert.equal(parts.button.disabled,false);
        assert.match(parts['[data-feedback]'].textContent,/saved minutes are preserved/);
        assert.equal(parts['[data-summary]'].textContent,stamp);
        window.dispatchEvent(new CustomEvent('nba-tabs-minutes-status',{detail:{rows:[{...row,published:'2026-11-18T22:15:00Z'}]}}));
        assert.match(parts['[data-summary]'].textContent,/3:15 PM MST/);
        cleanupMarker();cleanupBridge();
        assert.equal(timers.size,0);
        assert.equal(window.nbaTabsMinutesRefresh,undefined);
    } finally {Object.assign(globalThis,real);}
});
