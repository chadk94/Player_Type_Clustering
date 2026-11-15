import assert from 'node:assert/strict';
import test from 'node:test';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';
const src=readFileSync(new URL('../integrations/jae_sheet.gs',import.meta.url),'utf8');
test('pairing uses the saved production deployment and never accepts a preview or external URL',()=>{
  const url='https://script.google.com/macros/s/'+'a'.repeat(40)+'/exec';
  const props=new Map([['writeKey','private']]);
  const alerts=[];
  let entered=url.replace('/exec','/dev'), prompts=0;
  const ui={Button:{OK:'ok'}, ButtonSet:{OK:'ok',OK_CANCEL:'choice'},
    prompt(){prompts++; return {getSelectedButton:()=>'ok',getResponseText:()=>entered}},
    alert(...message){alerts.push(message)}};
  const ctx=vm.createContext({PropertiesService:{getUserProperties:()=>({
    getProperty:k=>props.get(k),setProperty:(k,v)=>props.set(k,v)
  })},SpreadsheetApp:{getUi:()=>ui}});
  vm.runInContext(src,ctx);
  ctx.showConnection();
  assert.equal(props.has('webAppUrl'),false);
  entered=url.replace('script.google.com','example.com');
  ctx.showConnection();
  assert.equal(props.has('webAppUrl'),false);
  entered=url;
  ctx.showConnection();
  assert.equal(alerts.at(-1)[1],url+'#private');
  ctx.showConnection();
  assert.equal(prompts,3);
  assert.equal(alerts.at(-1)[1],url+'#private');
});
test('the sheet creates daily tabs, freezes first delivery and feedback, and refuses unknown keys',()=>{
  const tabs=new Map();
  const book={
    getSheetByName:day=>tabs.get(day),
    insertSheet(day){
      const sheet={rows:[],getLastRow(){return this.rows.length},getMaxRows:()=>1000,getFrozenRows:()=>1,
        getRange(r,c,n,w){const i=r-1,j=c-1;return {
          getValues:()=>sheet.rows.slice(i,i+n).map(row=>row.slice(j,j+w)),
          setValues:values=>{for (const [k,row] of values.entries()){
            const dest=sheet.rows[i+k] ||= [];for (const [l,v] of row.entries()) dest[j+l]=v;
          }},
          setNumberFormat(){},
        }}};
      tabs.set(day,sheet);return sheet;
    }
  };
  const ctx=vm.createContext({PropertiesService:{getUserProperties:()=>({getProperty:k=>k==='writeKey'?'demo':'sheet'})},
    SpreadsheetApp:{openById:()=>book,flush(){}},
    Utilities:{formatDate:(date)=>date.toISOString()},
    LockService:{getScriptLock:()=>({waitLock(){},hasLock:()=>true,releaseLock(){}})},
    ContentService:{MimeType:{JSON:'json'},createTextOutput:t=>({setMimeType:()=>JSON.parse(t)})}
  });
  vm.runInContext(src,ctx);
  const day='2026-11-15', key='demo';
  const row={id:'a'.repeat(64),day,start:day,game:'SAC @ OKC',player:'Sample',side:'Under',line:10.5,stat:'PTS',book:'FanDuel',odds:102,worst:-103,breakeven:103/203,quoted:day,sent:day};
  const post=(params)=>ctx.doPost({postData:{contents:JSON.stringify({day,key,...params})}});
  assert.equal(post({key:'invalid',action:'append',rows:[row]}).ok,false);
  assert.deepEqual(post({action:'check',ids:[row.id]}).existing,[]);
  assert.equal(post({action:'append',rows:[row]}).added,1);
  tabs.get(day).rows[1][10]='Partner note';
  assert.equal(post({action:'append',rows:[{...row,odds:-115}]}).added,0);
  assert.deepEqual(tabs.get(day).rows[1].slice(5,8),[102,-103,103/203]);
  assert.equal(tabs.get(day).rows[1][10],'Partner note');
  const other='2026-11-16';
  assert.equal(post({day:other,action:'append',rows:[{...row,id:'b'.repeat(64),day:other}]}).added,1);
  assert.deepEqual([...tabs.keys()],[day,other]);
});
