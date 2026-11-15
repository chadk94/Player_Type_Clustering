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
      const sheet={rows:[],getName:()=>day,getLastColumn:()=>14,getLastRow(){return this.rows.length},getMaxRows:()=>1000,getFrozenRows:()=>1,
        setFrozenRows(){}, setColumnWidths(){}, setColumnWidth(){}, hideColumns(){},
        getRange(r,c,n,w){const i=r-1,j=c-1;return {
          setFontWeight(){return this},setBackground(){return this},setFontColor(){return this},setWrap(){return this},
          getValues:()=>sheet.rows.slice(i,i+n).map(row=>row.slice(j,j+w)),
          setValues:values=>{for (const [k,row] of values.entries()){
            const dest=sheet.rows[i+k] ||= [];for (const [l,v] of row.entries()) dest[j+l]=v;
          }},
          sort(keys){
            const sorted=sheet.rows.slice(i,i+n).sort((a,b)=>{
              for (const {column} of keys) {
                const x=a[column-1],y=b[column-1]; if (x!==y) return x<y?-1:1;
              }
              return 0;
            });
            sheet.rows.splice(i,n,...sorted);
          },
          setNumberFormats(){},setNumberFormat(){},
        }}};
      tabs.set(day,sheet);return sheet;
    }
  };
  const ctx=vm.createContext({PropertiesService:{getUserProperties:()=>({getProperty:k=>k==='writeKey'?'demo':'sheet'})},
    SpreadsheetApp:{openById:()=>book,flush(){}},
    Utilities:{formatDate:(date)=>date.toISOString().slice(0,19).replace('T',' ')},
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
  tabs.get(day).rows[1][12]='Partner note';
  assert.equal(post({action:'append',rows:[{...row,odds:-115}]}).added,0);
  assert.deepEqual(tabs.get(day).rows[1].slice(4,8),[102,-103,103/203,'']);
  assert.equal(tabs.get(day).rows[1][12],'Partner note');
  const other='2026-11-16';
  assert.equal(post({day:other,action:'append',rows:[{...row,id:'b'.repeat(64),day:other}]}).added,1);
  assert.deepEqual([...tabs.keys()],[day,other]);
  const over={...row,id:'c'.repeat(64),day:other,side:'Over',stat:'3PM',line:1.5,alt:{line:2.5,odds:400}};
  assert.equal(post({day:other,action:'append',rows:[over]}).added,1);
  assert.equal(tabs.get(other).rows[2][7],'Over 2.5 3PM  ·  +400');
  assert.deepEqual(tabs.get(other).rows[2].slice(8,10),[395,100/495]);
  const first={...over,id:'f'.repeat(64),start:'2026-11-14T18:00:00Z',player:'Earlier'};
  assert.equal(post({day:other,action:'append',rows:[first]}).added,1);
  assert.equal(tabs.get(other).rows[1][13],first.id);
  assert.equal(tabs.get(other).rows[3][13],over.id);
  assert.equal(ctx.altLabel_({...over,alt:{line:2.5,odds:null}}),'Over 2.5 3PM  ·  Unavailable');
  assert.equal(post({day:other,action:'append',rows:[{...over,id:'d'.repeat(64),book:'DraftKings'}]}).ok,false);
});

test('legacy tab migration converts actual wall times and keeps feedback and play IDs outside the write',()=>{
  let committed;
  const notes='Keep this feedback', id='e'.repeat(64);
  const rows=[[],['2026-11-15 13:30:00','DAL @ BOS','Test','Over 1.5 3PM','FanDuel',132,127,.44,
      '2026-11-15 11:15:00','2026-11-15 11:17:00',notes,id]];
  const sheet={getLastRow:()=>2,getRange(r,c,n,w){
    assert.deepEqual([r,c,n,w],[1,1,2,10]);
    return {getValues:()=>rows.map(row=>row.slice(0,10)),getFormulas:()=>rows.map(()=>Array(10).fill('')),
      setValues:values=>{committed=values}};
  }};
  const ctx=vm.createContext({Utilities:{
    parseDate:(s,zone)=>{assert.equal(zone,'America/Denver');return new Date(s.replace(' ','T')+'-07:00')},
    formatDate:(date,zone)=>{assert.equal(zone,'America/New_York');return date.toLocaleTimeString('en-US',{timeZone:zone,hour:'numeric',minute:'2-digit'})}
  }});
  vm.runInContext(src,ctx);ctx.migrateDay_(sheet,{});
  assert.equal(committed[0][0],'Game time (ET)');
  assert.equal(committed[1][0],'3:30 PM');assert.equal(committed[1][8],'1:15 PM');assert.equal(committed[1][9],'1:17 PM');
  assert.deepEqual(Array.from(committed[1].slice(4,8)),[132,127,.44,'']);
  assert.deepEqual(rows[1].slice(10),[notes,id]);
});

test('play-to limits cross even money and missing alternates have no limit',()=>{
  const c=vm.createContext({});vm.runInContext(src,c);
  for(const [raw,odds,p] of [[400,395,100/495],[102,-103,103/203],[-150,-155,155/255],[105,100,.5]]) {
    assert.deepEqual(Array.from(c.altLimit_(raw)),[odds,p]);
  }
  assert.deepEqual(Array.from(c.altLimit_(null)),['','']);
});
test('backfill uses frozen odds, preserves precision and dates, and displays off-slate dates',()=>{
  const c=vm.createContext({});vm.runInContext(src,c);
  const id='x',feedback='keep feedback';
  const row=['2026-11-15 12:10:40 AM EST','A @ B','Demo','Over 1.5 3PM',105,100,.5,
    'Over 2.5 3PM  ·  +400','','','2026-11-14 11:59:58 PM EST','2026-11-15 12:10:00 PM EST',feedback,id];
  const cols={};const sheet={getName:()=>'2026-11-15',getLastRow:()=>2,
    getRange(r,k,n,w){return {
      getValues:()=>[row.slice(k-1,k-1+w)],
      setValues:vs=>row.splice(k-1,w,...vs[0]),
      setNumberFormats:f=>{cols[k]=f[0][0]}
    }}};
  c.upgradeTimesAndAlts_(sheet,{});
  assert.equal(row[0],c.serialTime_('2026-11-15 00:10:40'));
  assert.equal(row[11],c.serialTime_('2026-11-15 12:10:00'));
  assert.deepEqual(row.slice(8,10),[395,100/495]);
  assert.deepEqual(cols,{1:'h:mm AM/PM',11:'mmm d, h:mm AM/PM',12:'h:mm AM/PM'});
  const once=[...row];c.upgradeTimesAndAlts_(sheet,{});assert.deepEqual(row,once);
  assert.deepEqual(row.slice(-2),[feedback,id]);
  assert.equal(c.storedTime_('2027-03-15 3:30:00 PM EDT',{}),c.serialTime_('2027-03-15 15:30:00'));
});
