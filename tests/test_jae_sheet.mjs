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
      const sheet={rows:[],getName:()=>day,getLastColumn(){return this.rows[0]?.length || 0},getFrozenColumns:()=>0,setFrozenColumns(){},getLastRow(){return this.rows.length},getMaxRows:()=>1000,getFrozenRows:()=>1,
        setFrozenRows(){}, setColumnWidths(){}, setColumnWidth(){}, getColumnWidth(){return 100}, hideColumns(){},
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
          isBlank:()=>!sheet.rows.slice(i,i+n).some(row=>row.slice(j,j+w).some(Boolean)),setNotes(){},
          setNumberFormats(){},setNumberFormat(){},setFontColors(){},setFontWeights(){},setBackgrounds(){},
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
  assert.deepEqual(tabs.get(other).rows[2].slice(8,10),['3+ 3PM · FD +400','']);
  const original=tabs.get(other).rows[2].slice();
  const edit=(cents)=>post({day:other,action:'append',rows:[],edits:[{id:over.id,cents}]});
  assert.equal(edit(15).updated,1);
  assert.deepEqual(tabs.get(other).rows[2],[...original.slice(0,5),-113,113/213,...original.slice(7)]);
  assert.equal(edit(15).updated,0);
  assert.equal(edit(10).updated,1);
  assert.equal(tabs.get(other).rows[2][5],-108);
  assert.equal(edit(7).ok,false);
  for(const [quote,limit,label] of [[-110,-125,'High'],[102,-108,'Medium'],[105,100,'Low']])
    assert.equal(ctx.cardConfidence_(quote,limit).label,label);
  ctx.UrlFetchApp={fetchAll:batch=>batch.map(()=>({getResponseCode:()=>200,getContentText:()=>'{"markets":[],"cursor":""}'}))};
  const future='2099-11-16';
  const futurePick={...over,day:future,start:future};
  assert.equal(post({day:future,action:'append',rows:[futurePick]}).added,1);
  const saved=tabs.get(future).rows[1];
  assert.equal(saved[13],over.id);
  assert.deepEqual(saved.slice(8,10),['3+ 3PM · FD +400','']);
  assert.equal(tabs.get(future).rows[0][8],'Alternate to shop');
  assert.equal(post({day:future,action:'append',rows:[futurePick]}).added,0);
  assert.equal(tabs.get(future).rows.length,2);
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
  assert.deepEqual(row.slice(8,10),['O2.5 3PM · FD +400','']);
  assert.deepEqual(cols,{1:'h:mm AM/PM',11:'mmm d, h:mm AM/PM',12:'h:mm AM/PM'});
  const once=[...row];c.upgradeTimesAndAlts_(sheet,{});assert.deepEqual(row,once);
  assert.deepEqual(row.slice(-2),[feedback,id]);
  assert.equal(c.storedTime_('2027-03-15 3:30:00 PM EDT',{}),c.serialTime_('2027-03-15 15:30:00'));
});

test('public asks are date/game, player, market and side specific; PM unders use a lower winning boundary',()=>{
  const ctx=vm.createContext({});vm.runInContext(src,ctx);
  const event='KXNBAPTS-26NOV16CHICLE';
  function market(n,extra={}) {return {market_type:'binary',status:'active',event_ticker:event,
    yes_sub_title:'Sample Jones: '+n+'+',strike_type:'greater',floor_strike:n-.5,
    yes_ask_dollars:'0.56',no_ask_dollars:'0.45',yes_ask_size_fp:'40',yes_bid_size_fp:'145',...extra};}
  const data=[market(5),market(8),market(10),market(11),market(15),market(7,{floor_strike:7}),market(9,{yes_bid_size_fp:'0'}),
    market(6,{yes_sub_title:'Other Jones: 6+'}),market(19,{status:'closed'}),market(20,{event_ticker:'OTHER'})];
  const no=ctx.marketAsks_(data,event,'Sample Jones','Under','PTS');
  assert.deepEqual([...no.keys()],[5,8,10,11,15]);
  assert.equal(ctx.marketAlt_(no,{side:'Under',line:10.5,stat:'PTS'}),10);
  assert.equal(ctx.shopAlternate_({side:'Under',line:10.5,stat:'PTS',shopping:{offers:[{line:9.5,odds:125}]}},no).label,
    'U9.5 PTS (No 10+) · FD +125 · Kalshi 45¢ (+122)');
  const yes=ctx.marketAsks_(data,event,'Sample Jones','Over','PTS');
  assert.equal(yes.get(10),.56);
  assert.equal(ctx.marketAlt_(yes,{side:'Over',line:6.5,stat:'PTS'}),10);
  assert.equal(ctx.marketAlt_(new Map([[5,.5],[6,.45]]),{side:'Over',line:3.5,stat:'REB'}),6);
  assert.equal(ctx.marketAlt_(new Map([[5,.5]]),{side:'Over',line:3.5,stat:'REB'}),5);
  assert.equal(ctx.marketAlt_(new Map([[6,.5]]),{side:'Over',line:5.5,stat:'REB'}),undefined);
  assert.equal(ctx.shopAlternate_({stat:'PTS',line:11.5,side:'Under'},new Map()).label,'No alternate quoted');
});
test('market lookup batches by public event and separates double-double from ten rebounds',()=>{
  const requests=[];
  const ctx=vm.createContext({Utilities:{formatDate:(d,t,fmt)=>fmt==='yyMMMdd'?'26Nov16':'checked'},
    UrlFetchApp:{fetchAll(batch){requests.push(...batch);return batch.map(({url})=>{
      const event=url.split('event_ticker=')[1],dd=event.startsWith('KXNBA2D-');
      return {getResponseCode:()=>200,getContentText:()=>JSON.stringify({cursor:'',markets:[
        {event_ticker:event,status:'active',market_type:'binary',yes_sub_title:dd?'Sample Smith':'Sample Smith: 10+',
         strike_type:'greater',floor_strike:9.5,yes_ask_dollars:dd?'0.32':'0.31',yes_ask_size_fp:'20'}]})};
    });}}});
  vm.runInContext(src,ctx);
  const p={id:'1',game:'CHI @ CLE',player:'Sample Smith',start:'2099-11-16T23:00:00Z',side:'Over',stat:'REB',line:7.5,shopping:{offers:[{line:8.5,odds:130},{line:9.5,odds:185}],double:195}};
  const rows=ctx.fetchCardMarkets_([p,{...p,id:'2'},{...p,id:'ended',start:'2001-01-01'}]);
  assert.equal(rows.size,2);assert.equal(requests.length,2);
  assert.equal(rows.get('1').label,'10+ REB · Kalshi 31¢ (+223) · FD +185');
  assert.deepEqual({...rows.get('1').alt},{line:9.5,odds:185});
  const dd=ctx.shopAlternate_(p,new Map(),.32);
  assert.equal(dd.label,'Double-double · Yes · Kalshi 32¢ (+213) · FD +195');
  assert.equal(dd.alt,null);
  ctx.UrlFetchApp.fetchAll=()=>{throw Error('offline')};
  assert.equal(ctx.fetchCardMarkets_([p]).get('1').label,'10+ REB · FD +185');
});
