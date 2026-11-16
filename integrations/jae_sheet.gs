// Copy to the Apps Script project bound to the private Jae workbook.
const LEGACY_HEADERS = ['Game time (Denver)', 'Game', 'Player', 'Prop', 'Sportsbook', 'Quoted odds',
  'Worst odds', 'Worst break-even %', 'Quote time (Denver)', 'Sent time (Denver)', 'Partner feedback', 'Play ID'];

const PREVIOUS_HEADERS = ['Game time (ET)', 'Game', 'Player', 'Prop', 'FanDuel odds', 'Worst odds',
  'Worst break-even %', 'Alternate over · FD odds', 'Quote time (ET)', 'Sent time (ET)', 'Partner feedback', 'Play ID'];
const LIMIT_HEADERS = ['Tip-off (ET)', 'Game', 'Player', 'Prop', 'FanDuel odds', 'Play to', 'Break-even %',
  'Alternate over · FD odds', 'Alt play to', 'Alt break-even %', 'Quoted (ET)', 'Sent (ET)', 'Partner feedback', 'Play ID'];
const HEADERS = LIMIT_HEADERS.map((name,i)=> i===8 ? 'Alternate to shop' : i===9 ? 'Reserved' : name);
const EASTERN = 'America/New_York';

function easternStamp_(date) {
  return Utilities.formatDate(date, EASTERN, 'yyyy-MM-dd h:mm:ss a z');
}

function onOpen() {
  SpreadsheetApp.getUi().createMenu('Jae card')
    .addItem('Set up or update daily tabs', 'setupCard')
    .addItem('Set deployed web app URL', 'setDeploymentUrl')
    .addItem('Show dashboard connection', 'showConnection').addToUi();
}

function setupCard() {
  const book = SpreadsheetApp.getActiveSpreadsheet();
  const props = PropertiesService.getUserProperties();
  props.setProperty('sheetId', book.getId());
  if (!props.getProperty('writeKey')) {
    props.setProperty('writeKey', (Utilities.getUuid() + Utilities.getUuid()).replace(/-/g, ''));
  }
  const day = Utilities.formatDate(new Date(), 'America/Denver', 'yyyy-MM-dd');
  const lock = LockService.getScriptLock();
  lock.waitLock(20000);
  try {
    const days = new Set(book.getSheets().map(s => s.getName()).filter(name => /^\d{4}-\d{2}-\d{2}$/.test(name)));
    days.add(day);
    for (const date of days) sortDay_(ensureDay_(book, date));
    SpreadsheetApp.flush();
    book.toast(days.size + ' daily cards ready. Game and quote times are Eastern.');
  } finally {
    lock.releaseLock();
  }
}

function setDeploymentUrl() {
  const ui = SpreadsheetApp.getUi();
  const response = ui.prompt('Set deployed web app URL',
    'Paste the Web app URL from Deploy → Manage deployments in this sheet’s Apps Script project.',
    ui.ButtonSet.OK_CANCEL);
  if (response.getSelectedButton() !== ui.Button.OK) return;
  const url = response.getResponseText().trim();
  if (!/^https:\/\/script\.google\.com\/macros\/s\/[\w-]{20,200}\/exec$/.test(url)) {
    ui.alert('Use the deployed Google Web app URL ending in /exec.');
    return;
  }
  PropertiesService.getUserProperties().setProperty('webAppUrl', url);
  return url;
}

function showConnection() {
  const props = PropertiesService.getUserProperties();
  const key = props.getProperty('writeKey');
  if (!key) {
    SpreadsheetApp.getUi().alert('Run Set up or update daily tabs in this sheet first.');
    return;
  }
  const url = props.getProperty('webAppUrl') || setDeploymentUrl();
  if (!url) return;
  SpreadsheetApp.getUi().alert('Private dashboard connection', url + '#' + key,
    SpreadsheetApp.getUi().ButtonSet.OK);
}

function migrateDay_(sheet, book) {
  const range = sheet.getRange(1, 1, sheet.getLastRow(), 10);
  const rows = range.getValues();
  const formulas = range.getFormulas();
  const convert = value => {
    if (value === '') return '';
    // Older tabs stored Denver wall times. Sheets can return either text
    // or a Date interpreted in the workbook's own time zone.
    const wallTime = value instanceof Date
      ? Utilities.formatDate(value, book.getSpreadsheetTimeZone(), 'yyyy-MM-dd HH:mm:ss')
      : String(value);
    return easternStamp_(Utilities.parseDate(wallTime, 'America/Denver', 'yyyy-MM-dd HH:mm:ss'));
  };
  const migrated = [PREVIOUS_HEADERS.slice(0, 10)];
  for (let i = 1; i < rows.length; i++) {
    const r = rows[i];
    const odds = r[4] && r[4] !== 'FanDuel' ? r[5] + ' (' + r[4] + ')' : r[5];
    migrated.push([convert(r[0]), formulas[i][1] || r[1], formulas[i][2] || r[2],
      formulas[i][3] || r[3], odds, r[6], r[7], '', convert(r[8]), convert(r[9])]);
  }
  // Commit the new header together with the converted times, so a retry cannot
  // convert a row twice. Leave the partner feedback and identity columns alone.
  range.setValues(migrated);
}

// Store an Eastern wall-clock serial so Sheets can sort and display the full
// timestamp regardless of the workbook owner's time zone.
function serialTime_(wallTime) {
  const ms = Date.parse(wallTime.replace(' ', 'T') + 'Z');
  if (!Number.isFinite(ms)) throw Error('Invalid card time.');
  return ms / 86400000 + 25569;
}

function storedTime_(value, book) {
  if (value === '' || typeof value === 'number') return value;
  if (value instanceof Date) {
    return serialTime_(Utilities.formatDate(value, book.getSpreadsheetTimeZone(), 'yyyy-MM-dd HH:mm:ss'));
  }
  const parts = String(value).match(/^(\d{4}-\d{2}-\d{2}) (\d{1,2}):(\d{2}):(\d{2}) (AM|PM) E[DS]T$/);
  if (!parts) throw Error('Unrecognized Eastern timestamp on the card.');
  const hour = Number(parts[2]) % 12 + (parts[5] === 'PM' ? 12 : 0);
  return serialTime_(parts[1]+' '+String(hour).padStart(2, '0')+':'+parts[3]+':'+parts[4]);
}

function playLimit_(price,cents) {
  if (!Number.isInteger(price) || Math.abs(price) < 100) return ['', ''];
  const ticks = (price > 0 ? price - 100 : price + 100) - cents;
  const worst = ticks >= 0 ? ticks + 100 : ticks - 100;
  const probability = worst > 0 ? 100 / (worst + 100) : -worst / (100 - worst);
  return [worst, probability];
}

function formatTimes_(sheet, first, rows) {
  if (!rows.length) return;
  const day = serialTime_(sheet.getName() + ' 00:00:00');
  // Show the date only if the timestamp falls outside the slate date.
  for (const col of [1, 11, 12]) {
    const formats = rows.map(r => {
      const value = r[col-1];
      return [value === '' || Math.floor(value) === day ? 'h:mm AM/PM' : 'mmm d, h:mm AM/PM'];
    });
    sheet.getRange(first, col, rows.length, 1).setNumberFormats(formats);
  }
}

function upgradeTimesAndAlts_(sheet, book) {
  const count = sheet.getLastRow() - 1;
  if (count < 1) return;
  const rows = sheet.getRange(2, 1, count, 12).getValues();
  for (const row of rows) {
    for (const col of [0, 10, 11]) row[col] = storedTime_(row[col], book);
    row[8] = savedAlt_(row[7]);
    row[9] = '';
  }
  sheet.getRange(2, 1, count, 1).setValues(rows.map(r => [r[0]]));
  sheet.getRange(2, 9, count, 4).setValues(rows.map(r => r.slice(8, 12)));
  formatTimes_(sheet, 2, rows);
}

function sortDay_(sheet) {
  const count = sheet.getLastRow() - 1;
  if (count > 1) {
    // Sort whole rows, including private IDs, feedback and any partner columns.
    sheet.getRange(2, 1, count, sheet.getLastColumn()).sort([
      {column:1, ascending:true}, {column:2, ascending:true}, {column:3, ascending:true}
    ]);
  }
  formatConfidence_(sheet,count);
  return sheet;
}

// E and F are the frozen quote and main play-to price. Keep F numeric for the grader.
function cardConfidence_(quote, limit) {
  if (![quote,limit].every(x=>Number.isInteger(x) && Math.abs(x)>=100)) return null;
  const ticks=x=>x-Math.sign(x)*100;
  return ({
    15: {label:'High',fill:'#166449',ink:'#ffffff',weight:'bold'},
    10: {label:'Medium',fill:'#ffe4a3',ink:'#633d00',weight:'bold'},
    5: {label:'Low',fill:'#edf0f3',ink:'#45515d',weight:'normal'}
  })[ticks(quote)-ticks(limit)] || null;
}

function formatConfidence_(sheet,count) {
  if (sheet.getColumnWidth(6)<134) sheet.setColumnWidth(6,134);
  if (count<1) return;
  const tiers=sheet.getRange(2,5,count,2).getValues().map(([quote,limit])=>cardConfidence_(quote,limit));
  const prices=sheet.getRange(2,6,count,1);
  prices.setNumberFormats(tiers.map(t=>[t ? '"'+t.label+' · "+0;"'+t.label+' · "-0;0' : '+0;-0;0']));
  prices.setBackgrounds(tiers.map(t=>[t ? t.fill : null]));
  prices.setFontColors(tiers.map(t=>[t ? t.ink : null]));
  prices.setFontWeights(tiers.map(t=>[t ? t.weight : 'normal']));
}

function ensureDay_(book, day) {
  let sheet = book.getSheetByName(day);
  if (!sheet) sheet = book.insertSheet(day);
  if (sheet.getLastRow()) {
    const current = sheet.getRange(1, 1, 1, HEADERS.length).getValues()[0];
    const same = (a,b) => JSON.stringify(a) === JSON.stringify(b);
    if (same(current, HEADERS)) return sheet;
    const old = current.slice(0, PREVIOUS_HEADERS.length);
    if (same(current,LIMIT_HEADERS)) {
      const count = sheet.getLastRow()-1;
      if (count>0) {
        const original=sheet.getRange(2,8,count,1).getValues();
        sheet.getRange(2,9,count,2).setValues(original.map(([label])=>[savedAlt_(label),'']));
      }
      const previous=sheet.getRange(1,19,1,4).getValues()[0];
      if (previous[0]==='Kalshi alternate' && previous[3]==='PM checked (ET)') {
        sheet.getRange(1,19,sheet.getLastRow(),4).clearContent();
        sheet.hideColumns(19,4);
      }
    } else if (same(old, LEGACY_HEADERS) || same(old, PREVIOUS_HEADERS)) {
      if (same(old, LEGACY_HEADERS)) migrateDay_(sheet, book);
      // Inserting preserves partner feedback cells, notes and the hidden ID.
      sheet.insertColumnsAfter(8, 2);
    } else if (!same(current, [...PREVIOUS_HEADERS.slice(0,8), '', '', ...PREVIOUS_HEADERS.slice(8)])) {
      throw Error('Unrecognized daily card headers.');
    }
    // Also resumes an interrupted upgrade after the columns were inserted.
    if (!same(current,LIMIT_HEADERS)) upgradeTimesAndAlts_(sheet, book);
  }
  sheet.setFrozenRows(1);
  sheet.setColumnWidths(1, HEADERS.length, 103);
  for (const [col, width] of [[1,100],[2,108],[3,165],[4,140],[9,385],[13,180]]) {
    sheet.setColumnWidth(col, width);
  }
  sheet.hideColumns(8);  // Frozen FD alternate for the existing grader.
  sheet.hideColumns(10);
  sheet.hideColumns(HEADERS.length);
  const count = sheet.getMaxRows() - 1;
  if (count > 0) {
    for (const col of [5,6]) sheet.getRange(2,col,count,1).setNumberFormat('+0;-0;0');
    for (const col of [7]) sheet.getRange(2,col,count,1).setNumberFormat('0.0%');
    sheet.getRange(2,8,count,2).setNumberFormat('@');
  }
  const header = sheet.getRange(1,1,1,HEADERS.length);
  header.setFontWeight('bold').setBackground('#214b56').setFontColor('#ffffff').setWrap(true);
  header.setValues([HEADERS]);  // Mark complete after all migration steps.
  return sheet;
}

function altLabel_(r) {
  if (r.side !== 'Over') return '';
  if (!r.alt || !Number.isFinite(r.alt.line) || r.alt.line <= r.line) return 'Unavailable';
  const label = 'Over ' + r.alt.line + ' ' + r.stat + '  ·  ';
  const odds = r.alt.odds;
  return label + (Number.isInteger(odds) && Math.abs(odds) >= 100
    ? (odds > 0 ? '+' : '') + odds : 'Unavailable');
}

// Each public event is one stat, one game and one Eastern slate date.
const PM_SERIES = {PTS:'KXNBAPTS', REB:'KXNBAREB', AST:'KXNBAAST', '3PM':'KXNBA3PT',
  STL:'KXNBASTL', BLK:'KXNBABLK', PRA:'KXNBAPRA', DD:'KXNBA2D'};
function savedAlt_(label) {
  const m = /^Over ([\d.]+) (\w+)\s*·\s*([+-]\d+)$/.exec(String(label).trim());
  return m ? 'O'+m[1]+' '+m[2]+' · FD '+m[3] : '';
}

function marketName_(name) {
  return String(name).normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase().replace(/[^a-z0-9]/g,'');
}

function marketEvent_(pick, stat) {
  const teams = /^([A-Z]{3}) @ ([A-Z]{3})$/.exec(pick.game);
  if (!teams || !PM_SERIES[stat]) return null;
  return PM_SERIES[stat]+'-'+Utilities.formatDate(new Date(pick.start), EASTERN,'yyMMMdd').toUpperCase()+teams[1]+teams[2];
}

function marketAsks_(markets, event, player, side, stat) {
  const asks = new Map();
  for (const m of markets || []) {
    if (m.event_ticker !== event || m.status !== 'active' || m.market_type !== 'binary') continue;
    const match = stat === 'DD' ? [null,m.yes_sub_title,0] : /^(.+): (\d+)\+$/.exec(m.yes_sub_title);
    if (!match || marketName_(match[1]) !== marketName_(player)) continue;
    const n = Number(match[2]);
    if (stat !== 'DD' && (!(n > 0) || m.strike_type !== 'greater' || Number(m.floor_strike) !== n-0.5)) continue;
    // The NO ask is backed by the YES bid. Kalshi exposes depth on the YES book.
    const price = Number(side === 'Over' ? m.yes_ask_dollars : m.no_ask_dollars);
    const size = Number(side === 'Over' ? m.yes_ask_size_fp : m.yes_bid_size_fp);
    if (!(price > 0 && price < 1 && size >= 1)) continue;
    asks.set(n, asks.has(n) ? null : price);
  }
  return asks;
}

function marketAlt_(asks, pick) {
  const over = pick.side === 'Over';
  const step = {REB:2,PTS:5}[pick.stat] || 1;
  const rungs = [...asks.keys()].filter(n => asks.get(n) && (over ? n-0.5 > pick.line : n-0.5 < pick.line))
    .sort((a,b) => over ? a-b : b-a);
  if (pick.stat === '3PM' && over) return rungs.find(n => n-0.5 === pick.line+1);
  return rungs.find(n => n % step === 0) || rungs[0];
}

function shopAlternate_(p, asks, ddAsk) {
  const over = p.side === 'Over';
  const fd = new Map();
  const offers = p.shopping ? p.shopping.offers : (p.alt ? [p.alt] : []);
  for (const offer of offers || []) {
    if (Number.isFinite(offer.line) && Number.isInteger(offer.odds) && Math.abs(offer.odds)>=100)
      fd.set(offer.line+0.5,offer.odds);
  }
  const all = new Map([...asks].filter(([,cost])=>cost));
  for (const n of fd.keys()) if (!all.has(n)) all.set(n,true);
  const shared = new Map([...asks].filter(([n,cost])=>cost && fd.has(n)));
  const sharedN = marketAlt_(shared,p);
  const fdDouble = p.shopping && p.shopping.double;
  const useDD = sharedN===undefined && over && p.stat==='REB' && p.line>=7.5 && p.line<=9.5 &&
    Number.isInteger(fdDouble) && Math.abs(fdDouble)>=100 && ddAsk>0;
  const n = sharedN===undefined ? marketAlt_(all,p) : sharedN;
  if (!useDD && n===undefined) return {label:'No alternate quoted',alt:null};
  const cost = useDD ? ddAsk : asks.get(n), odds = useDD ? fdDouble : fd.get(n);
  const signed = v => v>0 ? '+'+v : String(v);
  const prices = [];
  if (odds) prices.push({cost:odds>0 ? 100/(odds+100) : -odds/(100-odds),text:'FD '+signed(odds)});
  if (cost) {
    const american = Math.round(cost<=0.5 ? 100*(1-cost)/cost : -100*cost/(1-cost));
    prices.push({cost,text:'Kalshi '+Number((100*cost).toFixed(2))+'¢ ('+signed(american)+')'});
  }
  const label = useDD ? 'Double-double · Yes' : (Number.isInteger(n) ?
    (over ? n+'+ '+p.stat : 'U'+(n-0.5)+' '+p.stat+' (No '+n+'+)') :
    ((over?'Over ':'Under ')+(n-0.5)+' '+p.stat));
  return {label:[label,...prices.sort((a,b)=>a.cost-b.cost).map(x=>x.text)].join(' · '),
    alt:!useDD && over && odds ? {line:n-0.5,odds} : null};
}

function fetchCardMarkets_(picks) {
  const pending = picks.filter(p => new Date(p.start).getTime() > Date.now() && PM_SERIES[p.stat]);
  const events = new Set();
  for (const p of pending) {
    events.add(marketEvent_(p,p.stat));
    if (p.shopping && p.shopping.double && p.stat === 'REB' && p.side === 'Over') events.add(marketEvent_(p,'DD'));
  }
  events.delete(null);
  const names = [...events], books = new Map();
  // No player names, prices, card IDs, or workbook credentials go to the exchange.
  for (let i=0; i<names.length; i+=8) {
    const group = names.slice(i,i+8);
    try {
      const responses = UrlFetchApp.fetchAll(group.map(event=>({
        url:'https://api.elections.kalshi.com/trade-api/v2/markets?limit=1000&status=open&event_ticker='+event,
        muteHttpExceptions:true
      })));
      responses.forEach((res,j)=>{
        if (res.getResponseCode() !== 200) return;
        const page = JSON.parse(res.getContentText());
        if (Array.isArray(page.markets) && !page.cursor) books.set(group[j],page.markets);
      });
    } catch (_) { /* Record unavailability for this batch. Primary card delivery can proceed. */ }
  }
  const checked = new Date(), rows = new Map();
  for (const p of pending) {
    if (new Date(p.start) <= checked) continue;
    const event = marketEvent_(p,p.stat);
    const asks = marketAsks_(books.get(event),event,p.player,p.side,p.stat);
    const doubleEvent = marketEvent_(p,'DD');
    const dd = marketAsks_(books.get(doubleEvent),doubleEvent,p.player,'Over','DD').get(0);
    rows.set(p.id,shopAlternate_(p,asks,dd));
  }
  return rows;
}

function doPost(e) {
  const json = value => ContentService.createTextOutput(JSON.stringify(value)).setMimeType(ContentService.MimeType.JSON);
  let lock;
  try {
    const data = JSON.parse(e.postData.contents);
    const props = PropertiesService.getUserProperties();
    if (!data.key || data.key !== props.getProperty('writeKey') ||
        !/^\d{4}-\d{2}-\d{2}$/.test(data.day) || !['check','append'].includes(data.action)) throw Error();
    if (data.action === 'append' && (!Array.isArray(data.rows) || data.rows.length>400)) throw Error();
    const marketRows = data.action === 'append' && data.rows.length ? fetchCardMarkets_(data.rows) : new Map();
    const book = SpreadsheetApp.openById(props.getProperty('sheetId'));
    lock = LockService.getScriptLock();
    lock.waitLock(16000);
    let sheet = book.getSheetByName(data.day) && ensureDay_(book, data.day);
    const known = new Set(sheet && sheet.getLastRow() > 1 ?
      sheet.getRange(2, HEADERS.length, sheet.getLastRow()-1, 1).getValues().flat() : []);
    if (data.action === 'check') {
      if (sheet) sortDay_(sheet);
      return json({ok:true, confidenceEdits:true, existing:data.ids.filter(id=>known.has(id))});
    }
    if (!Array.isArray(data.rows) || data.rows.length > 400) throw Error();
    // One range write with its identity column included, under the same lock as the read.
    const stamp = iso => serialTime_(Utilities.formatDate(new Date(iso), EASTERN, 'yyyy-MM-dd HH:mm:ss'));
    const safe = value => /^[=+@-]/.test(value) ? "'" + value : value;
    const values = [];
    const edits=data.edits || [];
    if (!Array.isArray(edits) || edits.length>400) throw Error();
    const changes=[];
    const seenEdits=new Set();
    const prior=edits.length && sheet && sheet.getLastRow()>1
      ? sheet.getRange(2,5,sheet.getLastRow()-1,10).getValues() : [];
    const byId=new Map(prior.map((row,i)=>[row[9],{row,index:i+2}]));
    for (const edit of edits) {
      if (!/^[a-f0-9]{64}$/.test(edit.id) || ![5,10,15].includes(edit.cents) || seenEdits.has(edit.id)) throw Error();
      seenEdits.add(edit.id);
      const stored=byId.get(edit.id);
      if (!stored) {
        if (!data.rows.some(r=>r.id===edit.id)) throw Error();
        continue;
      }
      const [limit,prob]=playLimit_(stored.row[0],edit.cents);
      if (limit==='') throw Error('Original price is unavailable.');
      if (limit!==stored.row[1] || Math.abs(prob-Number(stored.row[2]))>1e-10) changes.push([stored.index,[limit,prob]]);
    }
    for (const r of data.rows) {
      if (!/^[a-f0-9]{64}$/.test(r.id) || r.day !== data.day) throw Error();
      if (known.has(r.id)) continue;
      if (r.book !== 'FanDuel' || !Number.isInteger(r.odds) || Math.abs(r.odds)<100 || !Number.isInteger(r.worst) ||
          !Number.isFinite(r.breakeven) || r.breakeven <= 0 || r.breakeven >= 1 ||
          !['Over','Under'].includes(r.side) || !Number.isFinite(r.line) ||
          !['PTS','REB','AST','3PM','PRA','BLK','STL'].includes(r.stat)) throw Error();
      const offer = marketRows.get(r.id) || shopAlternate_(r,new Map(),null);
      values.push([stamp(r.start),safe(r.game),safe(r.player),r.side+' '+r.line+' '+r.stat,
        r.odds,r.worst,r.breakeven,altLabel_({...r,alt:offer.alt}),offer.label,'',stamp(r.quoted),stamp(r.sent),'',r.id]);
      known.add(r.id);
    }
    for (const [index,terms] of changes) sheet.getRange(index,6,1,2).setValues([terms]);
    if (values.length) {
      sheet = sheet || ensureDay_(book, data.day);
      const first = sheet.getLastRow()+1;
      const last = first + values.length - 1;
      if (last > sheet.getMaxRows()) sheet.insertRowsAfter(sheet.getMaxRows(), last - sheet.getMaxRows());
      for (const col of [5,6]) sheet.getRange(first,col,values.length,1).setNumberFormat('+0;-0;0');
      for (const col of [7]) sheet.getRange(first,col,values.length,1).setNumberFormat('0.0%');
      formatTimes_(sheet, first, values);
      sheet.getRange(first,1,values.length,HEADERS.length).setValues(values);
      SpreadsheetApp.flush();
    }
    if (sheet) sortDay_(sheet);
    return json({ok:true, added:values.length, updated:changes.length});
  } catch (_) {
    return json({ok:false});
  } finally {
    if (lock && lock.hasLock()) lock.releaseLock();
  }
}
