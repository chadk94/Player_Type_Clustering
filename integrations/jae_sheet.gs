// Copy to the Apps Script project bound to the private Jae workbook.
const HEADERS = ['Game time (Denver)', 'Game', 'Player', 'Prop', 'Sportsbook', 'Quoted odds',
  'Worst odds', 'Worst break-even %', 'Quote time (Denver)', 'Sent time (Denver)', 'Partner feedback', 'Play ID'];

function onOpen() {
  SpreadsheetApp.getUi().createMenu('Jae card')
    .addItem('Set up daily tabs', 'setupCard')
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
  ensureDay_(book, day);
  book.toast('Daily card ready. Future dates are created when you send their first plays.');
}

function showConnection() {
  const url = ScriptApp.getService().getUrl();
  const key = PropertiesService.getUserProperties().getProperty('writeKey');
  if (!url || !key) {
    SpreadsheetApp.getUi().alert('Run Set up daily tabs and deploy this project as a web app first.');
    return;
  }
  SpreadsheetApp.getUi().alert('Private dashboard connection', url + '#' + key,
    SpreadsheetApp.getUi().ButtonSet.OK);
}

function ensureDay_(book, day) {
  let sheet = book.getSheetByName(day);
  if (!sheet) sheet = book.insertSheet(day);
  if (!sheet.getLastRow()) sheet.getRange(1, 1, 1, HEADERS.length).setValues([HEADERS]);
  const header = sheet.getRange(1, 1, 1, HEADERS.length);
  if (JSON.stringify(header.getValues()[0]) !== JSON.stringify(HEADERS)) throw Error('Headers changed.');
  if (sheet.getFrozenRows() === 0) {
    header.setFontWeight('bold').setBackground('#214b56').setFontColor('#ffffff').setWrap(true);
    sheet.setFrozenRows(1);
    sheet.setColumnWidths(1, HEADERS.length, 105);
    for (const col of [1, 3, 4, 9, 10, 11]) sheet.setColumnWidth(col, 165);
    sheet.hideColumns(HEADERS.length);
  }
  return sheet;
}

function doPost(e) {
  const json = value => ContentService.createTextOutput(JSON.stringify(value)).setMimeType(ContentService.MimeType.JSON);
  let lock;
  try {
    const data = JSON.parse(e.postData.contents);
    const props = PropertiesService.getUserProperties();
    if (!data.key || data.key !== props.getProperty('writeKey') ||
        !/^\d{4}-\d{2}-\d{2}$/.test(data.day) || !['check','append'].includes(data.action)) throw Error();
    const book = SpreadsheetApp.openById(props.getProperty('sheetId'));
    lock = LockService.getScriptLock();
    lock.waitLock(16000);
    let sheet = book.getSheetByName(data.day);
    const known = new Set(sheet && sheet.getLastRow() > 1 ?
      sheet.getRange(2, HEADERS.length, sheet.getLastRow()-1, 1).getValues().flat() : []);
    if (data.action === 'check') return json({ok:true, existing:data.ids.filter(id=>known.has(id))});
    if (!Array.isArray(data.rows) || data.rows.length > 400) throw Error();
    // One range write with its identity column included, under the same lock as the read.
    const stamp = iso => Utilities.formatDate(new Date(iso), 'America/Denver', 'yyyy-MM-dd HH:mm:ss');
    const safe = value => /^[=+@-]/.test(value) ? "'" + value : value;
    const values = [];
    for (const r of data.rows) {
      if (!/^[a-f0-9]{64}$/.test(r.id) || r.day !== data.day) throw Error();
      if (known.has(r.id)) continue;
      if (!Number.isInteger(r.odds) || Math.abs(r.odds)<100 || !Number.isInteger(r.worst) ||
          !Number.isFinite(r.breakeven) || r.breakeven <= 0 || r.breakeven >= 1 ||
          !['Over','Under'].includes(r.side) || !Number.isFinite(r.line) ||
          !['PTS','REB','AST','3PM','PRA','BLK','STL'].includes(r.stat)) throw Error();
      values.push([stamp(r.start),safe(r.game),safe(r.player),r.side+' '+r.line+' '+r.stat,safe(r.book),
        r.odds,r.worst,r.breakeven,stamp(r.quoted),stamp(r.sent),'',r.id]);
      known.add(r.id);
    }
    if (values.length) {
      sheet = ensureDay_(book, data.day);
      const first = sheet.getLastRow()+1;
      const last = first + values.length - 1;
      if (last > sheet.getMaxRows()) sheet.insertRowsAfter(sheet.getMaxRows(), last - sheet.getMaxRows());
      sheet.getRange(first,6,values.length,2).setNumberFormat('+0;-0;0');
      sheet.getRange(first,8,values.length,1).setNumberFormat('0.00%');
      sheet.getRange(first,1,values.length,HEADERS.length).setValues(values);
      SpreadsheetApp.flush();
    }
    return json({ok:true, added:values.length});
  } catch (_) {
    return json({ok:false});
  } finally {
    if (lock && lock.hasLock()) lock.releaseLock();
  }
}
