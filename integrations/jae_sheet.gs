// Copy to the Apps Script project bound to the private Jae workbook.
const LEGACY_HEADERS = ['Game time (Denver)', 'Game', 'Player', 'Prop', 'Sportsbook', 'Quoted odds',
  'Worst odds', 'Worst break-even %', 'Quote time (Denver)', 'Sent time (Denver)', 'Partner feedback', 'Play ID'];

const HEADERS = ['Game time (ET)', 'Game', 'Player', 'Prop', 'FanDuel odds', 'Worst odds',
  'Worst break-even %', 'Alternate over · FD odds', 'Quote time (ET)', 'Sent time (ET)', 'Partner feedback', 'Play ID'];
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
    for (const date of days) ensureDay_(book, date);
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
  const migrated = [HEADERS.slice(0, 10)];
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

function ensureDay_(book, day) {
  let sheet = book.getSheetByName(day);
  if (!sheet) sheet = book.insertSheet(day);
  const header = sheet.getRange(1, 1, 1, HEADERS.length);
  let layoutChanged = false;
  if (!sheet.getLastRow()) {
    header.setValues([HEADERS]);
    layoutChanged = true;
  } else {
    const current = JSON.stringify(header.getValues()[0]);
    if (current === JSON.stringify(LEGACY_HEADERS)) {
      migrateDay_(sheet, book);
      layoutChanged = true;
    } else if (current !== JSON.stringify(HEADERS)) {
      throw Error('Headers changed.');
    }
  }
  if (layoutChanged || sheet.getFrozenRows() === 0) {
    header.setFontWeight('bold').setBackground('#214b56').setFontColor('#ffffff').setWrap(true);
    sheet.setFrozenRows(1);
    sheet.setColumnWidths(1, HEADERS.length, 105);
    for (const col of [1, 3, 4, 8, 9, 10, 11]) sheet.setColumnWidth(col, 180);
    sheet.hideColumns(HEADERS.length);
    const count = sheet.getMaxRows() - 1;
    if (count > 0) {
      sheet.getRange(2, 5, count, 2).setNumberFormat('+0;-0;0');
      sheet.getRange(2, 7, count, 1).setNumberFormat('0.00%');
      sheet.getRange(2, 8, count, 1).setNumberFormat('@');
    }
  }
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
    const stamp = iso => easternStamp_(new Date(iso));
    const safe = value => /^[=+@-]/.test(value) ? "'" + value : value;
    const values = [];
    for (const r of data.rows) {
      if (!/^[a-f0-9]{64}$/.test(r.id) || r.day !== data.day) throw Error();
      if (known.has(r.id)) continue;
      if (r.book !== 'FanDuel' || !Number.isInteger(r.odds) || Math.abs(r.odds)<100 || !Number.isInteger(r.worst) ||
          !Number.isFinite(r.breakeven) || r.breakeven <= 0 || r.breakeven >= 1 ||
          !['Over','Under'].includes(r.side) || !Number.isFinite(r.line) ||
          !['PTS','REB','AST','3PM','PRA','BLK','STL'].includes(r.stat)) throw Error();
      values.push([stamp(r.start),safe(r.game),safe(r.player),r.side+' '+r.line+' '+r.stat,
        r.odds,r.worst,r.breakeven,altLabel_(r),stamp(r.quoted),stamp(r.sent),'',r.id]);
      known.add(r.id);
    }
    if (values.length) {
      sheet = ensureDay_(book, data.day);
      const first = sheet.getLastRow()+1;
      const last = first + values.length - 1;
      if (last > sheet.getMaxRows()) sheet.insertRowsAfter(sheet.getMaxRows(), last - sheet.getMaxRows());
      sheet.getRange(first,5,values.length,2).setNumberFormat('+0;-0;0');
      sheet.getRange(first,7,values.length,1).setNumberFormat('0.00%');
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
