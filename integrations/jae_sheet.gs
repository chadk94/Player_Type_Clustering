// Copy to the Apps Script project bound to the private Jae workbook.
const LEGACY_HEADERS = ['Game time (Denver)', 'Game', 'Player', 'Prop', 'Sportsbook', 'Quoted odds',
  'Worst odds', 'Worst break-even %', 'Quote time (Denver)', 'Sent time (Denver)', 'Partner feedback', 'Play ID'];

const PREVIOUS_HEADERS = ['Game time (ET)', 'Game', 'Player', 'Prop', 'FanDuel odds', 'Worst odds',
  'Worst break-even %', 'Alternate over · FD odds', 'Quote time (ET)', 'Sent time (ET)', 'Partner feedback', 'Play ID'];
const HEADERS = ['Tip-off (ET)', 'Game', 'Player', 'Prop', 'FanDuel odds', 'Play to', 'Break-even %',
  'Alternate over · FD odds', 'Alt play to', 'Alt break-even %', 'Quoted (ET)', 'Sent (ET)', 'Partner feedback', 'Play ID'];
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

function altLimit_(price) {
  if (!Number.isInteger(price) || Math.abs(price) < 100) return ['', ''];
  const ticks = (price > 0 ? price - 100 : price + 100) - 5;
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
    // Use the frozen alternate from this row, including on later retries.
    const match = String(row[7]).match(/·\s*([+-]\d+)\s*$/);
    [row[8], row[9]] = altLimit_(match ? Number(match[1]) : null);
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
  return sheet;
}

function ensureDay_(book, day) {
  let sheet = book.getSheetByName(day);
  if (!sheet) sheet = book.insertSheet(day);
  if (sheet.getLastRow()) {
    const current = sheet.getRange(1, 1, 1, HEADERS.length).getValues()[0];
    const same = (a,b) => JSON.stringify(a) === JSON.stringify(b);
    if (same(current, HEADERS)) return sheet;
    const old = current.slice(0, PREVIOUS_HEADERS.length);
    if (same(old, LEGACY_HEADERS) || same(old, PREVIOUS_HEADERS)) {
      if (same(old, LEGACY_HEADERS)) migrateDay_(sheet, book);
      // Inserting preserves partner feedback cells, notes and the hidden ID.
      sheet.insertColumnsAfter(8, 2);
    } else if (!same(current, [...PREVIOUS_HEADERS.slice(0,8), '', '', ...PREVIOUS_HEADERS.slice(8)])) {
      throw Error('Unrecognized daily card headers.');
    }
    // Also resumes an interrupted upgrade after the columns were inserted.
    upgradeTimesAndAlts_(sheet, book);
  }
  sheet.setFrozenRows(1);
  sheet.setColumnWidths(1, HEADERS.length, 103);
  for (const [col, width] of [[1,100],[2,108],[3,165],[4,140],[8,190],[13,180]]) {
    sheet.setColumnWidth(col, width);
  }
  sheet.hideColumns(HEADERS.length);
  const count = sheet.getMaxRows() - 1;
  if (count > 0) {
    for (const col of [5,6,9]) sheet.getRange(2,col,count,1).setNumberFormat('+0;-0;0');
    for (const col of [7,10]) sheet.getRange(2,col,count,1).setNumberFormat('0.0%');
    sheet.getRange(2,8,count,1).setNumberFormat('@');
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
    let sheet = book.getSheetByName(data.day) && ensureDay_(book, data.day);
    const known = new Set(sheet && sheet.getLastRow() > 1 ?
      sheet.getRange(2, HEADERS.length, sheet.getLastRow()-1, 1).getValues().flat() : []);
    if (data.action === 'check') {
      if (sheet) sortDay_(sheet);
      return json({ok:true, existing:data.ids.filter(id=>known.has(id))});
    }
    if (!Array.isArray(data.rows) || data.rows.length > 400) throw Error();
    // One range write with its identity column included, under the same lock as the read.
    const stamp = iso => serialTime_(Utilities.formatDate(new Date(iso), EASTERN, 'yyyy-MM-dd HH:mm:ss'));
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
        r.odds,r.worst,r.breakeven,altLabel_(r),...altLimit_(r.side === 'Over' && r.alt && r.alt.line > r.line ? r.alt.odds : null),stamp(r.quoted),stamp(r.sent),'',r.id]);
      known.add(r.id);
    }
    if (values.length) {
      sheet = sheet || ensureDay_(book, data.day);
      const first = sheet.getLastRow()+1;
      const last = first + values.length - 1;
      if (last > sheet.getMaxRows()) sheet.insertRowsAfter(sheet.getMaxRows(), last - sheet.getMaxRows());
      for (const col of [5,6,9]) sheet.getRange(first,col,values.length,1).setNumberFormat('+0;-0;0');
      for (const col of [7,10]) sheet.getRange(first,col,values.length,1).setNumberFormat('0.0%');
      formatTimes_(sheet, first, values);
      sheet.getRange(first,1,values.length,HEADERS.length).setValues(values);
      SpreadsheetApp.flush();
    }
    if (sheet) sortDay_(sheet);
    return json({ok:true, added:values.length});
  } catch (_) {
    return json({ok:false});
  } finally {
    if (lock && lock.hasLock()) lock.releaseLock();
  }
}
