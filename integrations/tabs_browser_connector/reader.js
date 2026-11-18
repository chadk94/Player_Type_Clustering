/* Read the rendered projections only. No network calls, clicks, cookies or page state. */
function tabsSnapshot(doc, now = new Date()) {
    const text = doc.body.innerText;
    if (text.includes('No projections yet'))
        throw Error('Tabs has not published this slate yet. Its projections page lists the release time; minutes comparisons remain unchecked.');
    if (!text.includes('No changes yet: these are Tabs numbers'))
        throw Error('Tabs has personal edits, or the projections are not loaded. Use an unedited Tabs view for an independent comparison.');
    if (doc.querySelector('#sd')?.checked)
        throw Error('Turn off Showdown CPT in Tabs before comparing minutes.');
    const chips = doc.querySelector('#scenChips');
    if (!chips) throw Error('The Tabs scenario controls have changed; comparison is unchecked.');
    const scenario = chips.innerText.replace(/[×✕]/g, '').replace(/\s+/g, ' ').trim() || 'Default';
    const months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'];
    const clock = text.match(/\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+(\d{1,2})\s+(\d{4})\s+(\d{1,2}):(\d{2})\s*(AM|PM)\b/);
    const slate = text.match(/\b(?:MON|TUE|WED|THU|FRI|SAT|SUN),\s*(\d{1,2})\/(\d{1,2})\s*·/);
    const published = text.match(/Last updated\s+(\d{1,2}):(\d{2})\s*(AM|PM)\s+(MST|MDT|EST|EDT|PST|PDT|CST|CDT)\b/i);
    if (!clock || !slate || !published) throw Error('Could not verify the Tabs slate date and update time.');
    const offsets = {MST:7,MDT:6,EST:5,EDT:4,PST:8,PDT:7,CST:6,CDT:5};
    const hour = (h, ampm) => Number(h)%12 + (ampm.toUpperCase()==='PM' ? 12 : 0);
    const year = Number(clock[3]), month = months.indexOf(clock[1]), day = Number(clock[2]);
    const localDay = Date.UTC(year, month, day);
    // Match the selected slate across New Year rather than silently assigning today's date.
    const candidates = [year-1,year,year+1].map(y=>Date.UTC(y,Number(slate[1])-1,Number(slate[2])));
    candidates.sort((a,b)=>Math.abs(a-localDay)-Math.abs(b-localDay));
    const slateDate = new Date(candidates[0]).toISOString().slice(0,10);
    const offset = offsets[published[4].toUpperCase()];
    let updated = localDay + (hour(published[1],published[3])+offset)*3600000 + Number(published[2])*60000;
    // A previous day's last update can remain visible after local midnight.
    if (hour(published[1],published[3])*60+Number(published[2]) > hour(clock[4],clock[6])*60+Number(clock[5])) updated -= 86400000;
    if (now.getTime()-updated < 0 || now.getTime()-updated > 7200000)
        throw Error('Tabs update is over two hours old or its timestamp cannot be verified.');
    const table = [...doc.querySelectorAll('table')].find(t=>{
        const head=[...t.querySelectorAll('tr')[0]?.querySelectorAll('th')||[]].map(c=>c.innerText.trim());
        return head.includes('PLAYER') && head.includes('TEAM') && head.includes('OPP') && head.some(h=>/^MIN\b/.test(h));
    });
    if (!table) throw Error('The Tabs minutes table is not available.');
    const allRows=[...table.querySelectorAll('tr')];
    const headers=[...allRows[0].querySelectorAll('th')].map(c=>c.innerText.trim());
    const col = name=>headers.indexOf(name), rows=[];
    for (const tr of allRows.slice(1)) {
        if (!tr.getClientRects().length) continue;
        const cells=[...tr.querySelectorAll('td')];
        if (!cells.length) continue;
        const label=cells[col('PLAYER')]?.innerText.trim();
        const input=tr.querySelector('input[aria-label^="Minutes for "]');
        // Status badges can touch the surname in innerText ("MobleyOUT").
        // Read the explicit badge and name separately, excluding photo initials.
        const badge=cells[col('PLAYER')]?.querySelector('.tag.out');
        const out=!input && (badge?.innerText.trim()==='OUT' || /\bOUT\b/.test(label || ''));
        // Only explicitly labeled OUT rows get zero. Omitted/unknown rows stay absent.
        if (!input && !out) throw Error('A Tabs player row has no verifiable minutes.');
        const playerName=cells[col('PLAYER')]?.querySelector('[data-pn]')?.innerText.trim();
        const Player=input ? input.getAttribute('aria-label').slice(12).trim() : playerName || label.split(/\bOUT\b/)[0].trim();
        const raw=input ? input.value.trim() : '0';
        const minutes=Number(raw);
        const Team=cells[col('TEAM')]?.innerText.trim(), Opp=cells[col('OPP')]?.innerText.trim();
        if (!Player || Player.length>100 || !raw || !Number.isFinite(minutes) || minutes<0 || minutes>48 || !/^[A-Z]{2,3}$/.test(Team||'') || !/^[A-Z]{2,3}$/.test(Opp||''))
            throw Error('A Tabs minutes row could not be verified.');
        rows.push({Player,Team,Opp,minutes,status:out?'Out':'Projected'});
    }
    if (!rows.length || rows.length>600) throw Error('No usable players are displayed in Tabs. Clear its player/team filters.');
    const scope=doc.querySelector('#game');
    return {schema:1,source:'TabsNBA',source_url:'https://app.tabsnba.com/projections',
        slate_date:slateDate,scenario,source_updated_at:new Date(updated).toISOString(),
        observed_at:now.toISOString(),manual_edits:false,
        scope:scope?.options[scope.selectedIndex]?.textContent?.trim()||'Displayed players',rows};
}

if (typeof chrome !== 'undefined' && chrome.runtime) chrome.runtime.onMessage.addListener((message,_sender,reply)=>{
    if (message?.type!=='tabs-minutes-capture-v1') return;
    try {reply({ok:true,snapshot:tabsSnapshot(document)});}
    catch(error) {reply({ok:false,error:String(error.message).slice(0,250)});}
});
