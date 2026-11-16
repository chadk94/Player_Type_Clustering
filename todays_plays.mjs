export function googleDocsTable(day, picks, tiers) {
    const headings = ['Player / Game', 'Prop / Sportsbook', 'Projected', 'Edge', 'Status', 'Confidence', 'Partner feedback'];
    const rows = [headings];
    for (const status of ['Review', 'Confirmed']) {
        for (const p of picks.filter(p => p.day === day && p.status === status)) {
            const odds = (p.odds > 0 ? '+' : '') + p.odds;
            const cents = p.confidence_cents ?? 5;
            rows.push([`${p.player}\n${p.game}`, `${p.side} ${p.line} ${p.stat} (${odds})\n${p.book}`,
                p.projected.toFixed(2), p.edge.toFixed(2) + '%', status === 'Review' ? 'In review' : status,
                tiers.get(cents)+' · '+cents+'¢', '']);
        }
    }
    const escape = value => value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
    const title = "Today's Plays · " + day;
    const htmlRows = rows.map((row, i) => {
        const tag = i === 0 ? 'th' : 'td';
        const color = i === 0 ? 'background:#e7eff4;' : '';
        return '<tr>' + row.map(value => `<${tag} style="border:1px solid #a5b3be;padding:4px;text-align:left;vertical-align:top;${color}">${escape(value).replaceAll('\n', '<br>') || '<br>'}</${tag}>`).join('') + '</tr>';
    });
    return {
        html: '<p><b>' + escape(title) + '</b></p><table style="border-collapse:collapse">' + htmlRows.join('') + '</table>',
        text: [title, ...rows.map(row => row.map(value => value.replaceAll('\n', ' · ').replaceAll('\t', ' ')).join('\t'))].join('\n'),
    };
}

export default function({data, parentElement, setStateValue}) {
    const root = parentElement.querySelector('section');
    const ui = parentElement.playsUI ||= {day: data.today, pending: null};
    const make = (tag, text) => {
        const e = document.createElement(tag);
        if (text != null) e.textContent = text;
        return e;
    };
    const button = (label, click) => {
        const b = make('button', label);
        b.type = 'button';
        b.onclick = click;
        return b;
    };
    const stateFor = day => JSON.parse(localStorage.getItem(SHARE_PREFIX + day) || '{}');
    const store = (day, state) => localStorage.setItem(SHARE_PREFIX + day, JSON.stringify(state));
    const storageFailure = () => {
        root.replaceChildren(make('p', 'This browser could not read or save your shortlist. Allow site storage and reopen Today’s Plays.'));
    };
    function send(type, picks) {
        try {
            const saved = stateFor(ui.day);
            if (!saved.owner) {
                saved.owner = Array.from(crypto.getRandomValues(new Uint8Array(32)), b => b.toString(16).padStart(2,'0')).join('');
                store(ui.day, saved);
            }
            ui.pending = crypto.randomUUID();
            ui.message = '';
            setStateValue('action', {type, nonce: ui.pending, owner:saved.owner, day:ui.day,
                                     picks: type === 'publish' ? picks : undefined});
            draw();
        } catch { storageFailure(); }
    }
    const result = data.result;
    if (result && result.nonce !== parentElement.seenPlayReply) {
        parentElement.seenPlayReply = result.nonce;
        ui.pending = null;
        ui.message = result.error || '';
        if (result.url) {
            try {
                const saved = stateFor(result.day);
                // Retain replies already read on this device if the review host restarts.
                const notes = new Map([...(saved.notes || []), ...result.notes].map(n=>[n.id,n]));
                store(result.day, {...saved, url:result.url, notes:[...notes.values()]});
            } catch { storageFailure(); return; }
        }
    }
    const connectionKey = 'nba-props.jae.connection.v1';
    const jaeReply = data.jae;
    if (jaeReply && jaeReply.nonce === ui.jaePending) {
        if (jaeReply.connected) localStorage.setItem(connectionKey, ui.jaeCandidate);
        ui.jaeCandidate = null;
        ui.jaePending = null;
        ui.jaeFeedback = jaeReply;
    }
    function requestJae(type, connection, picks) {
        ui.jaePending = crypto.randomUUID();
        ui.jaeFeedback = null;
        if (type === 'connect') ui.jaeCandidate = connection;
        setStateValue('jae', {type, nonce:ui.jaePending, day:ui.day, connection, picks});
        draw();
    }
    function renderJae(confirmed, hasPicks) {
        const connection = localStorage.getItem(connectionKey);
        const bar = make('div'); bar.className = 'tools';
        const settings = make('details');
        settings.append(make('summary', connection ? 'Jae sheet connection' : 'Connect Jae’s Google Sheet'));
        const instructions = make('p', 'In the prepared Google Sheet, choose Jae card → Show dashboard connection. Paste it here once per browser.');
        const input = make('input'); input.type = 'password'; input.autocomplete = 'off';
        input.placeholder = 'Private sheet connection'; input.setAttribute('aria-label',input.placeholder);
        const connect = button('Connect sheet', () => requestJae('connect', input.value.trim(), []));
        connect.disabled = !!ui.jaePending;
        settings.append(instructions,input,connect);
        if (hasPicks) {
            let action;
            if (!confirmed.length) {
                action = button('Choose plays for Jae', () => root.querySelector('.jae-pick')?.focus());
                bar.append(action, make('span', 'Check For Jae beside each play you want to send.'));
            } else if (!connection) {
                action = button('Connect sheet to send', () => {
                    settings.open = true;
                    input.focus();
                });
                bar.append(action, make('span', 'Connect Jae’s Google Sheet in this browser.'));
            } else {
                const count = confirmed.length;
                action = button(`Send ${count} confirmed ${count === 1 ? 'play' : 'plays'} to Jae`, () =>
                    requestJae('send', connection, confirmed));
                action.title = 'Your confidence sets the main play-to price from the current FanDuel quote. Alternate overs have a 5¢ limit.';
                action.className = 'jae-send';
                bar.append(action);
            }
            action.disabled = !!ui.jaePending;
        }
        root.append(bar,settings);
        if (ui.jaePending || ui.jaeFeedback?.day === ui.day) {
            const info = make('div'); info.setAttribute('role','status');
            info.append(make('p', ui.jaePending ? 'Checking Jae’s sheet…' : ui.jaeFeedback.error || ui.jaeFeedback.message));
            for (const reason of ui.jaeFeedback?.skipped || []) info.append(make('p',reason));
            root.append(info);
        }
    }
    const workbook = data.workbook;
    const exportReady = workbook && workbook.nonce === ui.exportNonce && workbook.day === ui.day;
    if (exportReady) {
        ui.exportNonce = null;
        ui.workbook = workbook;
    }
    function resetExport() {
        ui.exportNonce = null;
        ui.workbook = null;
    }
    function draw() {
        try { render(); } catch { storageFailure(); }
    }
    function render() {
        root.replaceChildren();
        const picks = allPicks(localStorage);
        const days = [...new Set([data.today, ui.day, ...picks.map(p => p.day),
            ...Object.keys(localStorage).filter(k=>k.startsWith(SHARE_PREFIX)).map(k=>k.slice(SHARE_PREFIX.length))])].sort().reverse();
        const bar = make('div'); bar.className='tools';
        const dateLabel = make('label','Slate ');
        const dateSelect = make('select');
        for (const day of days) dateSelect.add(new Option(day===data.today ? day+' · Today' : day, day));
        dateSelect.value = ui.day;
        dateSelect.onchange = () => { ui.day=dateSelect.value; ui.message=''; resetExport(); draw(); };
        dateLabel.append(dateSelect); bar.append(dateLabel); root.append(bar);
        const rows = picks.filter(p=>p.day===ui.day).sort((a,b)=>
            a.game.localeCompare(b.game) || b.edge-a.edge);
        const review = rows.filter(p=>p.status==='Review');
        const status = make('p', rows.length ? review.length+' in review · '+(rows.length-review.length)+' confirmed'
            : 'Choose a confidence level beside a prop in Top Sportsbook Edges to save it here.');
        root.append(status);
        renderJae(rows.filter(p => p.status === 'Confirmed'), rows.length > 0);
        if (rows.length) {
            root.append(make('p','Confidence sets how far past the current FanDuel price Jae can play the main line. Alternate overs have a 5¢ limit.'));
            const scroll=make('div'); scroll.className='play-table';
            const table=make('table'); table.setAttribute('aria-label','Saved player props');
            const header=table.createTHead().insertRow();
            for (const title of ['For Jae','Confidence','Player','Game','Prop','Odds','Sportsbook','Projection','Edge','Actions']) {
                const th=make('th',title); th.scope='col'; header.append(th);
            }
            const body=table.createTBody();
            for (const p of rows) {
                const tr=body.insertRow();
                const prop=p.side+' '+p.line+' '+p.stat;
                const check=make('input');
                check.type='checkbox'; check.className='jae-pick';
                check.checked=p.status==='Confirmed';
                check.dataset.pickId=p.id;
                check.setAttribute('aria-label','For Jae: '+p.player+' '+prop+' ('+p.book+')');
                check.onchange=()=>{
                    try {
                        setPickStatus(localStorage,p.id,check.checked?'Confirmed':'Review');
                        notifyPicks(p.day);
                        root.querySelector(`.jae-pick[data-pick-id="${CSS.escape(p.id)}"]`)?.focus({preventScroll:true});
                    } catch { storageFailure(); }
                };
                tr.insertCell().append(check);
                const confidence=confidenceSelect('Confidence for '+p.player+' '+prop+' ('+p.book+')', false);
                confidence.title='Price movement allowed from the main FanDuel quote when sent';
                confidence.value=String(p.confidence_cents ?? 5);
                confidence.dataset.pickId=p.id;
                confidence.onchange=()=>{
                    try {
                        saveConfidence(localStorage,p,Number(confidence.value));
                        notifyPicks(p.day);
                        root.querySelector(`select[data-pick-id="${CSS.escape(p.id)}"]`)?.focus({preventScroll:true});
                    } catch { storageFailure(); }
                };
                tr.insertCell().append(confidence);
                for (const text of [p.player,p.game,prop,(p.odds>0?'+':'')+p.odds,p.book,p.projected.toFixed(2),p.edge.toFixed(2)+'%']) {
                    tr.insertCell().textContent=text;
                }
                tr.title=p.quoted ? 'Quote updated '+p.quoted : 'Saved sportsbook quote';
                const actions=tr.insertCell();
                const remove=button('Remove',()=>{
                    try {
                        setPick(localStorage,p,false);
                        notifyPicks(p.day);
                    } catch { storageFailure(); }
                });
                remove.setAttribute('aria-label','Remove '+p.player+' '+prop);
                actions.append(remove);
            }
            scroll.append(table);root.append(scroll);

            const docs = make('div');
            docs.className = 'tools';
            const feedback = make('div');
            feedback.setAttribute('role', 'status');
            const copy = button('Copy table for Google Docs', async () => {
                const {html, text} = googleDocsTable(ui.day, rows, CONFIDENCE_TIERS);
                feedback.replaceChildren();
                try {
                    await navigator.clipboard.write([new ClipboardItem({
                        'text/html': new Blob([html], {type: 'text/html'}),
                        'text/plain': new Blob([text], {type: 'text/plain'}),
                    })]);
                    feedback.textContent = 'Table copied. Paste into Google Docs and give partners Editor access to type in the feedback column.';
                } catch {
                    const preview = make('div');
                    preview.className = 'play-table';
                    preview.innerHTML = html;
                    feedback.append(make('p', 'Copy the table below with Ctrl+C or ⌘C, then paste it into your document.'), preview);
                    const range = document.createRange();
                    range.selectNodeContents(preview);
                    const selection = window.getSelection();
                    selection?.removeAllRanges();
                    selection?.addRange(range);
                }
            });
            const newDoc = make('a', 'New Google Doc ↗');
            newDoc.href = 'https://docs.google.com/create';
            newDoc.target = '_blank';
            newDoc.rel = 'noopener noreferrer';
            let excel;
            if (ui.workbook?.base64) {
                excel = make('a', 'Download Excel');
                excel.download = 'Todays-Plays-' + ui.day + '.xlsx';
                excel.href = 'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,' + ui.workbook.base64;
                feedback.textContent = 'Workbook ready. If the download didn’t start, click Download Excel.';
            } else {
                excel = button(ui.exportNonce ? 'Preparing Excel…' : 'Download Excel', () => {
                    ui.exportNonce = crypto.randomUUID();
                    ui.workbook = null;
                    setStateValue('excel', {nonce: ui.exportNonce, day: ui.day, picks: rows});
                    draw();
                });
                excel.disabled = !!ui.exportNonce;
            }
            if (ui.workbook?.error) feedback.textContent = ui.workbook.error;
            docs.append(copy, newDoc, excel);
            root.append(docs, feedback);
        }
        const saved=stateFor(ui.day);
        if (rows.length || saved.url) {
            const share=make('div');share.className='tools';
            const publish=button(saved.url?'Update shared list':'Share app review link',()=>send('publish',rows));
            publish.disabled=!!ui.pending;share.append(publish);
            if (saved.url) {
                const open=make('a','Open partner review');open.href=saved.url;open.target='_blank';open.rel='noopener noreferrer';share.append(open);
                const copy=button('Copy review link',async()=>{
                    try { await navigator.clipboard.writeText(saved.url); copy.textContent='Copied'; }
                    catch {
                        const link=make('input');link.value=saved.url;link.readOnly=true;link.setAttribute('aria-label','Partner review link');
                        share.append(link);link.select();
                    }
                });share.append(copy);
                const refresh=button('Refresh feedback',()=>send('feedback',rows));refresh.disabled=!!ui.pending;share.append(refresh);
            }
            root.append(share,make('p',saved.url?'Partners see the last shared version. Update it after editing picks. The link lasts seven days from the latest update.':
                'Share this slate to let anyone with the link view the saved prices and comment. Only you can confirm or remove picks.'));
            if (saved.notes?.length) {
                const visible = saved.notes.filter(n=>rows.some(p=>p.id===n.play));
                if (visible.length) root.append(make('h4','Partner feedback'));
                for (const item of visible) {
                    const p=rows.find(p=>p.id===item.play);
                    root.append(make('p',item.author+' · '+p.player+' '+p.side+' '+p.line+' '+p.stat+' · '+item.note));
                }
            } else if (saved.url) root.append(make('p','No partner feedback yet.'));
        }
        if(ui.message || ui.pending) {
            const message=make('p',ui.message || 'Updating shared review…');message.setAttribute('role','status');root.append(message);
        }
    }
    const onPicks = e => {
        // Follow the slate just selected on Edges, including tomorrow's games.
        if (e?.detail && typeof e.detail==='string') ui.day=e.detail;
        resetExport();
        draw();
    };
    const cleanup=watchPicks(onPicks);
    const onConnection = e => { if (e.key === connectionKey) draw(); };
    window.addEventListener('storage', onConnection);
    draw();
    if (exportReady && workbook.base64) root.querySelector('a[download]')?.click();
    return () => { cleanup(); window.removeEventListener('storage', onConnection); };
}
