export default function({data, parentElement}) {
    const root = parentElement.querySelector('section');
    const make = (tag, text) => {
        const el = document.createElement(tag);
        if (text != null) el.textContent = text;
        return el;
    };
    root.replaceChildren();
    const scroll = make('div');
    scroll.className = 'edges-scroll';
    const table = make('table');
    table.setAttribute('aria-label', data.team + ' sportsbook edges');
    const headers = table.createTHead().insertRow();
    const body = table.createTBody();
    scroll.append(table);
    const tip = make('div');
    tip.id = 'rating-detail-' + data.team;
    tip.setAttribute('role', 'tooltip');
    tip.setAttribute('popover', 'manual');
    root.append(scroll, tip);
    let active;
    function hide() {
        if (active) active.removeAttribute('aria-describedby');
        active = null;
        if (tip.matches(':popover-open')) tip.hidePopover();
    }
    function show(button, text) {
        hide();
        active = button;
        tip.textContent = text;
        button.setAttribute('aria-describedby', tip.id);
        tip.showPopover();
        const b = button.getBoundingClientRect(), t = tip.getBoundingClientRect();
        tip.style.left = Math.max(8, Math.min(b.left, innerWidth - t.width - 8)) + 'px';
        tip.style.top = (b.top >= t.height + 12 ? b.top - t.height - 6 :
            Math.max(8, Math.min(b.bottom + 6, innerHeight - t.height - 8))) + 'px';
    }
    let sort = data.columns.indexOf(root.dataset.sort || 'Rank');
    if (sort < 0) sort = 0;
    let direction = Number(root.dataset.direction || 1);
    const note = make('div');
    note.setAttribute('role', 'status');
    root.append(note);
    const boxes = new Map();
    function updateChecks() {
        try {
            for (const [input, pick] of boxes) {
                input.checked = !!savedPlay(localStorage, pick.id);
                input.closest('tr').classList.toggle('saved-play', input.checked);
            }
        } catch { note.textContent = 'Allow site storage to save plays in this browser.'; }
    }
    function render() {
        boxes.clear();
        hide();
        body.replaceChildren();
        for (const [i, th] of [...headers.cells].slice(1).entries()) {
            th.setAttribute('aria-sort', i !== sort ? 'none' : direction === 1 ? 'ascending' : 'descending');
        }
        const rows = [...data.rows].sort((a, b) => {
            const x = a.values[sort], y = b.values[sort];
            if (x == null || y == null) return Number(x == null) - Number(y == null);
            return direction * (typeof x === 'number' && typeof y === 'number' ? x - y :
                String(x).localeCompare(String(y), undefined, {numeric: true}));
        });
        for (const row of rows) {
            const tr = body.insertRow();
            const saveCell = tr.insertCell();
            const pick = row.pick;
            if (pick) {
                const input = make('input'); input.type='checkbox';
                input.setAttribute('aria-label', 'Save '+pick.player+' '+pick.side+' '+pick.line+' '+pick.stat+" to Today's Plays");
                input.onchange = () => {
                    try {
                        setPick(localStorage, pick, input.checked);
                        note.textContent = pick.player + (input.checked ? " saved to Today's Plays" : " removed from Today's Plays") + ' · '+pick.day;
                        notifyPicks(pick.day);
                    } catch { input.checked=!input.checked; updateChecks(); note.textContent='Could not save this change. Check your browser storage.'; }
                };
                boxes.set(input,pick);saveCell.append(input);
            }
            for (const [i, label] of row.labels.entries()) {
                const td = tr.insertCell();
                if (row.styles[i]) td.style.cssText = row.styles[i];
                if (data.columns[i] === 'Matchup rating') {
                    const button = make('button', label);
                    button.type = 'button';
                    button.setAttribute('aria-label', row.labels[data.columns.indexOf('Player')] + ' ' +
                        row.labels[data.columns.indexOf('Stat')] + ' matchup rating ' + label);
                    button.onpointerenter = e => { if (e.pointerType !== 'touch') show(button, row.detail); };
                    button.onfocus = button.onclick = () => show(button, row.detail);
                    button.onpointerleave = button.onblur = hide;
                    td.append(button);
                } else {
                    td.textContent = label;
                }
            }
        }
    }
    const saveHeader=make('th','Save');saveHeader.scope='col';headers.append(saveHeader);
    data.columns.forEach((label, i) => {
        const th = make('th'), button = make('button', label);
        th.scope = 'col';
        button.type = 'button';
        button.onclick = () => {
            direction = sort === i ? -direction : 1;
            sort = i;
            root.dataset.sort = label;
            root.dataset.direction = direction;
            render(); updateChecks();
        };
        th.append(button);
        headers.append(th);
    });
    const dismiss = e => { if (e.key === 'Escape') hide(); };
    const outside = e => { if (active && !e.composedPath().includes(active)) hide(); };
    document.addEventListener('keydown', dismiss);
    document.addEventListener('pointerdown', outside, true);
    scroll.onscroll = hide;
    window.addEventListener('scroll', hide, true);
    window.addEventListener('resize', hide);
    render(); updateChecks();
    const unwatch=watchPicks(updateChecks);
    return () => {
        unwatch();
        hide();
        document.removeEventListener('keydown', dismiss);
        document.removeEventListener('pointerdown', outside, true);
        window.removeEventListener('scroll', hide, true);
        window.removeEventListener('resize', hide);
    };
}
