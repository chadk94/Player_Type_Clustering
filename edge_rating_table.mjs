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
    let active, closeTimer, restoringFocus = false;
    function hide() {
        clearTimeout(closeTimer);
        if (active) {
            active.removeAttribute('aria-describedby');
            if (active.hasAttribute('aria-expanded')) active.setAttribute('aria-expanded', 'false');
        }
        active = null;
        if (tip.matches(':popover-open')) tip.hidePopover();
    }
    function position() {
        if (!active) return;
        const b = active.getBoundingClientRect(), t = tip.getBoundingClientRect();
        tip.style.left = Math.max(8, Math.min(b.left, innerWidth - t.width - 8)) + 'px';
        tip.style.top = (b.top >= t.height + 12 ? b.top - t.height - 6 :
            Math.max(8, Math.min(b.bottom + 6, innerHeight - t.height - 8))) + 'px';
    }
    function show(button, text, technical) {
        clearTimeout(closeTimer);
        if (active === button || restoringFocus) return;
        hide();
        active = button;
        tip.replaceChildren(make('div', text));
        tip.classList.toggle('role-popup', !!technical);
        tip.setAttribute('role', technical ? 'dialog' : 'tooltip');
        if (technical) {
            tip.setAttribute('aria-label', 'Recent role review');
            const details = make('details');
            details.append(make('summary', 'Technical details'), make('div', technical));
            details.ontoggle = position;
            tip.append(details);
            button.setAttribute('aria-expanded', 'true');
        } else tip.removeAttribute('aria-label');
        button.setAttribute('aria-describedby', tip.id);
        tip.showPopover();
        position();
    }
    function leaveInteractive() {
        clearTimeout(closeTimer);
        closeTimer = setTimeout(() => {
            if (!tip.contains(tip.getRootNode().activeElement)) hide();
        }, 200);
    }
    tip.onpointerenter = () => clearTimeout(closeTimer);
    tip.onpointerleave = leaveInteractive;
    tip.onfocusout = e => { if (!tip.contains(e.relatedTarget) && e.relatedTarget !== active) hide(); };
    let sort = data.columns.indexOf(root.dataset.sort || 'Rank');
    if (sort < 0) sort = 0;
    let direction = Number(root.dataset.direction || 1);
    const note = make('div');
    note.setAttribute('role', 'status');
    root.append(note);
    const selectors = new Map();
    function updateChecks() {
        try {
            for (const [select, pick] of selectors) {
                const saved = savedPlay(localStorage, pick.id);
                select.options[0].text = saved ? 'Remove' : 'Save…';
                select.value = saved ? String(saved.confidence_cents ?? 5) : '';
                colorConfidence(select);
                select.closest('tr').classList.toggle('saved-play', !!saved);
            }
        } catch { note.textContent = 'Allow site storage to save plays in this browser.'; }
    }
    function render() {
        selectors.clear();
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
                const select = confidenceSelect('Confidence for '+pick.player+' '+pick.side+' '+pick.line+' '+pick.stat, true);
                select.onchange = () => {
                    try {
                        if (select.value) saveConfidence(localStorage, pick, Number(select.value));
                        else setPick(localStorage, pick, false);
                        note.textContent = pick.player + (select.value ? " saved to Today's Plays" : " removed from Today's Plays") + ' · '+pick.day;
                        notifyPicks(pick.day);
                    } catch { updateChecks(); note.textContent='Could not save this change. Check your browser storage.'; }
                };
                selectors.set(select,pick);saveCell.append(select);
            }
            for (const [i, label] of row.labels.entries()) {
                const td = tr.insertCell();
                if (row.styles[i]) td.style.cssText = row.styles[i];
                if (data.columns[i] === 'Review flags') {
                    for (const badge of row.reviewBadges || []) {
                        const button=make('button',badge.label);
                        button.type='button';button.className='review-badge review-'+badge.color;
                        button.setAttribute('aria-label',badge.label+': '+badge.detail);
                        button.onpointerenter=e=>{if(e.pointerType!=='touch')show(button,badge.detail,badge.technical);};
                        button.onfocus=()=>show(button,badge.detail,badge.technical);
                        if (badge.technical) {
                            button.setAttribute('aria-haspopup','dialog');
                            button.setAttribute('aria-controls',tip.id);
                            button.setAttribute('aria-expanded','false');
                            button.onclick=()=>{
                                show(button,badge.detail,badge.technical);
                                tip.querySelector('summary').focus();
                            };
                            button.onpointerleave=leaveInteractive;
                            button.onblur=e=>{if(!tip.contains(e.relatedTarget))hide();};
                        } else {
                            button.onclick=()=>show(button,badge.detail);
                            button.onpointerleave=button.onblur=hide;
                        }
                        td.append(button);
                    }
                } else if (data.columns[i] === 'Matchup rating') {
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
                    if (data.columns[i] === 'EV %' && pick && row.values[i] != null) {
                        td.title = pick.side + ' at ' + (pick.odds > 0 ? '+' : '') + pick.odds;
                    }
                }
            }
        }
    }
    const saveHeader=make('th','Save / confidence');saveHeader.scope='col';headers.append(saveHeader);
    data.columns.forEach((label, i) => {
        const th = make('th'), button = make('button', label);
        th.scope = 'col';
        button.type = 'button';
        button.onclick = () => {
            direction = sort === i ? -direction : (label === 'EV %' ? -1 : 1);
            sort = i;
            root.dataset.sort = label;
            root.dataset.direction = direction;
            render(); updateChecks();
        };
        th.append(button);
        headers.append(th);
    });
    const dismiss = e => {
        if (e.key !== 'Escape') return;
        const button = active, returnFocus = tip.contains(tip.getRootNode().activeElement);
        hide();
        if (button && returnFocus) {
            restoringFocus = true;
            button.focus();
            restoringFocus = false;
        }
    };
    const outside = e => { if (active && !e.composedPath().includes(active) && !e.composedPath().includes(tip)) hide(); };
    document.addEventListener('keydown', dismiss);
    document.addEventListener('pointerdown', outside, true);
    scroll.onscroll = hide;
    const pageScroll = e => { if (!tip.contains(e.target)) hide(); };
    window.addEventListener('scroll', pageScroll, true);
    window.addEventListener('resize', hide);
    render(); updateChecks();
    const unwatch=watchPicks(updateChecks);
    return () => {
        unwatch();
        hide();
        document.removeEventListener('keydown', dismiss);
        document.removeEventListener('pointerdown', outside, true);
        window.removeEventListener('scroll', pageScroll, true);
        window.removeEventListener('resize', hide);
    };
}
