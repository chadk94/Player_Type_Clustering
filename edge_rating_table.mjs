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
    const sort = data.columns.indexOf('Edge %');
    const note = make('div');
    note.setAttribute('role', 'status');
    root.append(note);
    const selectors = new Map();
    function updateChecks() {
        try {
            for (const [group, {pick, inputs}] of selectors) {
                const saved = savedPlay(localStorage, pick.id);
                for (const [cents, input] of inputs) {
                    input.checked = !!saved && Number(saved.confidence_cents ?? 5) === cents;
                    input.parentElement.classList.toggle('selected', input.checked);
                }
                group.closest('tr').classList.toggle('saved-play', !!saved);
            }
        } catch { note.textContent = 'Allow site storage to save plays in this browser.'; }
    }
    function render() {
        selectors.clear();
        hide();
        body.replaceChildren();
        for (const [i, th] of [...headers.cells].slice(1).entries()) {
            th.setAttribute('aria-sort', i === sort ? 'descending' : 'none');
        }
        const rows = [...data.rows].sort((a, b) => {
            const x = a.values[sort], y = b.values[sort];
            if (x == null || y == null) return Number(x == null) - Number(y == null);
            return typeof x === 'number' && typeof y === 'number' ? y - x : 0;
        });
        for (const row of rows) {
            const tr = body.insertRow();
            const saveCell = tr.insertCell();
            const pick = row.pick;
            if (pick) {
                const group = make('div'), inputs = new Map();
                const description = pick.player+' '+pick.side+' '+pick.line+' '+pick.stat;
                group.className = 'confidence-checks';
                group.setAttribute('role', 'group');
                group.setAttribute('aria-label', 'Save '+description);
                for (const [cents, tier] of CONFIDENCE_TIERS) {
                    const label = make('label'), input = make('input');
                    label.className = 'confidence-choice confidence-'+cents;
                    label.title = tier.label+' confidence · '+cents+'¢ play-to allowance. Choose one; uncheck to remove.';
                    input.type = 'checkbox';
                    input.setAttribute('aria-label', tier.label+' confidence ('+cents+'¢) for '+description);
                    input.onchange = () => {
                        try {
                            if (input.checked) saveConfidence(localStorage, pick, cents);
                            else {
                                const saved = savedPlay(localStorage, pick.id);
                                if (saved && Number(saved.confidence_cents ?? 5) === cents) setPick(localStorage, pick, false);
                            }
                            updateChecks();
                            const saved = savedPlay(localStorage, pick.id);
                            note.textContent = pick.player + (saved ? " saved to Today's Plays · "+CONFIDENCE_TIERS.get(Number(saved.confidence_cents ?? 5))?.label : " removed from Today's Plays") + ' · '+pick.day;
                            notifyPicks(pick.day);
                        } catch { updateChecks(); note.textContent='Could not save this change. Check your browser storage.'; }
                    };
                    inputs.set(cents,input);
                    label.append(input, make('span', cents === 5 ? 'Lo' : cents === 10 ? 'Med' : 'Hi'));
                    group.append(label);
                }
                selectors.set(group,{pick,inputs});saveCell.append(group);
            }
            for (const [i, label] of row.labels.entries()) {
                const td = tr.insertCell();
                td.dataset.column = data.columns[i];
                if (typeof row.values[i] === 'number' || data.columns[i] === 'Book line (Over)') td.classList.add('numeric');
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
    const saveHeader=make('th','Save');saveHeader.scope='col';saveHeader.title='Lo = 5¢, Med = 10¢, Hi = 15¢. Choose one confidence level; uncheck it to remove the play.';headers.append(saveHeader);
    data.columns.forEach((label, i) => {
        const th = make('th', data.headers?.[i] || label);
        th.scope = 'col';
        th.dataset.column = label;
        th.title = data.headerHelp?.[i] || label;
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
