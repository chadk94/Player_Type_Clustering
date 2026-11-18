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
    scroll.append(table);
    const tip = make('div');
    tip.id = 'rating-detail-' + data.team;
    tip.setAttribute('role', 'tooltip');
    tip.setAttribute('popover', 'manual');
    root.append(scroll, tip);
    let active, closeTimer, restoringFocus = false, pinned = false;
    function hide() {
        clearTimeout(closeTimer);
        if (active) {
            active.removeAttribute('aria-describedby');
            if (active.hasAttribute('aria-expanded')) active.setAttribute('aria-expanded', 'false');
        }
        active = null;
        pinned = false;
        if (tip.matches(':popover-open')) tip.hidePopover();
    }
    function position() {
        if (!active) return;
        tip.style.maxHeight = '';
        const b = active.getBoundingClientRect(), t = tip.getBoundingClientRect();
        let left = Math.max(8, Math.min(b.left, innerWidth - t.width - 8)), top;
        const above = b.top - 14, below = innerHeight - b.bottom - 14;
        if (t.height <= above) top = b.top - t.height - 6;
        else if (t.height <= below) top = b.bottom + 6;
        else if (b.left >= t.width + 14 || innerWidth - b.right >= t.width + 14) {
            // Tall panels must not cover the trigger between pointer-enter and click.
            left = b.left >= t.width + 14 ? b.left - t.width - 6 : b.right + 6;
            top = Math.max(8, Math.min(b.top, innerHeight - t.height - 8));
        } else {
            tip.style.maxHeight = Math.max(60, Math.max(above, below)) + 'px';
            top = below >= above ? b.bottom + 6 : 8;
        }
        tip.style.left = left + 'px'; tip.style.top = top + 'px';
    }
    function show(button, text, technical, projection, matchup, expanded = false, pin = false) {
        clearTimeout(closeTimer);
        if ((active === button && !pin) || restoringFocus) return;
        hide();
        active = button;
        pinned = pin;
        tip.replaceChildren(make('div', text));
        tip.classList.toggle('role-popup', !!technical);
        tip.classList.toggle('projection-popup', !!projection);
        tip.classList.toggle('matchup-popup', !!matchup);
        tip.setAttribute('role', technical || projection || matchup ? 'dialog' : 'tooltip');
        if (matchup) {
            tip.setAttribute('aria-label', 'Recent similar-player performances vs ' + matchup.opponent);
            const heading = make('div'), title = make('strong', 'Recent similar players vs ' + matchup.opponent);
            heading.className = 'matchup-heading';
            const close = make('button', 'Close');
            close.type = 'button'; close.className = 'matchup-close';
            close.onclick = () => {
                hide(); restoringFocus = true; button.focus(); restoringFocus = false;
            };
            heading.append(title, close);
            tip.replaceChildren(heading);
            tip.append(make('div', 'Same archetype · actual game totals'));
            const usualHint = make('small', '“Usual” compares each player at the minutes shown.');
            tip.append(usualHint);
            for (const panel of matchup.panels) {
                const section = make('section');
                section.className = 'matchup-peer-section';
                section.append(make('strong', panel.title));
                if (!panel.rows.length) {
                    section.append(make('p', 'No qualifying similar-player performances available this season.'));
                } else {
                    const viewport = make('div'), peers = make('table');
                    viewport.className = 'matchup-peer-scroll';
                    peers.setAttribute('aria-label', panel.title + ' recent performances');
                    const header = peers.createTHead().insertRow();
                    panel.columns.forEach((col, index) => {
                        const th = make('th', col); th.scope = 'col';
                        if (index === 2) th.className = 'peer-primary';
                        header.append(th);
                    });
                    const body = peers.createTBody();
                    for (const peer of panel.rows.slice(0, expanded ? 10 : 3)) {
                        const tr = body.insertRow(), name = tr.insertCell();
                        name.append(make('span', peer.player), make('small', peer.date));
                        peer.values.forEach((value, index) => {
                            const cell = tr.insertCell(); cell.textContent = value;
                            if (index === 1) cell.className = 'peer-primary';
                        });
                    }
                    viewport.append(peers); section.append(viewport);
                    const count = Math.min(panel.rows.length, expanded ? 10 : 3);
                    const caption = make('small', `Showing ${count} of ${panel.rows.length} available appearances${panel.rows.length === 10 ? ' (latest 10)' : ''}.`);
                    section.append(caption);
                }
                tip.append(section);
            }
            if (matchup.panels.some(panel => panel.rows.length > 3)) {
                const more = make('button', expanded ? 'Show 3 recent performances' : 'Show up to 10 performances');
                more.type = 'button'; more.className = 'matchup-more';
                more.setAttribute('aria-expanded', String(expanded));
                more.onclick = () => {
                    show(button, text, null, null, matchup, !expanded, true);
                    tip.querySelector('.matchup-more').focus();
                };
                tip.append(more);
            }
            const details = make('details');
            details.append(make('summary', 'Technical details'), make('div', matchup.note + '\n\n' + text));
            details.ontoggle = position;
            tip.append(details);
            button.setAttribute('aria-expanded', 'true');
        } else if (projection) {
            tip.setAttribute('aria-label', projection.player + ' full projection');
            const title = make('strong', projection.player);
            title.className = 'projection-title';
            const context = make('div', `${projection.team} vs ${projection.opponent} · ${projection.minutes} projected minutes`);
            context.className = 'projection-context';
            const stats = make('dl');
            stats.className = 'projection-stats';
            for (const {stat, value} of projection.stats) {
                const cell = make('div');
                cell.append(make('dt', stat), make('dd', value));
                stats.append(cell);
            }
            const caption = make('div', 'Current game projections, including stats without a qualifying edge. PF = fouls; — = unavailable.');
            caption.className = 'projection-caption';
            tip.replaceChildren(title, context, stats, caption);
            button.setAttribute('aria-expanded', 'true');
        } else if (technical) {
            tip.setAttribute('aria-label', 'Play review details');
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
            if (!pinned && !tip.contains(tip.getRootNode().activeElement)) hide();
        }, 200);
    }
    tip.onpointerenter = () => clearTimeout(closeTimer);
    tip.onpointerleave = leaveInteractive;
    tip.onfocusout = e => { if (!pinned && !tip.contains(e.relatedTarget) && e.relatedTarget !== active) hide(); };
    const sort = data.columns.indexOf('Edge %');
    const note = make('div');
    note.setAttribute('role', 'status');
    root.append(note);
    const selectors = new Map();
    const portraits = new Map();
    let portraitObserver;
    function playerIdentity(name, id, plays, projection) {
        const identity = make('button'), avatar = make('span');
        identity.type = 'button';
        identity.setAttribute('aria-label', 'Full projection for ' + name);
        identity.setAttribute('aria-haspopup', 'dialog');
        identity.setAttribute('aria-controls', tip.id);
        identity.setAttribute('aria-expanded', 'false');
        const openProjection = () => show(identity, name + '’s full game projection is unavailable.', null, projection);
        identity.onpointerenter = e => { if (e.pointerType !== 'touch') openProjection(); };
        identity.onfocus = identity.onclick = openProjection;
        identity.onpointerleave = leaveInteractive;
        identity.onblur = e => { if (!tip.contains(e.relatedTarget)) hide(); };
        identity.className = 'player-identity';
        if (plays > 1) identity.classList.add('multiple-plays');
        if (plays > 2) identity.classList.add('many-plays');
        avatar.className = 'player-avatar';
        avatar.setAttribute('aria-hidden', 'true');
        const parts = name.trim().split(/\s+/);
        const initials = make('span', (parts[0]?.[0] || '') + (parts.length > 1 ? parts.at(-1)[0] : ''));
        initials.className = 'player-initials';
        avatar.append(initials);
        if (Number.isSafeInteger(id) && id > 0) {
            const img = make('img');
            img.alt = '';
            img.width = 44; img.height = 32;
            img.loading = 'lazy'; img.decoding = 'async'; img.fetchPriority = 'low';
            img.referrerPolicy = 'no-referrer';
            img.onload = () => avatar.classList.add('photo-loaded');
            img.onerror = () => img.remove();
            // No request until the table is rendered and this portrait nears view.
            portraits.set(img, `https://cdn.nba.com/headshots/nba/latest/260x190/${id}.png`);
            avatar.append(img);
        }
        const playerName = make('span', name);
        playerName.className = 'player-name';
        if (plays > 1) identity.append(playerName, avatar);
        else identity.append(avatar, playerName);
        return identity;
    }
    function loadVisiblePortraits() {
        const load = img => {
            img.src = portraits.get(img);
            portraits.delete(img);
        };
        if ('IntersectionObserver' in window) {
            portraitObserver = new IntersectionObserver(entries => {
                for (const entry of entries) if (entry.isIntersecting) {
                    portraitObserver.unobserve(entry.target);
                    load(entry.target);
                }
            }, {rootMargin: '100px'});
            for (const img of portraits.keys()) portraitObserver.observe(img);
        } else {
            // Older browsers still get native lazy loading and low priority.
            for (const img of portraits.keys()) load(img);
        }
    }
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
        for (const body of [...table.tBodies]) body.remove();
        for (const [i, th] of [...headers.cells].slice(1).entries()) {
            th.setAttribute('aria-sort', i === sort ? 'other' : 'none');
        }
        let body;
        // The payload groups players by their best edge, then their props by edge.
        for (const row of data.rows) {
            if (row.playerRowSpan) body = table.createTBody();
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
                const playerCell = ['Rank', 'Player'].includes(data.columns[i]);
                if (playerCell && !row.playerRowSpan) continue;
                const td = tr.insertCell();
                td.dataset.column = data.columns[i];
                if (playerCell) {
                    td.rowSpan = row.playerRowSpan;
                    td.classList.add('player-group');
                }
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
                    button.setAttribute('aria-haspopup', 'dialog');
                    button.setAttribute('aria-controls', tip.id);
                    button.setAttribute('aria-expanded', 'false');
                    const preview = () => show(button, row.detail, null, null, row.matchup);
                    button.onpointerenter = e => { if (e.pointerType !== 'touch') preview(); };
                    button.onfocus = preview;
                    button.onclick = () => {
                        show(button, row.detail, null, null, row.matchup, true, true);
                        (tip.querySelector('.matchup-more') || tip.querySelector('.matchup-close'))?.focus();
                    };
                    button.onpointerleave = leaveInteractive;
                    button.onblur = e => { if (!pinned && !tip.contains(e.relatedTarget)) hide(); };
                    td.append(button);
                } else if (data.columns[i] === 'Player') {
                    td.append(playerIdentity(label, row.playerId, row.playerRowSpan, row.projection));
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
    const hasIdleCallback = 'requestIdleCallback' in window;
    const portraitTask = hasIdleCallback ? window.requestIdleCallback(loadVisiblePortraits, {timeout:1500}) : setTimeout(loadVisiblePortraits, 0);
    const unwatch=watchPicks(updateChecks);
    return () => {
        if (hasIdleCallback) window.cancelIdleCallback(portraitTask); else clearTimeout(portraitTask);
        portraitObserver?.disconnect();
        unwatch();
        hide();
        document.removeEventListener('keydown', dismiss);
        document.removeEventListener('pointerdown', outside, true);
        window.removeEventListener('scroll', pageScroll, true);
        window.removeEventListener('resize', hide);
    };
}
