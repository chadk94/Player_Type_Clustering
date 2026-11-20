import assert from 'node:assert/strict';
import test from 'node:test';
import render from '../edge_rating_table.mjs';

// Minimal DOM for exercising the real component's event handlers without feeds.
class Element {
    constructor(tag) {
        this.tag = tag; this.children = []; this.attributes = {}; this.style = {}; this.dataset = {};
        this.open = false; this.shows = 0; this.hides = 0;
        this.classes = new Set();
        this.classList = {add: name => this.classes.add(name),
            toggle: (name, enabled) => enabled ? this.classes.add(name) : this.classes.delete(name)};
    }
    append(...children) { for (const child of children) { child.parentElement = this; this.children.push(child); } }
    replaceChildren(...children) { this.children = []; this.text = ''; this.append(...children); }
    set textContent(text) { this.text = text; }
    get textContent() { return [this.text ?? '', ...this.children.map(child => child.textContent)].join(' '); }
    setAttribute(key, value) { this.attributes[key] = value; }
    getAttribute(key) { return this.attributes[key]; }
    removeAttribute(key) { delete this.attributes[key]; }
    hasAttribute(key) { return key in this.attributes; }
    get tBodies() { return this.children.filter(child => child.tag === 'tbody'); }
    get cells() { return this.children; }
    insert(tag) { const child = new Element(tag); this.append(child); return child; }
    createTHead() { return this.insert('thead'); }
    createTBody() { return this.insert('tbody'); }
    insertRow() { return this.insert('tr'); }
    insertCell() { return this.insert('td'); }
    contains(target) { return this === target || this.children.some(child => child.contains(target)); }
    find(predicate) { return predicate(this) ? this : this.children.map(child => child.find(predicate)).find(Boolean); }
    querySelector(selector) {
        return this.find(el => selector.startsWith('.') ? el.className === selector.slice(1) : el.tag === selector);
    }
    getBoundingClientRect() { return {left:900, right:950, top:100, bottom:130, width:600, height:400}; }
    getRootNode() { return document; }
    focus() { document.activeElement = this; this.onfocus?.(); }
    matches() { return this.open; }
    showPopover() { this.open = true; this.shows++; }
    hidePopover() { this.open = false; this.hides++; }
}

function fixture(shared = false, reviews = false) {
    if (!shared) {
        globalThis.document = new EventTarget();
        document.createElement = tag => new Element(tag);
        globalThis.window = new EventTarget();
    }
    window.requestIdleCallback = () => 1; window.cancelIdleCallback = () => {};
    globalThis.innerWidth = 1600; globalThis.innerHeight = 900;
    globalThis.watchPicks = () => () => {};
    const root = new Element('section');
    const data = {team:'Demo', columns:['Player','Stat','Matchup rating'], rows:[]};
    for (const [player, role] of [['Guard','Playmaker'],['Center','Stretch Big']]) {
        const values = [player,'3PM','4'];
        data.rows.push({values, labels:values, styles:[], playerRowSpan:1, detail:role,
            matchup:{opponent:'MIA', note:'Test only', panels:[{title:role,columns:['Player','MIN','3PA'],
                rows:Array.from({length:10},(_,i) => ({player:role+' peer '+i,date:'Nov 18',values:['30','5']}))}]}});
    }
    if (reviews) {
        data.columns.push('Review flags');
        data.rows.forEach((row, i) => {
            row.values.push('History'); row.labels = [...row.values];
            row.reviewBadges = [{label:i ? 'History −' : 'History +',
                color:i ? 'history' : 'history-support', popupStyle:'history',
                detail:i ? 'Opposes Under · 4/5 over 4.5' : 'Supports Over · 4/5 over 4.5', technical:'Sample rules'}];
        });
    }
    const cleanup = render({data, parentElement:root});
    const button = name => root.find(el => el.getAttribute('aria-label') === name+' 3PM matchup rating 4');
    const tip = root.find(el => el.hasAttribute('popover'));
    const dispatch = (type, values) => {
        const event = new Event(type); Object.assign(event, values); document.dispatchEvent(event);
    };
    const hover = button => button.onpointerenter({pointerType:'mouse'});
    return {root,tip,guard:button('Guard'),center:button('Center'),cleanup,dispatch,hover};
}

test('expansion stays in the top layer and keeps its archetype under stray hover events', () => {
    const f = fixture();
    try {
        f.hover(f.guard);
        assert.match(f.tip.textContent, /Showing 3 of 10/);
        assert.equal(f.tip.shows, 1);
        f.tip.querySelector('.matchup-more').onclick();
        assert.equal(f.tip.hides, 0, 'expansion must not expose underlying rows');
        assert.equal(f.tip.shows, 1, 'no hide/reopen cycle');
        assert.match(f.tip.textContent, /Showing 10 of 10/);
        f.hover(f.center); f.center.onfocus();
        assert.match(f.tip.textContent, /Playmaker peer 9/);
        assert.doesNotMatch(f.tip.textContent, /Stretch Big/);
        f.tip.querySelector('.matchup-more').onclick();
        assert.match(f.tip.textContent, /Showing 3 of 10/);
        assert.equal(f.tip.hides, 0);
        f.hover(f.center);
        assert.match(f.tip.textContent, /Playmaker/);
    } finally { f.cleanup(); }
});

test('deliberate selection and dismissal still allow another matchup', () => {
    const f = fixture();
    try {
        f.guard.onclick();
        f.dispatch('pointerdown', {composedPath:() => [f.center]});
        f.center.onclick();
        assert.match(f.tip.textContent, /Stretch Big peer 9/);
        f.dispatch('keydown', {key:'Escape'});
        assert.equal(f.tip.open, false);
        f.hover(f.guard);
        assert.match(f.tip.textContent, /Playmaker/);
        f.tip.querySelector('.matchup-close').onclick();
        assert.equal(f.tip.open, false);
        f.hover(f.center);
        assert.match(f.tip.textContent, /Stretch Big/);
    } finally { f.cleanup(); }
});

test('a second team table cannot replace a pinned matchup on passive hover', () => {
    const first = fixture(), second = fixture(true);
    try {
        first.guard.onclick();
        second.hover(second.center);
        assert.equal(first.tip.open, true);
        assert.equal(second.tip.open, false);
        assert.match(first.tip.textContent, /Playmaker peer 9/);
        // Explicit click switches ownership even across components.
        second.dispatch('pointerdown', {composedPath:() => [second.center]});
        second.center.onclick();
        assert.equal(first.tip.open, false);
        assert.equal(second.tip.open, true);
        assert.match(second.tip.textContent, /Stretch Big peer 9/);
    } finally { first.cleanup(); second.cleanup(); }
});

test('both history colors use compact popups and reset style for other details', () => {
    const f = fixture(false, true);
    try {
        for (const sign of ['+', '−']) {
            const button = f.root.find(el => el.tag === 'button' && el.textContent === 'History '+sign);
            f.hover(button);
            assert.equal(f.tip.classes.has('history-popup'), true);
            assert.match(f.tip.textContent, sign === '+' ? /Supports Over/ : /Opposes Under/);
            button.onclick();
            assert.equal(document.activeElement.tag, 'summary');
            assert.equal(document.activeElement.textContent, 'Technical details');
            f.dispatch('keydown', {key:'Escape'});
        }
        f.hover(f.guard);
        assert.equal(f.tip.classes.has('history-popup'), false);
        assert.match(f.tip.textContent, /Playmaker peer/);
    } finally { f.cleanup(); }
});
