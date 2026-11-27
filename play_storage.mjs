// Each prop has its own key so selecting two different plays in two tabs is safe.
const PICK_PREFIX = 'nba-props.plays.v1.pick.';
const SHARE_PREFIX = 'nba-props.plays.v1.share.';
const PICK_EVENT = 'nba-props:plays-changed';
export const CONFIDENCE_TIERS = new Map([
    [5,{label:'Low',fill:'#edf0f3',ink:'#45515d'}],
    [10,{label:'Medium',fill:'#ffe4a3',ink:'#633d00'}],
    [15,{label:'High',fill:'#166449',ink:'#ffffff'}],
]);

function colorConfidence(select) {
    const tier=CONFIDENCE_TIERS.get(Number(select.value));
    select.style.backgroundColor=tier?.fill || '';
    select.style.color=tier?.ink || '';
    select.style.fontWeight=Number(select.value)===15?'bold':'';
}

function confidenceSelect(label, allowRemove) {
    const select = document.createElement('select');
    select.setAttribute('aria-label', label);
    if (allowRemove) select.add(new Option('Save…', ''));
    for (const [cents, tier] of CONFIDENCE_TIERS) select.add(new Option(tier.label+' · '+cents+'¢', String(cents)));
    return select;
}

export function saveConfidence(storage, pick, cents) {
    if (!CONFIDENCE_TIERS.has(cents)) throw new Error('Choose a valid confidence tier.');
    const saved = savedPlay(storage, pick.id) || pick;
    storage.setItem(PICK_PREFIX + pick.id, JSON.stringify({...saved, confidence_cents:cents, confidence_edited:true}));
}
export function acknowledgeConfidence(storage, synced) {
    for (const [id,cents] of Object.entries(synced)) {
        const p=savedPlay(storage,id);
        if (p && p.confidence_cents===cents && p.confidence_edited!==false)
            storage.setItem(PICK_PREFIX+id,JSON.stringify({...p,confidence_edited:false}));
    }
}
export function normalizeAltChoice(choice) {
    if(Array.isArray(choice)) {
        if(!choice.length || choice.length>20 || choice.some(c=>typeof c!=='string' || ['auto','none'].includes(c)))
            throw new Error('Choose up to 20 alternates.');
        const choices=[...new Set(choice.map(normalizeAltChoice))].sort();
        return choices.length===1 ? choices[0] : choices;
    }
    if (typeof choice!=='string' || (!['auto','none','dd'].includes(choice) && !/^line:\d+(?:\.5)?$/.test(choice)))
        throw new Error('Choose a valid alternate.');
    return choice;
}
export function toggleAlternate(choice, value, checked) {
    if(checked && ['auto','none'].includes(value)) return value;
    const choices=(Array.isArray(choice)?choice:[choice]).filter(c=>!['auto','none',value].includes(c));
    if(checked) choices.push(value);
    return choices.length ? normalizeAltChoice(choices) : 'none';
}
export function saveAlternate(storage, pick, choice) {
    choice=normalizeAltChoice(choice);
    const saved=savedPlay(storage,pick.id) || pick;
    storage.setItem(PICK_PREFIX+pick.id,JSON.stringify({...saved,alt_choice:choice,alt_edited:true}));
}
export function acknowledgeAlternates(storage, synced) {
    for (const [id,choice] of Object.entries(synced)) {
        const p=savedPlay(storage,id);
        if (p && JSON.stringify(normalizeAltChoice(p.alt_choice || 'auto'))===JSON.stringify(normalizeAltChoice(choice)))
            storage.setItem(PICK_PREFIX+id,JSON.stringify({...p,alt_edited:false}));
    }
}
export function savedPlay(storage, id) {
    const raw = storage.getItem(PICK_PREFIX + id);
    return raw === null ? null : JSON.parse(raw);
}
export function setPick(storage, pick, selected) {
    if (!selected) storage.removeItem(PICK_PREFIX + pick.id);
    else if (!savedPlay(storage, pick.id)) storage.setItem(PICK_PREFIX + pick.id, JSON.stringify(pick));
}
export function setPickStatus(storage, id, status) {
    const pick = savedPlay(storage, id);
    if (pick && ['Review', 'Confirmed'].includes(status)) {
        storage.setItem(PICK_PREFIX + id, JSON.stringify({...pick, status}));
    }
}
export function allPicks(storage) {
    return Object.keys(storage).filter(k => k.startsWith(PICK_PREFIX))
        .map(k => savedPlay(storage, k.slice(PICK_PREFIX.length))).filter(Boolean);
}
function notifyPicks(day) {
    window.dispatchEvent(new CustomEvent(PICK_EVENT, {detail: day}));
}
function watchPicks(update) {
    const changed = e => {
        if (e.key == null || e.key.startsWith(PICK_PREFIX) || e.key.startsWith(SHARE_PREFIX)) update();
    };
    window.addEventListener(PICK_EVENT, update);
    window.addEventListener('storage', changed);
    return () => {
        window.removeEventListener(PICK_EVENT, update);
        window.removeEventListener('storage', changed);
    };
}
