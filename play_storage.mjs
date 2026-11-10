// Each prop has its own key so selecting two different plays in two tabs is safe.
const PICK_PREFIX = 'nba-props.plays.v1.pick.';
const SHARE_PREFIX = 'nba-props.plays.v1.share.';
const PICK_EVENT = 'nba-props:plays-changed';
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
