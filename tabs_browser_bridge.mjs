export const SAVED_TABS_KEY='nba-tabs-last-valid-minutes-v1';

export function readSavedTabs(storage) {
    try {
        const text=storage?.getItem(SAVED_TABS_KEY);
        if (!text || text.length>150000) return null;
        const snapshot=JSON.parse(text);
        return snapshot?.schema===1 && Array.isArray(snapshot.rows) ? snapshot : null;
    } catch {return null;}
}

export function saveValidatedTabs(storage,snapshot) {
    // Only the Python-validated acknowledgement is persisted, never a raw feed reply.
    try {if (snapshot) storage?.setItem(SAVED_TABS_KEY,JSON.stringify(snapshot));} catch {}
}

export default function({data,parentElement,setStateValue}) {
    const status=parentElement.querySelector('[role=status]');
    const button=parentElement.querySelector('button');
    // One listener across rerenders; heartbeats do not rerun the dashboard.
    window.nbaTabsBridgeCleanup?.();
    const canonical=value=>value && typeof value==='object' ? (Array.isArray(value) ? value.map(canonical) : Object.fromEntries(Object.keys(value).sort().map(key=>[key,canonical(value[key])]))) : value;
    const snapshotKey=snapshot=>snapshot ? {...snapshot,observed_at:null,
        rows:[...(snapshot.rows||[])].sort((a,b)=>JSON.stringify(canonical(a)).localeCompare(JSON.stringify(canonical(b))))} : snapshot;
    const fingerprint=value=>JSON.stringify(canonical(value ? {...value,
        snapshot:snapshotKey(value.snapshot),saved_snapshot:snapshotKey(value.saved_snapshot)} : value));
    let storage;try {storage=localStorage;} catch {}
    const saved=data.saved_snapshot || readSavedTabs(storage);
    saveValidatedTabs(storage,data.saved_snapshot);
    let last=fingerprint(data.previous),waiting=null,timeout,disposed=false;
    function publish(value) {
        const hash=fingerprint(value);
        if (hash!==last) {last=hash;setStateValue('result',value);}
    }
    function unavailable(message) {
        status.textContent=saved ? `Using saved Tabs minutes · ${message}` : message;
        publish({ok:false,error:message,...(saved?{saved_snapshot:saved}:{})});
    }
    function request() {
        if (disposed || waiting) return;
        waiting=crypto.randomUUID();
        window.postMessage({channel:'nba-tabs-minutes-v1',type:'request',id:waiting},location.origin);
        timeout=setTimeout(()=>{waiting=null;unavailable('Browser connector unavailable. Open both sites in Chrome with the connector enabled.');},8000);
    }
    function receive(event) {
        const msg=event.data;
        if (event.source!==window || event.origin!==location.origin || msg?.channel!=='nba-tabs-minutes-v1' || msg.type!=='response' || msg.id!==waiting) return;
        clearTimeout(timeout);waiting=null;
        const value=msg.result;
        if (!value?.ok || !value.snapshot) {unavailable(value?.error||'Tabs comparison unavailable.');return;}
        const s=value.snapshot;
        if (JSON.stringify(s).length>150000) {unavailable('Tabs comparison too large.');return;}
        status.textContent=`Connected · ${s.scope} · ${s.scenario} · Tabs updated ${new Date(s.source_updated_at).toLocaleTimeString([], {hour:'numeric',minute:'2-digit'})}`;
        publish(value);
    }
    window.addEventListener('message',receive);
    button.onclick=request;
    const timer=setInterval(request,30000);
    const focus=()=>request();window.addEventListener('focus',focus);
    window.nbaTabsBridgeCleanup=()=>{disposed=true;clearInterval(timer);clearTimeout(timeout);window.removeEventListener('message',receive);window.removeEventListener('focus',focus);};
    if (!data.previous && saved) unavailable('Reconnecting to Tabs…');
    request();
    return window.nbaTabsBridgeCleanup;
}
