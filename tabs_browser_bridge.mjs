export default function({data,parentElement,setStateValue}) {
    const status=parentElement.querySelector('[role=status]');
    const button=parentElement.querySelector('button');
    // One listener across rerenders; heartbeats do not rerun the dashboard.
    window.nbaTabsBridgeCleanup?.();
    const canonical=value=>value && typeof value==='object' ? (Array.isArray(value) ? value.map(canonical) : Object.fromEntries(Object.keys(value).sort().map(key=>[key,canonical(value[key])]))) : value;
    const fingerprint=value=>JSON.stringify(canonical(value?.snapshot ? {
        ...value.snapshot,observed_at:null,
        rows:[...(value.snapshot.rows||[])].sort((a,b)=>JSON.stringify(canonical(a)).localeCompare(JSON.stringify(canonical(b))))
    } : value));
    let last=fingerprint(data.previous),waiting=null,timeout,disposed=false;
    function publish(value) {
        const hash=fingerprint(value);
        if (hash!==last) {last=hash;setStateValue('result',value);}
    }
    function unavailable(message) {status.textContent=message;publish({ok:false,error:message});}
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
    request();
    return window.nbaTabsBridgeCleanup;
}
