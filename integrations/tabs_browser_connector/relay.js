// Page messages cannot choose URLs, other tabs, recipients or actions.
let pending=false,lastRequest=0;
window.addEventListener('message',async event=>{
    const msg=event.data;
    if (event.source!==window || event.origin!==location.origin || msg?.channel!=='nba-tabs-minutes-v1' || msg.type!=='request' || typeof msg.id!=='string' || msg.id.length>80) return;
    if (pending || Date.now()-lastRequest<1000) return;
    pending=true;lastRequest=Date.now();
    let result;
    try {result=await chrome.runtime.sendMessage({type:'tabs-minutes-read-v1'});}
    catch {result={ok:false,error:'Reload this prop-model tab after enabling the connector.'};}
    pending=false;
    window.postMessage({channel:'nba-tabs-minutes-v1',type:'response',id:msg.id,result},location.origin);
});
