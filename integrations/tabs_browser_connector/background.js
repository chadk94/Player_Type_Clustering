// Bind explicitly to one source tab: another Tabs window must not replace its scenario.
const isSource = url => {try {const u=new URL(url);return u.origin==='https://app.tabsnba.com' && u.pathname.replace(/\/$/,'')==='/projections';}catch{return false;}};
const isModel = url => {try{return new URL(url).origin==='https://cktest2.streamlit.app';}catch{return false;}};
chrome.action.onClicked.addListener(async tab=>{
    if (!isSource(tab.url)) {
        await chrome.action.setBadgeText({tabId:tab.id,text:'Tabs?'});
        await chrome.action.setTitle({tabId:tab.id,title:'Open your TabsNBA projections tab, then click this connector.'});
        return;
    }
    const previous=(await chrome.storage.session.get('sourceTab')).sourceTab;
    if (previous!=null && previous!==tab.id) await chrome.action.setBadgeText({tabId:previous,text:''}).catch(()=>{});
    await chrome.storage.session.set({sourceTab:tab.id});
    await chrome.action.setBadgeBackgroundColor({tabId:tab.id,color:'#177b54'});
    await chrome.action.setBadgeText({tabId:tab.id,text:'ON'});
    await chrome.action.setTitle({tabId:tab.id,title:'This Tabs tab supplies the comparison. Your scenario and minutes are never changed.'});
});
chrome.tabs.onRemoved.addListener(async id=>{
    if ((await chrome.storage.session.get('sourceTab')).sourceTab===id) await chrome.storage.session.remove('sourceTab');
});
chrome.runtime.onMessage.addListener((message,sender,reply)=>{
    // Streamlit renders its app in a same-origin /~/+/ frame. Other origins remain excluded.
    if (message?.type!=='tabs-minutes-read-v1' || !isModel(sender.url)) return;
    (async()=>{
        try {
            const id=(await chrome.storage.session.get('sourceTab')).sourceTab;
            if (id==null) throw Error('Open TabsNBA and click the Tabs minutes comparison extension to connect that tab.');
            const tab=await chrome.tabs.get(id);
            if (!isSource(tab.url)) throw Error('The connected Tabs tab is no longer on its projections page.');
            const result=await chrome.tabs.sendMessage(id,{type:'tabs-minutes-capture-v1'},{frameId:0});
            if (!result || JSON.stringify(result).length>300000) throw Error('Tabs returned an invalid comparison.');
            reply(result);
        } catch(error) {reply({ok:false,error:'Tabs comparison unavailable. Keep the connected Tabs page open and reload it if needed.'});}
    })();
    return true;
});
