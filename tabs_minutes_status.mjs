const EVENT='nba-tabs-minutes-status';
const key=row=>`${row.day}|${row.team}|${row.opponent}`;
const time=value=>new Intl.DateTimeFormat('en-US',{timeZone:'America/Denver',month:'short',day:'numeric',hour:'numeric',minute:'2-digit',timeZoneName:'short'}).format(new Date(value));

export function summarize(scopes,rows,now=Date.now()) {
    const indexed=new Map(rows.map(row=>[key(row),row]));
    const teams=scopes.map(scope=>{
        const row=indexed.get(key(scope));
        if(!row)return {team:scope.team,text:`${scope.team}: no saved minutes`,missing:true};
        const age=value=>value ? now-new Date(value).getTime() : NaN;
        const fresh=[row.published,row.captured].every(value=>age(value)>=0 && age(value)<=7200000);
        const status=!row.complete?'partial':row.saved?'saved':!fresh?'stale':'';
        return {team:scope.team,published:row.published,status,
            text:`${scope.team}: ${row.published ? time(row.published) : 'time unverified'}${status?` (${status})`:''}`};
    });
    if(teams.length<=2)return {summary:teams.map(row=>row.text).join(' · '),teams};
    const stamps=teams.map(row=>row.published).filter(Boolean).sort();
    let summary=stamps.length ? time(stamps[0])+(stamps.at(-1)!==stamps[0]?` – ${time(stamps.at(-1))}`:'') : 'No verified update time';
    const count=predicate=>teams.filter(predicate).length;
    summary+=` · ${teams.length} teams`;
    for(const [label,predicate] of [['saved',r=>r.status==='saved'],['stale',r=>r.status==='stale'],['partial',r=>r.status==='partial'],['missing',r=>r.missing]]) {
        const n=count(predicate);if(n)summary+=` · ${n} ${label}`;
    }
    return {summary,teams};
}

export default function({data,parentElement}) {
    const summary=parentElement.querySelector('[data-summary]'),feedback=parentElement.querySelector('[data-feedback]');
    const details=parentElement.querySelector('details'),teams=parentElement.querySelector('[data-teams]');
    const button=parentElement.querySelector('button');
    let rows=data.rows||[];
    function draw() {
        const result=summarize(data.scopes||[],rows);
        summary.textContent=result.summary;
        details.hidden=result.teams.length<=2;
        teams.textContent=result.teams.map(row=>row.text).join(' · ');
    }
    function receive(event) {
        const value=event.detail||{};
        if(value.rows) {rows=value.rows;draw();}
        button.disabled=value.state==='checking';
        feedback.textContent=value.message||'';
    }
    button.hidden=!data.browser;
    if(!data.browser)feedback.textContent='CSV snapshot — upload a new export in TabsNBA minutes comparison to update.';
    button.onclick=()=>{
        if(typeof window.nbaTabsMinutesRefresh==='function')window.nbaTabsMinutesRefresh();
        else feedback.textContent='Connection unavailable. Open TabsNBA with the connector ON; your saved minutes are preserved.';
    };
    window.addEventListener(EVENT,receive);
    draw();
    // Age labels update locally; no polling, app rerun or projection work.
    const timer=setInterval(draw,60000);
    return ()=>{clearInterval(timer);window.removeEventListener(EVENT,receive);button.onclick=null;};
}
