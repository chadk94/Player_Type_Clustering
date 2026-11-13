// The forecasts stay on this origin in the user's browser, never in shared app storage.
export function openHistory(factory = indexedDB) {
    return new Promise((resolve, reject) => {
        const request = factory.open('nba-props-points-probability-history-v1', 1);
        request.onupgradeneeded = () => {
            request.result.createObjectStore('forecasts', {keyPath: 'id'});
            request.result.createObjectStore('results', {keyPath: 'id'});
        };
        request.onerror = () => reject(request.error);
        request.onblocked = () => reject(Error('History is blocked'));
        request.onsuccess = () => resolve(request.result);
    });
}

export function writeHistory(db, data) {
    return new Promise((resolve, reject) => {
        // A single transaction protects first-write-wins across tabs and pages.
        const tx = db.transaction(['forecasts', 'results'], 'readwrite');
        const forecasts = tx.objectStore('forecasts');
        const results = tx.objectStore('results');
        for (const row of new Map(data.records.map(r => [r.id, r])).values()) {
            const existing = forecasts.get(row.id);
            existing.onsuccess = () => { if (!existing.result) forecasts.add(row); };
        }
        for (const row of data.results) results.put(row);
        tx.oncomplete = resolve;
        tx.onerror = tx.onabort = () => reject(tx.error || Error('History write failed'));
    });
}

// The store is append-only. Keep the last acknowledged key set on this
// component, so another page's writes are discovered without sending the
// browser's entire history over the connection on each player change.
export function readHistory(db, season, previous) {
    return new Promise((resolve, reject) => {
        const tx = db.transaction(['forecasts', 'results'], 'readonly');
        const forecasts = tx.objectStore('forecasts');
        const keys = forecasts.getAllKeys();
        const results = tx.objectStore('results').getAll();
        let ids, full, bodies = [];
        keys.onsuccess = () => {
            ids = new Set(keys.result);
            full = !previous || [...previous.ids].some(id => !ids.has(id));
            if (full) {
                const all = forecasts.getAll();
                all.onsuccess = () => { bodies = all.result.filter(r => r.season === season).map(r => r.body); };
            } else {
                for (const id of ids) {
                    if (previous.ids.has(id)) continue;
                    const one = forecasts.get(id);
                    one.onsuccess = () => { if (one.result?.season === season) bodies.push(one.result.body); };
                }
            }
        };
        tx.oncomplete = () => {
            const knownResults = new Map();
            const updates = [];
            for (const result of results.result) {
                const encoded = JSON.stringify(result);
                knownResults.set(result.id, encoded);
                if (full || previous.results.get(result.id) !== encoded) updates.push(result);
            }
            resolve({reply: {full, records: bodies, results: updates}, index: {ids, results: knownResults}});
        };
        tx.onerror = tx.onabort = () => reject(tx.error || Error('History read failed'));
    });
}

export default function({data, parentElement, setTriggerValue}) {
    if (parentElement.pointsRequest === data.request) return;
    parentElement.pointsRequest = data.request;
    (async () => {
        let db;
        try {
            db = await openHistory();
            await writeHistory(db, data);
            // A superseded response, remount, or explicit reload recovers from
            // the durable store. Never assume that the server received a reply.
            const prior = parentElement.pointsHistory;
            const previous = !data.restore && prior && prior.request === data.base ? prior.index : null;
            const history = await readHistory(db, data.season, previous);
            if (parentElement.pointsRequest === data.request) {
                parentElement.pointsHistory = {request: data.request, index: history.index};
                setTriggerValue('reply', {request: data.request, base: data.base, status: 'ok', ...history.reply});
            }
        } catch (_) {
            if (parentElement.pointsRequest === data.request) {
                setTriggerValue('reply', {request: data.request, status: 'unavailable'});
            }
        } finally { db?.close(); }
    })();
}
