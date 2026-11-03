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

export function readHistory(db, season) {
    return new Promise((resolve, reject) => {
        const tx = db.transaction(['forecasts', 'results'], 'readonly');
        const records = tx.objectStore('forecasts').getAll();
        const results = tx.objectStore('results').getAll();
        tx.oncomplete = () => {
            const selected = records.result.filter(r => r.season === season);
            const ids = new Set(selected.map(r => r.id));
            // Keep forecast JSON as text: JS would otherwise turn 0.0 into 0,
            // invalidating the canonical Python forecast checksum on restore.
            resolve({records: selected.map(r => r.body), results: results.result.filter(r => ids.has(r.id))});
        };
        tx.onerror = tx.onabort = () => reject(tx.error || Error('History read failed'));
    });
}

export default function({data, parentElement, setStateValue}) {
    if (parentElement.pointsRequest === data.request) return;
    parentElement.pointsRequest = data.request;
    (async () => {
        let db;
        try {
            db = await openHistory();
            await writeHistory(db, data);
            const history = await readHistory(db, data.season);
            if (parentElement.pointsRequest === data.request) {
                setStateValue('reply', {request: data.request, status: 'ok', ...history});
            }
        } catch (_) {
            if (parentElement.pointsRequest === data.request) {
                setStateValue('reply', {request: data.request, status: 'unavailable'});
            }
        } finally { db?.close(); }
    })();
}
