export default function({ data, parentElement, setStateValue }) {
    // Replace listeners on rerender; keep request/value memory on the host so
    // acknowledgements and focus events cannot create repeated app reruns.
    parentElement.minutesCleanup?.();
    const read = () => {
        const raw = localStorage.getItem(data.storage_key);
        const result = {bundle: raw, status: 'ok'};
        try { if (raw !== null) JSON.parse(raw); }
        catch (_) { result.status = 'invalid'; }
        return result;
    };
    const publish = (result, source) => {
        parentElement.minutesValue = result.bundle;
        parentElement.minutesStatus = result.status;
        parentElement.minutesRevision = (parentElement.minutesRevision || 0) + 1;
        setStateValue('reply', {request: data.request, ...result, source,
                              revision: parentElement.minutesRevision});
    };
    const checkUpdates = () => {
        // A new explicit upload/clear wins over an old restore or focus event.
        if (data.pending !== false) return;
        let result;
        try { result = read(); }
        catch (_) { result = {status: 'unavailable'}; }
        if (result.bundle !== parentElement.minutesValue || result.status !== parentElement.minutesStatus) {
            publish(result, 'storage');
        }
    };
    if (data.pending !== false && parentElement.minutesRequest !== data.request) {
        parentElement.minutesRequest = data.request;
        let result;
        try {
            if (data.action === 'save') localStorage.setItem(data.storage_key, data.bundle);
            else if (data.action === 'clear') localStorage.removeItem(data.storage_key);
            result = read();
        } catch (_) { result = {status: 'unavailable'}; }
        publish(result, 'command');
    } else {
        checkUpdates();
    }
    const onStorage = event => {
        if (event.key === data.storage_key || event.key === null) checkUpdates();
    };
    const onVisible = () => { if (document.visibilityState === 'visible') checkUpdates(); };
    window.addEventListener('storage', onStorage);
    window.addEventListener('focus', checkUpdates);
    document.addEventListener('visibilitychange', onVisible);
    const cleanup = () => {
        window.removeEventListener('storage', onStorage);
        window.removeEventListener('focus', checkUpdates);
        document.removeEventListener('visibilitychange', onVisible);
    };
    parentElement.minutesCleanup = cleanup;
    return cleanup;
}
