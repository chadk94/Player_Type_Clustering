export default function({ data, parentElement, setStateValue }) {
    // A Python rerun after acknowledgement must not start another rerun loop.
    if (parentElement.minutesRequest === data.request) return;
    parentElement.minutesRequest = data.request;
    const reply = {request: data.request, status: 'ok'};
    try {
        if (data.action === 'save') {
            localStorage.setItem(data.storage_key, data.bundle);
        } else if (data.action === 'clear') {
            localStorage.removeItem(data.storage_key);
        } else {
            const raw = localStorage.getItem(data.storage_key);
            try {
                if (raw !== null) JSON.parse(raw); // Syntax check; preserve the exact text.
                reply.bundle = raw;
            }
            catch (_) { reply.status = 'invalid'; }
        }
    } catch (_) { reply.status = 'unavailable'; }
    setStateValue('reply', reply);
}
