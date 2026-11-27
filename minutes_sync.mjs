export default function({data, parentElement, setStateValue}) {
    parentElement.connectionCleanup?.();
    parentElement.replaceChildren();
    const storageKey = 'nba-props.minutes-connection.v1';
    const row = document.createElement('details');
    const title = document.createElement('summary');
    title.textContent = data.connected ? 'NBA model connection' : 'Link NBA model';
    row.append(title);
    const input = document.createElement('input');
    input.type = 'password';
    input.placeholder = 'Private minutes connection';
    input.setAttribute('aria-label', input.placeholder);
    input.style.cssText = 'width:min(90%,580px);padding:8px;margin:8px 0';
    const connect = document.createElement('button');
    connect.textContent = 'Connect';
    const disconnect = document.createElement('button');
    disconnect.textContent = 'Disconnect';
    const note = document.createElement('p');
    note.textContent = `Approve minutes in the NBA model. This page checks every ${data.refresh_seconds || 10} seconds and refreshes projections automatically when approvals change.`;
    const publish = (connection, error=null) => {
        parentElement.connectionValue = connection;
        setStateValue('connection', {ready:true, connection, error, revision:Date.now()});
    };
    connect.onclick = () => {
        const value = input.value.trim();
        if (!value) return;
        try {localStorage.setItem(storageKey, value); publish(value);}
        catch {publish(value, 'Connection is available only for this session.');}
    };
    disconnect.onclick = () => {
        try {localStorage.removeItem(storageKey); publish(null);}
        catch {publish(null, 'Browser storage could not be cleared.');}
    };
    row.append(note, input, connect);
    if (data.connected) row.append(disconnect);
    parentElement.append(row);
    if (!parentElement.restored) {
        parentElement.restored = true;
        try {publish(localStorage.getItem(storageKey));}
        catch {publish(null, 'Browser storage is unavailable.');}
    }
    const check = () => {
        try {
            const value = localStorage.getItem(storageKey);
            if (value !== parentElement.connectionValue) publish(value);
        } catch { /* Keep the working session connection if storage is blocked. */ }
    };
    const onStorage = event => { if (event.key === storageKey || event.key === null) check(); };
    window.addEventListener('storage', onStorage);
    window.addEventListener('focus', check);
    const cleanup = () => {
        window.removeEventListener('storage', onStorage);
        window.removeEventListener('focus', check);
    };
    parentElement.connectionCleanup = cleanup;
    return cleanup;
}
