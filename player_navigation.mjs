const REQUEST = 'nba-projections:open-player';
const READY = 'nba-projections:player-ready';

export default function({data, parentElement, setStateValue}) {
    parentElement.navigationCleanup?.();
    if (data.mode === 'send') {
        const request = data.request;
        if (!request) return;
        const send = () => window.dispatchEvent(new CustomEvent(REQUEST, {detail: request}));
        if (parentElement.sentRequest !== request.request_id) {
            parentElement.sentRequest = request.request_id;
            send();
        }
        // Handles a receiver mounting after the sender without polling.
        window.addEventListener(READY, send);
        const cleanup = () => window.removeEventListener(READY, send);
        parentElement.navigationCleanup = cleanup;
        return cleanup;
    }
    let observer;
    let timer;
    const stopWaiting = () => { observer?.disconnect(); clearTimeout(timer); };
    const open = () => {
        const request = parentElement.navigationRequest;
        if (!request || parentElement.navigationOpened === request.request_id) return true;
        const heading = document.getElementById('selected-player-projection');
        if (!heading || heading.textContent.trim() !== `${request.player_name} vs ${request.opponent}`) return false;
        const tabs = heading.closest('[data-testid="stTabs"]');
        const tab = [...(tabs?.querySelectorAll('[role="tab"]') || [])]
            .find(item => item.textContent.trim() === 'Player vs Opponent');
        if (!tab) return false;
        parentElement.navigationOpened = request.request_id;
        stopWaiting();
        tab.click();
        tab.focus({preventScroll: true});
        heading.scrollIntoView({block: 'start', behavior: 'instant'});
        return true;
    };
    const waitForPlayer = () => {
        stopWaiting();
        if (!open()) {
            observer = new MutationObserver(open);
            observer.observe(document.body, {childList: true, subtree: true, characterData: true});
            timer = setTimeout(stopWaiting, 15000);
        }
    };
    const receive = event => {
        const request = event.detail;
        if (!request?.request_id || parentElement.navigationRequest?.request_id === request.request_id) return;
        parentElement.navigationRequest = request;
        setStateValue('requested', request); // Reruns only the player fragment.
        waitForPlayer();
    };
    window.addEventListener(REQUEST, receive);
    window.dispatchEvent(new Event(READY));
    waitForPlayer();
    const cleanup = () => { stopWaiting(); window.removeEventListener(REQUEST, receive); };
    parentElement.navigationCleanup = cleanup;
    return cleanup;
}
