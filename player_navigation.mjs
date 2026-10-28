export default function({data, parentElement, setStateValue}) {
    parentElement.navigationCleanup?.();
    if (!data || parentElement.navigationOpened === data.id) return;
    let observer;
    let timer;
    const cleanup = () => {
        observer?.disconnect();
        clearTimeout(timer);
    };
    const open = () => {
        const heading = document.getElementById('selected-player-projection');
        if (!heading || heading.textContent.trim() !== data.heading) return false;
        const tabs = heading.closest('[data-testid="stTabs"]');
        const tab = [...(tabs?.querySelectorAll('[role="tab"]') || [])]
            .find(item => item.textContent.trim() === 'Player vs Opponent');
        if (!tab) return false;
        parentElement.navigationOpened = data.id;
        cleanup();
        tab.click();
        tab.focus({preventScroll: true});
        heading.scrollIntoView({block: 'start', behavior: 'instant'});
        setStateValue('opened', data.id);
        return true;
    };
    if (!open()) {
        // Wait only for this requested player to finish rendering. No polling
        // or ongoing refresh when browsing/scrolling through the dashboard.
        observer = new MutationObserver(open);
        observer.observe(document.body, {childList: true, subtree: true, characterData: true});
        timer = setTimeout(cleanup, 15000);
    }
    parentElement.navigationCleanup = cleanup;
    return cleanup;
}
