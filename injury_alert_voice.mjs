const previewText = 'Voice test. Boston Celtics. Jayson Tatum.';

export default function render({data, parentElement}) {
    // Keep the unlocked voice and its queue across Streamlit fragment remounts.
    const voice = window.nbaInjuryVoice ??= {
        enabled: false, queue: [], current: null, seen: new Set(), message: '',
    };
    const synth = window.speechSynthesis;
    const Utterance = window.SpeechSynthesisUtterance;
    const preview = parentElement.querySelector('.preview');
    const mute = parentElement.querySelector('.mute');
    const status = parentElement.querySelector('[role=status]');
    preview.parentElement.hidden = !data.games.length;

    function refresh() {
        preview.textContent = voice.enabled ? 'Test voice' : 'Enable spoken alerts';
        preview.disabled = !synth || !Utterance;
        mute.hidden = !voice.enabled;
        status.textContent = preview.disabled ? 'This browser does not support spoken alerts.' :
            voice.message || (voice.enabled ? 'Spoken alerts on for this tab.' : 'Click to hear a voice test.');
    }
    voice.refresh = refresh;

    function stop() {
        clearTimeout(voice.timer);
        voice.queue.length = 0;
        if (voice.current) {
            voice.current = null;
            synth.cancel();
        }
        voice.message = '';
    }
    function speakNext() {
        if (!voice.enabled || voice.current || !voice.queue.length) return;
        const line = new Utterance(voice.queue.shift());
        line.lang = 'en-US';
        voice.current = line;
        function failed() {
            if (voice.current !== line) return;
            stop();
            voice.enabled = false;
            voice.message = 'Voice playback could not start. Check the browser’s sound settings and try again.';
            voice.refresh?.();
        }
        line.onstart = () => {
            if (voice.current !== line) return;
            clearTimeout(voice.timer);
            voice.message = 'Speaking…';
            voice.refresh?.();
        };
        line.onend = () => {
            if (voice.current !== line) return;
            clearTimeout(voice.timer);
            voice.current = null;
            voice.message = '';
            voice.refresh?.();
            speakNext();
        };
        line.onerror = failed;
        voice.timer = setTimeout(failed, 8000);
        try {
            if (synth.paused) synth.resume();
            synth.speak(line);
        } catch { failed(); }
    }
    const scope = JSON.stringify(data.games);
    if (scope !== voice.scope) {
        stop();
        voice.scope = scope;
    }
    preview.onclick = () => {
        if (voice.current) return;
        voice.enabled = true;
        voice.message = 'Starting voice test…';
        voice.queue.push(previewText);
        refresh();
        speakNext();
    };
    mute.onclick = () => {
        stop();
        voice.enabled = false;
        refresh();
    };
    if (data.games.length && data.messages.length) {
        if (data.report !== voice.report) {
            voice.report = data.report;
            voice.seen.clear();
        }
        for (const text of data.messages) {
            if (voice.seen.has(text)) continue;
            voice.seen.add(text);
            if (voice.enabled) voice.queue.push(text);
        }
        speakNext();
    }
    refresh();
    return () => {
        if (voice.refresh === refresh) voice.refresh = null;
    };
}
