import { render } from 'preact';
import { useEffect, useMemo, useRef, useState } from 'preact/hooks';
import { html } from 'htm/preact';

function nowLabel() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function makeId(prefix) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function usageLabel(usage) {
  if (!usage) {
    return '';
  }
  return `in ${usage.input_tokens || 0} / out ${usage.output_tokens || 0} / total ${usage.total_tokens || 0}`;
}

const THINK_MODE_KEY = 'mantle.think_mode';

function loadThinkMode() {
  try {
    const value = localStorage.getItem(THINK_MODE_KEY);
    if (value === 'collapsed' || value === 'hidden' || value === 'expanded') {
      return value;
    }
  } catch {
    // ignore localStorage access errors
  }
  return 'collapsed';
}

function parseThinkBlocks(text, includeUnclosedThink = false) {
  const source = String(text || '');
  const parts = [];
  const lower = source.toLowerCase();
  const openTag = '<think>';
  const closeTag = '</think>';
  let cursor = 0;

  while (cursor < source.length) {
    const start = lower.indexOf(openTag, cursor);
    if (start === -1) {
      if (cursor < source.length) {
        parts.push({ type: 'text', value: source.slice(cursor) });
      }
      break;
    }

    if (start > cursor) {
      parts.push({ type: 'text', value: source.slice(cursor, start) });
    }

    const thinkStart = start + openTag.length;
    const end = lower.indexOf(closeTag, thinkStart);
    if (end === -1) {
      if (includeUnclosedThink) {
        parts.push({ type: 'think', value: source.slice(thinkStart), openEnded: true });
      } else {
        parts.push({ type: 'text', value: source.slice(start) });
      }
      break;
    }

    parts.push({ type: 'think', value: source.slice(thinkStart, end), openEnded: false });
    cursor = end + closeTag.length;
  }

  return parts;
}

function renderMessageContent(text, thinking, streaming = false, thinkMode = 'collapsed') {
  const contentParts = parseThinkBlocks(text, streaming).map((part, idx) => {
    if (part.type === 'think') {
      if (thinkMode === 'hidden') {
        return null;
      }
      const summaryText = part.openEnded ? 'Thinking (live)' : 'Thinking';
      const shouldOpen = thinkMode === 'expanded';
      return html`
        <details key=${`think-${idx}`} class="think-block" open=${shouldOpen ? true : undefined}>
          <summary>${summaryText}</summary>
          <pre class="think-text">${part.value.trim()}</pre>
        </details>
      `;
    }
    return html`<span key=${`text-${idx}`}>${part.value}</span>`;
  });

  if (!thinking || thinkMode === 'hidden') {
    return contentParts;
  }

  const summaryText = streaming ? 'Thinking (live)' : 'Thinking';
  const shouldOpen = thinkMode === 'expanded';
  const thinkingBlock = html`
    <details key="thinking-separated" class="think-block" open=${shouldOpen ? true : undefined}>
      <summary>${summaryText}</summary>
      <pre class="think-text">${thinking.trim()}</pre>
    </details>
  `;

  return [thinkingBlock, ...contentParts];
}

function App() {
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState('');
  const [models, setModels] = useState(['mantle']);
  const [model, setModel] = useState('mantle');
  const [temperature, setTemperature] = useState(0.7);
  const [maxOutputTokens, setMaxOutputTokens] = useState(1024);
  const [connected, setConnected] = useState(false);
  const [pending, setPending] = useState(false);
  const [previousResponseID, setPreviousResponseID] = useState('');
  const [lastUsage, setLastUsage] = useState(null);
  const [errorText, setErrorText] = useState('');
  const [thinkMode, setThinkMode] = useState(loadThinkMode);

  const abortRef = useRef(null);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    try {
      localStorage.setItem(THINK_MODE_KEY, thinkMode);
    } catch {
      // ignore localStorage access errors
    }
  }, [thinkMode]);

  useEffect(() => {
    void refreshModels();
    const onFocus = () => {
      if (document.visibilityState === 'hidden') {
        return;
      }
      void refreshModels();
    };
    window.addEventListener('focus', onFocus);
    document.addEventListener('visibilitychange', onFocus);
    return () => {
      window.removeEventListener('focus', onFocus);
      document.removeEventListener('visibilitychange', onFocus);
    };
  }, []);

  const canSend = draft.trim().length > 0 && !pending;

  const tokenCount = useMemo(() => {
    return messages.reduce((acc, msg) => acc + (msg.content || '').length, 0);
  }, [messages]);

  async function refreshModels() {
    try {
      const response = await fetch('/v1/models');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      const list = Array.isArray(data?.data)
        ? data.data.map((item) => item.id).filter(Boolean)
        : ['mantle'];
      setModels(list.length > 0 ? list : ['mantle']);
      if (!list.includes(model) && list.length > 0) {
        setModel(list[0]);
      }
      setConnected(true);
    } catch {
      setConnected(false);
    }
  }

  function upsertMessage(id, updater) {
    setMessages((prev) => prev.map((m) => (m.id === id ? updater(m) : m)));
  }

  async function sendMessage(event) {
    event.preventDefault();
    const prompt = draft.trim();
    if (!prompt || pending) {
      return;
    }

    setErrorText('');

    const userMessage = {
      id: makeId('user'),
      role: 'user',
      content: prompt,
      time: nowLabel(),
      streaming: false,
    };

    const assistantId = makeId('assistant');
    const assistantMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      thinking: '',
      time: nowLabel(),
      streaming: true,
      usage: null,
    };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setDraft('');
    setPending(true);

    const requestBody = {
      model,
      input: prompt,
      stream: true,
      temperature,
      max_output_tokens: maxOutputTokens,
    };

    if (previousResponseID) {
      requestBody.previous_response_id = previousResponseID;
    }

    const abortController = new AbortController();
    abortRef.current = abortController;

    try {
      const response = await fetch('/v1/responses', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
        signal: abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const contentType = response.headers.get('content-type') || '';
      if (!contentType.includes('text/event-stream')) {
        const data = await response.json();
        upsertMessage(assistantId, (msg) => ({
          ...msg,
          content: data.output_text || '',
          streaming: false,
          usage: data.usage || null,
        }));
        setLastUsage(data.usage || null);
        setPreviousResponseID(data.id || '');
        void refreshModels();
        return;
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('stream reader unavailable');
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) {
            continue;
          }
          const payload = line.slice(6).trim();
          if (!payload || payload === '[DONE]') {
            continue;
          }

          let eventData;
          try {
            eventData = JSON.parse(payload);
          } catch {
            continue;
          }

          const eventType = eventData?.type;

          if (eventType === 'response.output_text.delta') {
            const delta = eventData.delta || '';
            if (delta) {
              upsertMessage(assistantId, (msg) => ({ ...msg, content: msg.content + delta }));
            }
          }

          if (eventType === 'response.output_reasoning.delta') {
            const delta = eventData.delta || '';
            if (delta) {
              upsertMessage(assistantId, (msg) => ({ ...msg, thinking: (msg.thinking || '') + delta }));
            }
          }

          if (eventType === 'response.output_text.done') {
            const text = eventData.text || '';
            upsertMessage(assistantId, (msg) => ({ ...msg, content: text }));
          }

          if (eventType === 'response.output_reasoning.done') {
            const text = eventData.text || '';
            upsertMessage(assistantId, (msg) => ({ ...msg, thinking: text }));
          }

          if (eventType === 'response.completed') {
            const resp = eventData.response || {};
            upsertMessage(assistantId, (msg) => ({
              ...msg,
              content: msg.content || resp.output_text || '',
              thinking: msg.thinking || resp.reasoning_text || '',
              streaming: false,
              usage: resp.usage || null,
            }));
            setLastUsage(resp.usage || null);
            setPreviousResponseID(resp.id || '');
            void refreshModels();
          }

          if (eventType === 'response.failed' || eventType === 'response.incomplete') {
            const reason = eventData?.response?.error?.message || eventType;
            upsertMessage(assistantId, (msg) => ({
              ...msg,
              content: msg.content || `Request ended: ${reason}`,
              streaming: false,
              role: 'system',
            }));
          }
        }
      }

      upsertMessage(assistantId, (msg) => ({ ...msg, streaming: false }));
    } catch (err) {
      const message = err?.name === 'AbortError' ? 'stream cancelled' : (err?.message || 'request failed');
      upsertMessage(assistantId, (msg) => ({
        ...msg,
        role: 'system',
        content: `Error: ${message}`,
        streaming: false,
      }));
      setErrorText(message);
    } finally {
      setPending(false);
      abortRef.current = null;
    }
  }

  function stopStreaming() {
    if (abortRef.current) {
      abortRef.current.abort();
    }
  }

  function clearChat() {
    setMessages([]);
    setPreviousResponseID('');
    setLastUsage(null);
    setErrorText('');
  }

  return html`
    <div class="shell">
      <aside class="sidebar">
        <div class="brand">
          <h1>Mantle Console</h1>
          <span class="badge">responses</span>
        </div>

        <div class="status">
          <span class="dot ${connected ? 'online' : ''}"></span>
          <span>${connected ? 'API online' : 'API unreachable'}</span>
        </div>

        <div class="card">
          <h2>Model</h2>
          <span class="model-name">${model}</span>
          <select value=${model} onChange=${(e) => setModel(e.currentTarget.value)}>
            ${models.map((m) => html`<option key=${m} value=${m}>${m}</option>`)}
          </select>
        </div>

        <div class="card stack">
          <h2>Generation</h2>
          <div>
            <label for="temperature">Temperature</label>
            <input
              id="temperature"
              type="range"
              min="0"
              max="2"
              step="0.1"
              value=${temperature}
              onInput=${(e) => setTemperature(Number(e.currentTarget.value))}
            />
            <div class="hint"><span>Focused</span><span>${temperature.toFixed(1)}</span><span>Creative</span></div>
          </div>
          <div>
            <label for="max">Max output tokens</label>
            <input
              id="max"
              type="number"
              min="1"
              max="8192"
              step="1"
              value=${maxOutputTokens}
              onInput=${(e) => setMaxOutputTokens(Number(e.currentTarget.value) || 1)}
            />
          </div>
        </div>

        <div class="card">
          <h2>Thinking View</h2>
          <select value=${thinkMode} onChange=${(e) => setThinkMode(e.currentTarget.value)}>
            <option value="collapsed">Collapsed (click to open)</option>
            <option value="hidden">Hidden</option>
            <option value="expanded">Expanded</option>
          </select>
        </div>

        <div class="stack">
          <button class="secondary" onClick=${clearChat} disabled=${messages.length === 0 || pending}>New Session</button>
          <button class="warn" onClick=${stopStreaming} disabled=${!pending}>Stop Stream</button>
        </div>
      </aside>

      <section class="main">
        <header class="topbar">
          <strong>Local inference chat</strong>
          <div class="metrics">
            <span>messages ${messages.length}</span>
            <span>chars ${tokenCount}</span>
            <span>${lastUsage ? usageLabel(lastUsage) : 'no usage yet'}</span>
          </div>
        </header>

        <div class="messages">
          ${messages.length === 0 && html`
            <div class="empty">
              <h3>Ready when you are</h3>
              <p>Use this panel to hit <code>/v1/responses</code> with streaming enabled. The session automatically chains with <code>previous_response_id</code>.</p>
            </div>
          `}

          ${messages.map((msg) => html`
            <article key=${msg.id} class="msg ${msg.role}">
              <div class="meta">
                <span>${msg.role}</span>
                <span>${msg.time}</span>
              </div>
              <div class="content ${msg.streaming ? 'stream' : ''}">${renderMessageContent(msg.content, msg.thinking, msg.streaming, thinkMode)}</div>
              ${msg.usage ? html`<div class="meta"><span>${usageLabel(msg.usage)}</span></div>` : ''}
            </article>
          `)}
          <div ref=${endRef}></div>
        </div>

        <form class="composer" onSubmit=${sendMessage}>
          <textarea
            placeholder="Ask Mantle anything... Shift+Enter for newline"
            value=${draft}
            onInput=${(e) => setDraft(e.currentTarget.value)}
            onKeyDown=${(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                void sendMessage(e);
              }
            }}
            disabled=${pending}
          ></textarea>
          <div class="actions">
            <small>${errorText ? `last error: ${errorText}` : `thread id: ${previousResponseID || 'not started'}`}</small>
            <div class="action-group">
              <button type="button" class="secondary" onClick=${() => setDraft('')} disabled=${pending || !draft}>Clear</button>
              <button type="submit" disabled=${!canSend}>${pending ? 'Generating...' : 'Send'}</button>
            </div>
          </div>
        </form>
      </section>
    </div>
  `;
}

render(html`<${App} />`, document.getElementById('app'));
