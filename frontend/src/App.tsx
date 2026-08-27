import { useEffect, useState } from "react";
import "./App.css";

// Dev mode (npm run dev, port 5173) talks to the separately-run backend
// on 8001. The production build (served by `devpilot` from the same
// FastAPI process as the API) uses relative fetches to whatever origin
// actually served the page -- no separate config needed per mode.
const API_BASE = import.meta.env.DEV ? "http://localhost:8001" : "";
const POLL_INTERVAL_MS = 2000;

interface ToolCall {
  name: string;
  arguments: Record<string, unknown>;
  result: string;
}

interface Turn {
  question: string;
  answer: string;
  toolCalls: ToolCall[];
}

interface PendingApproval {
  tool: string;
  arguments: Record<string, unknown>;
  message: string;
}

interface SessionView {
  status: "running" | "awaiting_approval" | "awaiting_plan_approval" | "done" | "error";
  pending: PendingApproval | null;
  plan: string[] | null;
  answer: string | null;
  tool_calls: ToolCall[] | null;
  error: string | null;
}

interface ConversationSummary {
  id: string;
  title: string;
  updated_at: string;
}

interface ConversationDetail {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  turns: { question: string; answer: string; tool_calls: ToolCall[] }[];
}

type Mode = "ask" | "plan" | "agent" | "planner";

const MODE_INFO: Record<Mode, { label: string; description: string }> = {
  ask: { label: "Ask", description: "Plain conversation, no tools -- no project access" },
  plan: { label: "Plan", description: "Explores and plans, read-only -- never writes or executes" },
  agent: { label: "Agent", description: "Full access, mutations still require your approval" },
  planner: {
    label: "Planner",
    description: "Shows a step-by-step plan first, waits for your approval, then executes it",
  },
};

// Set by the VS Code extension via a ?theme= query param on this page's
// iframe src (Entry 38) -- an iframe is a separate document from the
// extension's own webview, so VS Code's --vscode-* CSS variables can't
// cross that boundary automatically; this is the bridge instead.
function detectTheme(): "light" | "dark" {
  const param = new URLSearchParams(window.location.search).get("theme");
  return param === "dark" ? "dark" : "light";
}

function App() {
  const [input, setInput] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [pending, setPending] = useState<PendingApproval | null>(null);
  const [pendingPlan, setPendingPlan] = useState<string[] | null>(null);
  // The conversation's identity, persisted across turns (unlike sessionId
  // above, which is cleared after each turn to stop that turn's polling).
  // Sent on every /ask so the backend continues the same conversation
  // instead of starting over with no memory of it (Entry 31).
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [historyList, setHistoryList] = useState<ConversationSummary[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [mode, setMode] = useState<Mode>("agent");
  const [modeMenuOpen, setModeMenuOpen] = useState(false);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", detectTheme());
  }, []);

  // Poll the session while a question is in flight -- a single /ask
  // request can't block until a human clicks Approve/Decline, so progress
  // (including a paused gated-tool call) only shows up via this poll.
  //
  // Deliberately NOT setInterval: setInterval fires on a fixed clock
  // regardless of whether the previous poll has resolved yet. Under load
  // (a slow multi-minute LLM turn, a busy extension host) polls can pile
  // up faster than they complete; once the session finally flips to
  // "done", every overlapping in-flight poll sees it independently and
  // each one appended its own duplicate turn -- the real cause of the
  // same answer appearing repeated many times in the panel. A
  // self-scheduling setTimeout only starts the next poll after the
  // current one fully resolves, so more than one request in flight for
  // the same session is structurally impossible, and polling stops for
  // good the moment "done"/"error" is reached instead of relying on a
  // cleanup race to catch up.
  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    let timeoutId: ReturnType<typeof setTimeout>;

    async function poll() {
      try {
        const response = await fetch(`${API_BASE}/session/${sessionId}`);
        if (!response.ok) throw new Error(`Backend returned ${response.status}`);
        const data: SessionView = await response.json();
        if (cancelled) return;

        setPending(data.status === "awaiting_approval" ? data.pending : null);
        setPendingPlan(data.status === "awaiting_plan_approval" ? data.plan : null);

        if (data.status === "done") {
          setTurns((prev) => [
            ...prev,
            {
              question: pendingQuestion ?? "",
              answer: data.answer ?? "",
              toolCalls: data.tool_calls ?? [],
            },
          ]);
          setLoading(false);
          setSessionId(null);
          setPendingQuestion(null);
          return; // done -- no further poll scheduled
        }
        if (data.status === "error") {
          setError(data.error ?? "Unknown error");
          setLoading(false);
          setSessionId(null);
          setPendingQuestion(null);
          return; // error -- no further poll scheduled
        }

        timeoutId = setTimeout(poll, POLL_INTERVAL_MS);
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : String(err));
        setLoading(false);
        setSessionId(null);
        setPendingQuestion(null);
      }
    }

    timeoutId = setTimeout(poll, POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      clearTimeout(timeoutId);
    };
  }, [sessionId, pendingQuestion]);

  async function handleSend() {
    const message = input.trim();
    if (!message || loading) return;

    setInput("");
    setLoading(true);
    setError(null);
    setPendingQuestion(message);

    try {
      const response = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, session_id: conversationId, mode }),
      });
      if (!response.ok) {
        throw new Error(`Backend returned ${response.status}`);
      }
      const data = await response.json();
      setSessionId(data.session_id);
      setConversationId(data.session_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setLoading(false);
      setPendingQuestion(null);
    }
  }

  function newConversation() {
    setTurns([]);
    setConversationId(null);
    setError(null);
    setHistoryOpen(false);
  }

  async function toggleHistory() {
    const opening = !historyOpen;
    setHistoryOpen(opening);
    if (!opening) return;

    setHistoryLoading(true);
    try {
      const response = await fetch(`${API_BASE}/conversations`);
      if (!response.ok) throw new Error(`Backend returned ${response.status}`);
      setHistoryList(await response.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setHistoryLoading(false);
    }
  }

  async function openConversation(id: string) {
    setError(null);
    try {
      await fetch(`${API_BASE}/conversations/${id}/resume`, { method: "POST" });
      const response = await fetch(`${API_BASE}/conversations/${id}`);
      if (!response.ok) throw new Error(`Backend returned ${response.status}`);
      const detail: ConversationDetail = await response.json();

      setTurns(
        detail.turns.map((t) => ({ question: t.question, answer: t.answer, toolCalls: t.tool_calls }))
      );
      setConversationId(id);
      setHistoryOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  function formatTimestamp(iso: string): string {
    const date = new Date(iso);
    return date.toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  }

  async function respondToApproval(approved: boolean) {
    if (!sessionId) return;
    setPending(null); // optimistic -- the next poll confirms real state
    try {
      const response = await fetch(`${API_BASE}/session/${sessionId}/approve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ approved }),
      });
      if (!response.ok) {
        throw new Error(`Backend returned ${response.status}`);
      }
    } catch (err) {
      // Previously swallowed silently (Entry 42 gap-fix) -- a failed
      // approve/decline call (e.g. a stale session after a backend
      // restart) left the user staring at "thinking" forever with no
      // indication anything went wrong.
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  // Entry 46: the coarser-grained sibling of respondToApproval above --
  // approves/declines the WHOLE generated plan, not one tool call.
  async function respondToPlan(approved: boolean) {
    if (!sessionId) return;
    setPendingPlan(null); // optimistic -- the next poll confirms real state
    try {
      const response = await fetch(`${API_BASE}/session/${sessionId}/approve_plan`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ approved }),
      });
      if (!response.ok) {
        throw new Error(`Backend returned ${response.status}`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <div className="brand-mark">D</div>
          <div className="brand-text">
            <h1>DevPilot AI</h1>
            <p className="subtitle">Local AI engineering copilot — MCP-powered</p>
          </div>
        </div>
        <div className="header-actions">
          <div className="history-dropdown-wrap">
            <button className="history-btn" onClick={toggleHistory}>
              History
            </button>
            {historyOpen && (
              <div className="history-dropdown">
                {historyLoading && <div className="history-empty">Loading…</div>}
                {!historyLoading && historyList.length === 0 && (
                  <div className="history-empty">No past conversations yet.</div>
                )}
                {!historyLoading &&
                  historyList.map((c) => (
                    <button key={c.id} className="history-item" onClick={() => openConversation(c.id)}>
                      <span className="history-item-title">{c.title}</span>
                      <span className="history-item-time">{formatTimestamp(c.updated_at)}</span>
                    </button>
                  ))}
              </div>
            )}
          </div>
          {(turns.length > 0 || conversationId) && (
            <button className="new-conversation-btn" onClick={newConversation} disabled={loading}>
              <span className="btn-label-full">New conversation</span>
              <span className="btn-label-short">New</span>
            </button>
          )}
        </div>
      </header>

      <main className="conversation">
        {turns.length === 0 && !loading && (
          <p className="empty-state">Ask DevPilot something about this project.</p>
        )}

        {turns.map((turn, i) => (
          <div className="turn" key={i}>
            <div className="message user-message">{turn.question}</div>
            <div className="message ai-message">
              {turn.answer}
              {turn.toolCalls.length > 0 && (
                <div className="tool-trace">
                  <div className="tool-trace-label">Tools used</div>
                  <ul>
                    {turn.toolCalls.map((call, j) => (
                      <li key={j}>
                        <span className="tool-check">✓</span> {call.name}(
                        {JSON.stringify(call.arguments)})
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && pendingQuestion && (
          <div className="turn">
            <div className="message user-message">{pendingQuestion}</div>

            {pendingPlan ? (
              <div className="message ai-message plan-request">
                <div className="approval-label">DevPilot proposes this plan -- approve to carry it out</div>
                <ol className="plan-steps">
                  {pendingPlan.map((step, i) => (
                    <li key={i}>{step}</li>
                  ))}
                </ol>
                <div className="approval-actions">
                  <button className="approve-btn" onClick={() => respondToPlan(true)}>
                    Approve plan
                  </button>
                  <button className="decline-btn" onClick={() => respondToPlan(false)}>
                    Decline
                  </button>
                </div>
              </div>
            ) : pending ? (
              <div className="message ai-message approval-request">
                <div className="approval-label">DevPilot wants to run a tool that needs your approval</div>
                <div className="approval-tool">{pending.tool}({JSON.stringify(pending.arguments)})</div>
                <div className="approval-message">{pending.message}</div>
                <div className="approval-actions">
                  <button className="approve-btn" onClick={() => respondToApproval(true)}>
                    Approve
                  </button>
                  <button className="decline-btn" onClick={() => respondToApproval(false)}>
                    Decline
                  </button>
                </div>
              </div>
            ) : (
              <div className="message ai-message thinking">
                <span className="spinner" />
                DevPilot is thinking... this can take a few minutes on local hardware.
              </div>
            )}
          </div>
        )}

        {error && <div className="message error-message">Error: {error}</div>}
      </main>

      <footer className="input-bar">
        <div className="mode-select-wrap">
          <button
            className={`mode-select-btn mode-${mode}`}
            onClick={() => setModeMenuOpen((v) => !v)}
            disabled={loading}
          >
            <span className="mode-select-btn-label">{MODE_INFO[mode].label}</span>
            <span className="mode-caret">▾</span>
          </button>
          {modeMenuOpen && (
            <div className="mode-menu">
              {(Object.keys(MODE_INFO) as Mode[]).map((m) => (
                <button
                  key={m}
                  className={`mode-menu-item${m === mode ? " active" : ""}`}
                  onClick={() => {
                    setMode(m);
                    setModeMenuOpen(false);
                  }}
                >
                  <span className="mode-menu-item-label">{MODE_INFO[m].label}</span>
                  <span className="mode-menu-item-desc">{MODE_INFO[m].description}</span>
                </button>
              ))}
            </div>
          )}
        </div>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder="Ask DevPilot..."
          disabled={loading}
        />
        <button onClick={handleSend} disabled={loading || !input.trim()}>
          Send
        </button>
      </footer>
    </div>
  );
}

export default App;
