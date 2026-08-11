import { useState } from "react";
import "./App.css";

const API_BASE = "http://localhost:8001";

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

function App() {
  const [input, setInput] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSend() {
    const message = input.trim();
    if (!message || loading) return;

    setInput("");
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
      if (!response.ok) {
        throw new Error(`Backend returned ${response.status}`);
      }
      const data = await response.json();
      setTurns((prev) => [
        ...prev,
        { question: message, answer: data.answer, toolCalls: data.tool_calls },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>DevPilot AI</h1>
        <p className="subtitle">Local AI engineering copilot — MCP-powered</p>
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

        {loading && (
          <div className="message ai-message thinking">
            DevPilot is thinking... this can take a few minutes on local hardware.
          </div>
        )}

        {error && <div className="message error-message">Error: {error}</div>}
      </main>

      <footer className="input-bar">
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
