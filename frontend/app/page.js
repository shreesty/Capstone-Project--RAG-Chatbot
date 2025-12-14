"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

const defaultModel = "gpt-4o-mini";

function relativeTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  const diff = Date.now() - d.getTime();
  const mins = Math.round(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.round(hrs / 24);
  return `${days}d ago`;
}

export default function Page() {
  const [sessions, setSessions] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [sessionName, setSessionName] = useState("New chat");
  const [history, setHistory] = useState([]);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState("Idle");
  const [model, setModel] = useState(defaultModel);
  const [healthOk, setHealthOk] = useState(null);
  const [busy, setBusy] = useState(false);
  const [theme, setTheme] = useState("light");
  const [username, setUsername] = useState("");
  const [token, setToken] = useState(null);
  const [bootstrapped, setBootstrapped] = useState(false);
  const [sources, setSources] = useState([]);
  const chatWindowRef = useRef(null);
  const router = useRouter();

  useEffect(() => {
    const savedToken = window.localStorage.getItem("token");
    const savedUser = window.localStorage.getItem("username");
    if (savedToken) {
      setToken(savedToken);
      if (savedUser) setUsername(savedUser);
    } else {
      router.replace("/login");
    }
    checkHealth();
    const stored = window.localStorage.getItem("theme");
    if (stored === "light" || stored === "dark") {
      applyTheme(stored);
      setTheme(stored);
    } else {
      applyTheme("light");
    }
    setBootstrapped(true);
  }, []);

  useEffect(() => {
    if (token) loadSessions(token);
  }, [token]);

  useEffect(() => {
    if (chatWindowRef.current) {
      chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
    }
  }, [history]);

  const apiFetch = async (path, options = {}) => {
    const res = await fetch(`${API_BASE}${path}`, options);
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || `HTTP ${res.status}`);
    }
    return res.json();
  };

  const checkHealth = async () => {
    setStatus("Checking health...");
    try {
      await apiFetch("/health");
      setHealthOk(true);
      setStatus("");
    } catch (err) {
      setHealthOk(false);
      setStatus(err.message || "Health check failed");
    }
  };

  const loadSessions = async (tok = token) => {
    if (!tok) return;
    try {
      const data = await apiFetch(`/sessions?token=${encodeURIComponent(tok)}`);
      setSessions(data.sessions || []);
      if (!sessionId && data.sessions?.length) {
        await openSession(data.sessions[0].id);
      }
    } catch (err) {
      setStatus("Could not load sessions");
    }
  };

  const openSession = async (id) => {
    if (!token) {
      router.push("/login");
      return;
    }
    setStatus("Loading session...");
    try {
      const data = await apiFetch(`/sessions/${id}?token=${encodeURIComponent(token)}`);
      setSessionId(data.session.id);
      setSessionName(data.session.name);
      setHistory(data.history || []);
      setStatus("Ready");
    } catch (err) {
      setStatus(err.message || "Failed to load session");
    }
  };

  const newChat = async () => {
    if (!token) {
      setStatus("Login required");
      router.push("/login");
      return;
    }
    setStatus("Creating chat...");
    try {
      const data = await apiFetch(`/sessions?token=${encodeURIComponent(token)}`, {
        method: "POST",
      });
      setSessionId(data.id);
      setSessionName(data.name);
      setHistory([]);
      setInput("");
      await loadSessions();
      setStatus("");
    } catch (err) {
      setStatus(err.message || "Could not create chat");
    }
  };

  const renameSession = async (id, name) => {
    if (!token) return;
    if (!name.trim()) return;
    try {
      await apiFetch(`/sessions/${id}/rename?token=${encodeURIComponent(token)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      await loadSessions();
      if (id === sessionId) setSessionName(name);
    } catch (err) {
      setStatus(err.message || "Rename failed");
    }
  };

  const deleteSession = async (id) => {
    if (!token) return;
    try {
      await apiFetch(`/sessions/${id}?token=${encodeURIComponent(token)}`, { method: "DELETE" });
      if (id === sessionId) {
        setSessionId(null);
        setSessionName("New chat");
        setHistory([]);
      }
      await loadSessions();
    } catch (err) {
      setStatus(err.message || "Delete failed");
    }
  };

  const sendMessage = async () => {
    if (!token) {
      setStatus("Login required");
      return;
    }
    const trimmed = input.trim();
    if (!trimmed || busy) return;
    setBusy(true);
    setStatus("Thinking...");
    const optimisticHistory = [...history, { role: "user", content: trimmed }];
    setHistory(optimisticHistory);
    setInput("");
    setSources([]);
    try {
      const payload = {
        query: trimmed,
        backend: "azure",
        model,
        return_contexts: false,
        token,
      };
      if (sessionId) payload.session_id = sessionId;
      if (sessionName) payload.session_name = sessionName;
      const data = await apiFetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      setSessionId(data.session_id);
      setSessionName(data.session_name || sessionName);
      await openSession(data.session_id);
      await loadSessions();
      setSources(data.hits || []);
      setStatus("");
    } catch (err) {
      setStatus(err.message || "Failed to send");
    } finally {
      setBusy(false);
    }
  };

  const applyTheme = (next) => {
    document.body.classList.remove("theme-dark", "theme-light");
    document.body.classList.add(`theme-${next}`);
  };

  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    applyTheme(next);
    window.localStorage.setItem("theme", next);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="page">
      <aside className="sidebar panel">
        <div className="header" style={{ gap: 8 }}>
          <div className="pill" style={{ marginTop: 6 }}>User: {username || "signed in"}</div>
          <div className="controls" style={{ gap: 8 }}>
            <button
              className="button secondary"
              onClick={async () => {
                if (token) {
                  try {
                    await apiFetch(`/logout?token=${encodeURIComponent(token)}`, { method: "POST" });
                  } catch (_err) {
                    // ignore logout errors
                  }
                }
                setToken(null);
                window.localStorage.removeItem("token");
                window.localStorage.removeItem("username");
                setSessions([]);
                setSessionId(null);
                setSessionName("New chat");
                setHistory([]);
                setStatus("Logged out");
                router.push("/login");
              }}
            >
              Logout
            </button>
          </div>
        </div>
        <button className="button" onClick={newChat} style={{ width: "100%" }}>
          New chat
        </button>
        <div className="session-list">
          {sessions.length === 0 ? (
            <p className="small">No conversations yet.</p>
          ) : (
            sessions.map((s) => (
              <div
                key={s.id}
                className={`session-card ${s.id === sessionId ? "active" : ""}`}
                onClick={() => openSession(s.id)}
              >
                <div className="header" style={{ gap: 6 }}>
                  <p className="session-title" style={{ flex: 1 }}>{s.name || "Conversation"}</p>
                  <button
                    className="button secondary"
                    style={{ padding: "6px 8px", minWidth: 0 }}
                    onClick={(e) => {
                      e.stopPropagation();
                      const next = prompt("Rename chat", s.name || "Conversation");
                      if (next !== null) renameSession(s.id, next);
                    }}
                  >
                    ✎
                  </button>
                  <button
                    className="button secondary"
                    style={{ padding: "6px 8px", minWidth: 0 }}
                    onClick={(e) => {
                      e.stopPropagation();
                      if (confirm("Delete this chat?")) deleteSession(s.id);
                    }}
                  >
                    🗑
                  </button>
                </div>
                <div className="session-meta">{relativeTime(s.updated_at || s.created_at)}</div>
              </div>
            ))
          )}
        </div>
      </aside>

      <main className="panel">
        <div className="header" style={{ marginBottom: 12 }}>
          <div className="title">
            <div className="pill">NepEd Bot</div>
            <h2 style={{ margin: 0 }}>{sessionName}</h2>
          </div>
            <div className="controls" style={{ justifyContent: "flex-end" }}>
              <button className="button secondary" onClick={toggleTheme}>
                {theme === "dark" ? "Light mode" : "Dark mode"}
              </button>
              <div className="small status-line">{status}</div>
            </div>
          </div>

        <div className="pill" style={{ marginBottom: 8 }}>Backend: Azure OpenAI ({model})</div>

        {history.length === 0 ? (
          <div className="chat-shell" style={{ justifyContent: "center", alignItems: "center" }}>
            <div style={{ textAlign: "center", marginBottom: 12 }}>
              <div className="pill">New Chat</div>
              <p className="small" style={{ marginTop: 6 }}>Your prompt will appear here once you send it.</p>
            </div>
            <div className="composer hero">
              <textarea
                className="textarea"
                placeholder="Ask your questions ..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
              />
              <button className="button" onClick={sendMessage} disabled={busy || !input.trim()}>
                {busy ? "Generating…" : "Send"}
              </button>
            </div>
            <p className="small">Enter to send · Shift+Enter for a new line</p>
          </div>
        ) : (
          <div className="chat-shell">
            <div className="chat-window" ref={chatWindowRef}>
              {history.map((turn, idx) => (
                <div key={idx} className={`bubble ${turn.role === "user" ? "user" : "bot"}`}>
                  <h4>{turn.role === "user" ? "You" : "NepEd Bot"}</h4>
                  <div>{turn.content}</div>
                  {turn.role === "assistant" && sources.length > 0 && idx === history.length - 1 && (
                    <div className="sources" style={{ marginTop: 10 }}>
                      <h4>References</h4>
                      {sources.slice(0, 2).map((hit, hIdx) => {
                        const meta = hit.metadata || {};
                        const source = meta.source_url || meta.source_path || meta.source_domain || "unknown";
                        const label = source.replace(/^https?:\/\//, "");
                        return (
                          <div key={`${hit.id}-${hIdx}`} className="source-item">
                            <span className="small">[{hIdx + 1}]</span>
                            <a className="source-pill" href={source} target="_blank" rel="noreferrer">
                              {label}
                            </a>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              ))}
            </div>
            <div className="composer">
              <textarea
                className="textarea"
                placeholder="Ask your questions ..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
              />
              <button className="button" onClick={sendMessage} disabled={busy || !input.trim()}>
                {busy ? "Generating…" : "Send"}
              </button>
            </div>
            <p className="small">Enter to send · Shift+Enter for a new line</p>
          </div>
        )}
      </main>
    </div>
  );
}
