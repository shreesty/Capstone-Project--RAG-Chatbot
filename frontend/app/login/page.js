"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [status, setStatus] = useState("");
  const router = useRouter();

  const apiFetch = async (path, options = {}) => {
    const res = await fetch(`${API_BASE}${path}`, options);
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || `HTTP ${res.status}`);
    }
    return res.json();
  };

  const handleAuth = async (mode) => {
    if (!username.trim() || !password.trim()) {
      setStatus("Enter username and password");
      return;
    }
    setStatus(mode === "signup" ? "Signing up..." : "Logging in...");
    try {
      const data = await apiFetch(`/${mode}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      window.localStorage.setItem("token", data.token);
      window.localStorage.setItem("username", data.username);
      setStatus("Success");
      router.replace("/");
      setTimeout(() => {
        window.location.href = "/";
      }, 50);
    } catch (err) {
      setStatus(err.message || "Auth failed");
    }
  };

  useEffect(() => {
    const tok = window.localStorage.getItem("token");
    if (tok) {
      router.replace("/");
    }
  }, [router]);

  return (
    <div className="page" style={{ display: "flex", justifyContent: "center", alignItems: "center" }}>
      <div className="panel" style={{ width: "420px" }}>
        <h2 style={{ marginTop: 0 }}>NepEd Bot</h2>
        <p className="small">Sign up or log in to access your chats.</p>
        <div style={{ display: "grid", gap: 10, marginTop: 12 }}>
          <input
            className="input"
            placeholder="Username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
          />
          <input
            className="input"
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          <div className="controls" style={{ justifyContent: "space-between" }}>
            <button className="button" onClick={() => handleAuth("signup")}>
              Sign up
            </button>
            <button className="button secondary" onClick={() => handleAuth("login")}>
              Login
            </button>
          </div>
          <div className="small status-line">{status}</div>
        </div>
      </div>
    </div>
  );
}
