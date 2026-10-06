import { useEffect, useState } from "react";
import { NavLink, Link } from "react-router-dom";
import { api } from "../lib/api";

function ApiStatus() {
  const [state, setState] = useState<"checking" | "up" | "down">("checking");
  const [detail, setDetail] = useState("");

  useEffect(() => {
    let alive = true;
    const check = () =>
      api
        .health()
        .then((h) => {
          if (!alive) return;
          setState(h.model_loaded ? "up" : "down");
          setDetail(h.model_loaded ? `API v${h.version}, model loaded` : "API up, model not loaded");
        })
        .catch(() => alive && (setState("down"), setDetail(`No response from ${api.base}`)));
    check();
    const id = setInterval(check, 30000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const text = state === "checking" ? "Checking API" : state === "up" ? "API connected" : "API offline";
  return (
    <div className="api-status" data-state={state} title={detail} role="status">
      <span className="dot" aria-hidden />
      {text}
    </div>
  );
}

export function Header() {
  return (
    <header className="site-header">
      <div className="wrap">
        <Link to="/" className="brand" aria-label="Zenith home">
          <img src="/favicon.svg" width="24" height="24" alt="" />
          Zenith
        </Link>
        <nav className="site-nav" aria-label="Main">
          <NavLink to="/" end>Overview</NavLink>
          <NavLink to="/console">Console</NavLink>
          <NavLink to="/model">Model</NavLink>
        </nav>
        <ApiStatus />
      </div>
    </header>
  );
}

export function Footer() {
  return (
    <footer className="site-footer">
      <div className="wrap">
        <span>Zenith — forecast-to-action for solar operators. Built in Greater Noida.</span>
        <span>
          <a href="https://github.com/Siddarth070/Solar-Forecasting-And-Optimization-AI-Platform">Source on GitHub</a>
          {" · "}
          <a href="mailto:siddharthaagrawal07@gmail.com">siddharthaagrawal07@gmail.com</a>
        </span>
      </div>
    </footer>
  );
}
