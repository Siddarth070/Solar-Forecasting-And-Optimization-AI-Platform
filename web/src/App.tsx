import { lazy, Suspense } from "react";
import { Route, Routes, Link } from "react-router-dom";
import { Footer, Header } from "./components/Header";
import { Landing } from "./pages/Landing";

// Charts (recharts) only load on the pages that use them.
const Console = lazy(() => import("./pages/Console").then((m) => ({ default: m.Console })));
const Model = lazy(() => import("./pages/Model").then((m) => ({ default: m.Model })));

function NotFound() {
  return (
    <main className="wrap section">
      <h1>No page here</h1>
      <p style={{ marginTop: 16 }}>
        Go to the <Link to="/">overview</Link> or open the <Link to="/console">console</Link>.
      </p>
    </main>
  );
}

export function App() {
  return (
    <>
      <Header />
      <Suspense fallback={<p className="wrap loading">Loading…</p>}>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/console" element={<Console />} />
          <Route path="/model" element={<Model />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </Suspense>
      <Footer />
    </>
  );
}
