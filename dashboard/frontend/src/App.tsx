import React, { Suspense, lazy } from "react";
import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";

const Overview = lazy(() => import("./pages/Overview"));
const Logs     = lazy(() => import("./pages/Logs"));
const Trades   = lazy(() => import("./pages/Trades"));
const Settings = lazy(() => import("./pages/Settings"));
const Debug    = lazy(() => import("./pages/Debug"));
const ControlTower = lazy(() => import("./pages/ControlTower"));
const Recovery = lazy(() => import("./pages/Recovery"));
const LiveMonitor = lazy(() => import("./pages/LiveMonitor"));
const ResearchEvents = lazy(() => import("./pages/ResearchEvents"));

function Loading() {
  return (
    <div style={{ padding: 32, color: "var(--muted)", textAlign: "center" }}>
      Loading...
    </div>
  );
}

export default function App() {
  return (
    <Suspense fallback={<Loading />}>
      <Routes>
        <Route element={<Layout />}>
          <Route index        element={<Overview />} />
          <Route path="logs"     element={<Logs />} />
          <Route path="trades"   element={<Trades />} />
          <Route path="tower"    element={<ControlTower />} />
          <Route path="live"     element={<LiveMonitor />} />
          <Route path="research" element={<ResearchEvents />} />
          <Route path="recovery" element={<Recovery />} />
          <Route path="debug"    element={<Debug />} />
          <Route path="settings" element={<Settings />} />
        </Route>
      </Routes>
    </Suspense>
  );
}


