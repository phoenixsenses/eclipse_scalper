import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import "./index.css";
import { DashboardAuthProvider } from "./context/AuthContext";
import { ApiErrorProvider } from "./context/ApiErrorContext";

const RootMode = import.meta.env.DEV ? React.Fragment : React.StrictMode;

ReactDOM.createRoot(document.getElementById("root")!).render(
  <RootMode>
    <ApiErrorProvider>
      <DashboardAuthProvider>
        <BrowserRouter>
          <App />
        </BrowserRouter>
      </DashboardAuthProvider>
    </ApiErrorProvider>
  </RootMode>
);
