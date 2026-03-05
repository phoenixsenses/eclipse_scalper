import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import "./index.css";
import { DashboardAuthProvider } from "./context/AuthContext";
import { ApiErrorProvider } from "./context/ApiErrorContext";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ApiErrorProvider>
      <DashboardAuthProvider>
        <BrowserRouter>
          <App />
        </BrowserRouter>
      </DashboardAuthProvider>
    </ApiErrorProvider>
  </React.StrictMode>
);
