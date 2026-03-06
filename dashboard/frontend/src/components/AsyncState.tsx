import React from "react";

interface AsyncStateProps {
  loading: boolean;
  error: Error | null;
  isEmpty?: boolean;
  loadingText?: string;
  emptyText?: string;
  children: React.ReactNode;
}

export default function AsyncState({
  loading,
  error,
  isEmpty = false,
  loadingText = "Loading...",
  emptyText = "No data",
  children,
}: AsyncStateProps) {
  if (loading) {
    return <div style={{ color: "var(--muted)", padding: 8 }}>{loadingText}</div>;
  }
  if (error) {
    return <div style={{ color: "var(--red)", padding: 8 }}>{error.message}</div>;
  }
  if (isEmpty) {
    return <div style={{ color: "var(--muted)", padding: 8 }}>{emptyText}</div>;
  }
  return <>{children}</>;
}
