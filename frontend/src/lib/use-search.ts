"use client";

import { useCallback, useRef, useState } from "react";
import { ApiError, searchPlaces } from "./api-client";
import type { SearchResponse } from "./api-types";

export type SearchStatus = "idle" | "loading" | "success" | "empty" | "error";

export function useSearch() {
  const [status, setStatus] = useState<SearchStatus>("idle");
  const [data, setData] = useState<SearchResponse | null>(null);
  const [activeQuery, setActiveQuery] = useState("");
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const inflight = useRef<AbortController | null>(null);

  const run = useCallback(async (query: string) => {
    setActiveQuery(query);
    inflight.current?.abort();
    const controller = new AbortController();
    inflight.current = controller;
    setStatus("loading");
    const t0 = performance.now();
    try {
      const resp = await searchPlaces({ q: query, limit: 3, explain: true }, controller.signal);
      setElapsedMs(Math.round(performance.now() - t0));
      setData(resp);
      setStatus(resp.results.length ? "success" : "empty");
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setData(null);
      setErrorMessage(
        err instanceof ApiError
          ? err.message
          : "Cannot reach the service - make sure `make api` is running.",
      );
      setStatus("error");
    }
  }, []);

  return { status, data, activeQuery, elapsedMs, errorMessage, run };
}
