"use client";

import { useEffect, useState } from "react";
import { useSearch } from "@/lib/use-search";
import { LensIcon, PinIcon } from "./icons";
import { QueryPlanPanel } from "./query-plan-panel";
import { ResultCard } from "./result-card";
import { ResultSkeleton } from "./result-skeleton";

const PRESETS = [
  "cafe có wifi gần hồ gươm",
  "nơi ăn uống lãng mạn view thành phố",
  "quiet coffee shop to work near hoan kiem",
  "quan cafe yen tinh lam viec gan ho guom",
  "bv gần đây",
  "ks gần biển đà nẵng",
];
const HERO = PRESETS[1];

export function SearchExperience() {
  const [query, setQuery] = useState(HERO);
  const { status, data, activeQuery, elapsedMs, errorMessage, run } = useSearch();

  useEffect(() => {
    run(HERO);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const firstExplanation = data?.results[0]?.explanation;
  const planCoord = firstExplanation?.plan?.resolved_coord ?? null;

  let statline = "offline - deterministic";
  if (status === "success" || status === "empty") {
    statline = `offline - deterministic - ${elapsedMs} ms`;
  } else if (status === "error") {
    statline = "offline - service unreachable";
  }

  return (
    <>
      <header>
        <div className="brand">
          <span className="pin">
            <PinIcon />
          </span>
          <span className="bname">Tasco Maps</span>
          <span className="btrack">AABW 2026 - Track 2</span>
        </div>
        <h1>
          Semantic search, <em>every score explained</em>
        </h1>
        <p className="sub">
          BM25 + dense union pool with multi-signal rerank. Fully offline, deterministic,
          not a single LLM call.
        </p>
      </header>

      <div className="search">
        <LensIcon />
        <input
          id="q"
          type="text"
          autoComplete="off"
          spellCheck={false}
          placeholder="Search cafes, restaurants, ATMs..."
          aria-label="Search query"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && query.trim()) run(query.trim());
          }}
        />
        <span className="key">Enter</span>
      </div>

      <div className="chips" role="group" aria-label="Demo queries">
        {PRESETS.map((preset) => (
          <button
            key={preset}
            type="button"
            className="chip"
            aria-pressed={preset === activeQuery}
            onClick={() => {
              setQuery(preset);
              run(preset);
            }}
          >
            {preset}
          </button>
        ))}
      </div>

      {status === "success" && firstExplanation && (
        <QueryPlanPanel
          explanation={firstExplanation}
          submittedQuery={activeQuery}
          results={data!.results}
        />
      )}

      {status === "empty" && <div className="state">No results for this query.</div>}
      {status === "error" && <div className="state">{errorMessage}</div>}

      <div className="results" aria-live="polite">
        {status === "loading" &&
          [0, 1, 2].map((i) => <ResultSkeleton key={i} />)}
        {status === "success" &&
          data!.results.map((res, i) => (
            <ResultCard key={res.id} result={res} index={i} planCoord={planCoord} />
          ))}
      </div>

      <footer>
        <div className="status">
          <span>tasco-semantic-search - /v1/search?explain=true</span>
          <span>{statline}</span>
        </div>
      </footer>
    </>
  );
}
