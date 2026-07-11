import type { ConstraintDetail, PlaceResult } from "@/lib/api-types";
import { fmtDist, haversineKm } from "@/lib/geo";
import { CheckIcon, HalfIcon, NoIcon } from "./icons";

const GREENS = ["#0F6B45", "#1E9E6A", "#3DBE8C", "#74D4AE", "#ADE6CF", "#D6F0E4"];

const SIG_LABELS: Record<string, string> = {
  dense_relevance: "dense",
  bm25_relevance: "bm25",
  category: "category",
  attr: "attr",
  city: "city",
  distance: "distance",
  rating: "rating",
  pop: "pop",
  name: "name",
};

function constraintGroup(d: ConstraintDetail): 0 | 1 | 2 {
  if (d.score >= 1) return 0;
  if (d.score > 0) return 1;
  return 2;
}

export function ResultCard({
  result,
  index,
  planCoord,
}: {
  result: PlaceResult;
  index: number;
  planCoord: [number, number] | null;
}) {
  const ex = result.explanation;

  let km: number | null = null;
  if (planCoord) {
    km = haversineKm({ lat: planCoord[0], lon: planCoord[1] }, result.coordinates);
  } else if (result.distanceMeters != null) {
    km = result.distanceMeters / 1000;
  }
  const distPct = km != null ? Math.max(6, Math.round(100 / (1 + km))) : 0;

  const sigParts = ex?.signals
    ? Object.entries(ex.signals)
        .map(([k, v]) => ({ k: SIG_LABELS[k] || k, v }))
        .filter((p) => p.v > 0.0005)
        .sort((a, b) => b.v - a.v)
    : [];
  const sigTotal = sigParts.reduce((s, p) => s + p.v, 0) || 1;

  const cons = ex?.constraints;
  const consDetail =
    cons && cons.total
      ? [...cons.detail].sort(
          (a, b) =>
            constraintGroup(a) - constraintGroup(b) ||
            a.priority - b.priority ||
            a.label.localeCompare(b.label),
        )
      : [];

  return (
    <article className="card" style={{ animationDelay: `${index * 80}ms` }}>
      <div className="card-top">
        <span className="rank">{index + 1}</span>
        <span className="name">{result.name}</span>
        <span className="score-col">
          <span className="score">{result.score.toFixed(3)}</span>
          <span className="score-label">relevance</span>
        </span>
      </div>
      <div className="meta">
        <span className="cat">{result.category}</span>
        {result.address}
      </div>

      {consDetail.length > 0 && (
        <div className="cons">
          <span
            className="badge-kn"
            data-tip={`meets ${cons!.satisfied}/${cons!.total} constraints`}
          >
            {cons!.satisfied}/{cons!.total}
          </span>
          {consDetail.map((d) => {
            const tip = `${d.type} - ${d.label} - score ${d.score}${d.satisfied ? "" : " - relaxed"}`;
            const group = constraintGroup(d);
            if (group === 0)
              return (
                <span className="ctoken ok" data-tip={tip} key={d.label}>
                  <CheckIcon />
                  {d.label}
                </span>
              );
            if (group === 1)
              return (
                <span className="ctoken part" data-tip={tip} key={d.label}>
                  <HalfIcon />
                  {d.label}
                  <span className="part-note">(partial)</span>
                  {!d.satisfied && <span className="relax-tag">relaxed</span>}
                </span>
              );
            return (
              <span className="ctoken relax" data-tip={tip} key={d.label}>
                <NoIcon />
                {d.label}
                <span className="relax-tag">relaxed</span>
              </span>
            );
          })}
        </div>
      )}

      {km != null && (
        <div className="dist-row">
          <span className="dist-val">{fmtDist(km)}</span>
          <span className="dist-track">
            <span className="dist-fill" style={{ width: `${distPct}%` }} />
          </span>
          <span className="dist-label">proximity</span>
        </div>
      )}

      {sigParts.length > 0 && (
        <div className="signals">
          <div className="sig-label">Signal contribution</div>
          <div className="sig-bar">
            {sigParts.map((p, j) => (
              <span
                key={p.k}
                className="sig-seg"
                style={{
                  width: `${((p.v / sigTotal) * 100).toFixed(2)}%`,
                  background: GREENS[Math.min(j, GREENS.length - 1)],
                }}
                title={`${p.k} ${p.v.toFixed(3)}`}
              />
            ))}
          </div>
          <div className="sig-legend">
            {sigParts.slice(0, 4).map((p, j) => (
              <span key={p.k}>
                <span
                  className="swatch"
                  style={{ background: GREENS[Math.min(j, GREENS.length - 1)] }}
                />
                {p.k} <b>{p.v.toFixed(3)}</b>
              </span>
            ))}
          </div>
        </div>
      )}
    </article>
  );
}
