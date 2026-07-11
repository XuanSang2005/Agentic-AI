import type { PlaceResult, QueryPlanDict } from "@/lib/api-types";
import { bearingRad, haversineKm } from "@/lib/geo";

const SIZE = 160;
const CENTER = SIZE / 2;
const R_MAX = SIZE / 2 - 16;
const RINGS = [1 / 3, 2 / 3, 1];

export function RadialMap({
  plan,
  results,
  landmarkLabel,
}: {
  plan: QueryPlanDict | null | undefined;
  results: PlaceResult[];
  landmarkLabel?: string | null;
}) {
  if (!plan?.resolved_coord) return null;

  const center = { lat: plan.resolved_coord[0], lon: plan.resolved_coord[1] };
  const points = results.map((res, i) => ({
    i,
    km: haversineKm(center, res.coordinates),
    brg: bearingRad(center, res.coordinates),
  }));
  const maxKm = Math.max(...points.map((p) => p.km), 0.5);
  const caption = landmarkLabel || plan.landmark || "focus";
  const outer = maxKm < 1 ? `${Math.round(maxKm * 1000)} m` : `${maxKm.toFixed(1)} km`;

  return (
    <div className="map show">
      <svg
        width={SIZE}
        height={SIZE}
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        role="img"
        aria-label="Result distances around the landmark"
      >
        {RINGS.map((f) => (
          <circle
            key={f}
            cx={CENTER}
            cy={CENTER}
            r={R_MAX * f}
            fill="none"
            stroke="rgba(22,40,28,.10)"
            strokeWidth="1"
          />
        ))}
        <circle cx={CENTER} cy={CENTER} r="3.5" fill="#0F6B45" />
        {points.map((p) => {
          const rr = R_MAX * Math.pow(p.km / maxKm, 0.6);
          const x = CENTER + rr * Math.sin(p.brg);
          const y = CENTER - rr * Math.cos(p.brg);
          return (
            <g key={p.i}>
              <circle cx={x} cy={y} r="5" fill="#1E9E6A" opacity={p.i === 0 ? 1 : 0.5} />
              <text
                x={x + 8}
                y={y + 4}
                fontSize="10"
                fill="#5C6B60"
                style={{ fontFamily: "var(--mono)" }}
              >
                {p.i + 1}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="map-caption">
        {caption} - outer ring ~{outer}
      </div>
    </div>
  );
}
