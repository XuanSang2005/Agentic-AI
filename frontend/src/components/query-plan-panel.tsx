import type { Explanation, InterpretedPlan } from "@/lib/api-types";
import { RadialMap } from "./radial-map";
import type { PlaceResult } from "@/lib/api-types";

const SLOT_DEFS: Array<[key: keyof InterpretedPlan, label: string]> = [
  ["categories", "category"],
  ["attributes", "attributes"],
  ["excluded", "excluded"],
  ["city", "city"],
  ["district", "district"],
  ["landmark", "landmark"],
];

function asList(value: string[] | string | null | undefined): string[] {
  if (value == null) return [];
  return Array.isArray(value) ? value : [value];
}

export function QueryPlanPanel({
  explanation,
  submittedQuery,
  results,
}: {
  explanation: Explanation;
  submittedQuery: string;
  results: PlaceResult[];
}) {
  const interpreted = explanation.interpreted ?? ({} as InterpretedPlan);
  const plan = explanation.plan;
  const nq = explanation.normalized_query;
  const showUnderstood = !!nq && nq.trim() !== submittedQuery.trim();

  const slots = SLOT_DEFS.map(([key, label]) => {
    const vals = asList(interpreted[key]);
    if (!vals.length) return null;
    const neg = key === "excluded";
    return (
      <div className="slot" key={key}>
        <div className="slot-name">{label}</div>
        <div className="tokens">
          {vals.map((v, i) => (
            <span
              key={v}
              className={`token${neg ? " neg" : ""}`}
              style={{ animationDelay: `${i * 45}ms` }}
            >
              {v}
            </span>
          ))}
        </div>
      </div>
    );
  }).filter(Boolean);

  return (
    <section className="plan show" aria-label="Query plan">
      <div className="plan-main">
        <div className="panel-label">Query plan - how L1 reads this query</div>
        {showUnderstood && (
          <div className="understood show">
            <span className="u-label">understood as</span>
            <span className="u-text">{nq}</span>
          </div>
        )}
        <div className="slots">
          {slots.length ? (
            slots
          ) : (
            <div className="slot">
              <div className="slot-name">plan</div>
              <div className="tokens">
                <span className="token empty">no structured slots - pure relevance ranking</span>
              </div>
            </div>
          )}
        </div>
        {plan?.resolved_coord && (
          <div className="coord">
            <span>resolved_coord</span> {plan.resolved_coord[0].toFixed(4)},{" "}
            {plan.resolved_coord[1].toFixed(4)}
          </div>
        )}
      </div>
      <RadialMap plan={plan} results={results} landmarkLabel={interpreted.landmark} />
    </section>
  );
}
