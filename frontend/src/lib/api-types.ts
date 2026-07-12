/**
 * TypeScript mirror of the backend DTOs. Kept in sync by hand with:
 *   - src/api/dto.py           (Coordinates, PlaceResult, SearchMeta,
 *                                SearchResponse, ErrorBody, ErrorResponse)
 *   - src/search.py            (Explanation — the shape SearchService.search
 *                                builds under PlaceResult.explanation)
 *   - src/understanding/query_plan.py (QueryPlanDict, mirrors QueryPlan)
 *   - src/reasoning/constraints.py    (Constraints, ConstraintDetail)
 *
 * `explanation` is typed as a raw `dict` on the Python side, so it can't be
 * derived from the OpenAPI schema — if the shape of the dict built in
 * SearchService.search changes, update this file to match.
 */

export interface Coordinates {
  lat: number;
  lon: number;
}

export interface ConstraintDetail {
  type: "category" | "attribute" | "time" | "location" | "price";
  label: string;
  score: number;
  satisfied: boolean;
  priority: number;
}

export interface Constraints {
  total: number;
  satisfied: number;
  detail: ConstraintDetail[];
  relaxed: string[];
}

export interface QueryPlanDict {
  query: string;
  norm_query: string;
  categories: string[];
  attr_concepts: string[];
  neg_concepts: string[];
  city: string | null;
  want_pop: boolean;
  district: string | null;
  landmark: string | null;
  resolved_coord: [number, number] | null;
}

export interface InterpretedPlan {
  categories: string[];
  attributes: string[];
  excluded: string[];
  city: string | null;
  district: string | null;
  landmark: string | null;
}

export interface Explanation {
  plan: QueryPlanDict | null;
  expanded_query: string | null;
  typo_corrected: string | null;
  normalized_query: string | null;
  interpreted: InterpretedPlan | null;
  signals: Record<string, number>;
  constraints: Constraints | null;
}

export interface PlaceResult {
  id: string;
  type: string;
  name: string;
  label: string;
  address: string;
  category: string;
  coordinates: Coordinates;
  distanceMeters?: number;
  score: number;
  source: string;
  tags: string[];
  /** Verify status (policy A, display-only): "verified" | "unverified" |
   *  "active" (xlsx default — no verify info, render nothing). */
  status?: string;
  explanation?: Explanation;
}

export interface SearchMeta {
  limit: number;
  lang: string;
}

export interface SearchResponse {
  query: string;
  results: PlaceResult[];
  meta: SearchMeta;
}

export interface ErrorBody {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface ErrorResponse {
  error: ErrorBody;
  requestId: string;
}
