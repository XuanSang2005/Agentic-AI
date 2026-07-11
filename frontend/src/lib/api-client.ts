import type { ErrorResponse, SearchResponse } from "./api-types";

export interface SearchParams {
  q: string;
  lat?: number;
  lon?: number;
  radiusMeters?: number;
  bbox?: string;
  category?: string;
  limit?: number;
  lang?: string;
  explain?: boolean;
}

export class ApiError extends Error {
  code: string;
  requestId: string;
  details?: Record<string, unknown>;

  constructor(body: ErrorResponse, status: number) {
    super(`${body.error.code} (${status}): ${body.error.message}`);
    this.name = "ApiError";
    this.code = body.error.code;
    this.requestId = body.requestId;
    this.details = body.error.details;
  }
}

function buildQuery(params: SearchParams): string {
  const qs = new URLSearchParams({ q: params.q });
  if (params.lat != null) qs.set("lat", String(params.lat));
  if (params.lon != null) qs.set("lon", String(params.lon));
  if (params.radiusMeters != null) qs.set("radiusMeters", String(params.radiusMeters));
  if (params.bbox) qs.set("bbox", params.bbox);
  if (params.category) qs.set("category", params.category);
  if (params.limit != null) qs.set("limit", String(params.limit));
  if (params.lang) qs.set("lang", params.lang);
  if (params.explain != null) qs.set("explain", String(params.explain));
  return qs.toString();
}

export async function searchPlaces(
  params: SearchParams,
  signal?: AbortSignal,
): Promise<SearchResponse> {
  const resp = await fetch(`/v1/search?${buildQuery(params)}`, { signal });
  const body = await resp.json();
  if (!resp.ok) throw new ApiError(body as ErrorResponse, resp.status);
  return body as SearchResponse;
}
