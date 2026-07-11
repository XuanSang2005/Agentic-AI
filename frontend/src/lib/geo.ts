export interface LatLon {
  lat: number;
  lon: number;
}

export function haversineKm(a: LatLon, b: LatLon): number {
  const R = 6371;
  const r = Math.PI / 180;
  const dLat = (b.lat - a.lat) * r;
  const dLon = (b.lon - a.lon) * r;
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(a.lat * r) * Math.cos(b.lat * r) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}

/** Bearing in radians from `center` to `point`, 0 = north, clockwise. */
export function bearingRad(center: LatLon, point: LatLon): number {
  const r = Math.PI / 180;
  return Math.atan2(
    (point.lon - center.lon) * Math.cos(center.lat * r),
    point.lat - center.lat,
  );
}

export function fmtDist(km: number): string {
  return km < 1 ? `${Math.round(km * 1000)} m` : `${km.toFixed(1)} km`;
}
