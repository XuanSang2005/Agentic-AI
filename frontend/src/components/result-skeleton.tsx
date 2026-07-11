export function ResultSkeleton() {
  return (
    <div className="skeleton-card" aria-hidden="true">
      <div className="card-top">
        <span className="skeleton-bar" style={{ width: 20, height: 13 }} />
        <span className="skeleton-bar" style={{ flex: 1, height: 21, marginLeft: 14 }} />
        <span className="skeleton-bar" style={{ width: 56, height: 26, borderRadius: 999 }} />
      </div>
      <div className="skeleton-bar" style={{ width: "60%", height: 14, marginTop: 14, marginLeft: 34 }} />
      <div className="skeleton-bar" style={{ width: "40%", height: 11, marginTop: 16, marginLeft: 34 }} />
    </div>
  );
}
