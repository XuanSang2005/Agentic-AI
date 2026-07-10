"""FastAPI app.

- GET /v1/search (+ alias /search, /v1/geocode-search) → PlaceResult DTO theo docs/tasco_api.pdf
- GET /health
- Auth pluggable: Authorization Bearer hoặc X-API-Key (mock chấp nhận không auth).
"""

# TODO Ngày 1 (sau khi search() chạy end-to-end)
