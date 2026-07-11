from __future__ import annotations

import gradio as gr
from fastapi import FastAPI, Response

from src.tasco_query.contracts import QueryResponse, QueryUnderstandRequest
from src.tasco_query.demo import build_demo
from src.tasco_query.service import get_service


def create_app(*, mount_demo: bool = True) -> FastAPI:
    app = FastAPI(
        title="Vietnamese Map Query Understanding",
        version="0.2.0",
        description=(
            "Offline-first query-understanding API with optional validated model assistance "
            "and structured tracing."
        ),
    )

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, object]:
        return get_service().health()

    @app.post(
        "/v1/query/understand",
        response_model=QueryResponse,
        tags=["query"],
        summary="Understand a noisy map query",
        description=(
            "Returns the compact evaluation-compatible response by default. Set "
            "`include_trace` to `true` for a bounded structured transformation trace."
        ),
    )
    async def understand(
        request: QueryUnderstandRequest,
        response: Response,
    ) -> QueryResponse:
        result = get_service().understand(request)
        response.headers["X-Trace-Id"] = result.trace_id
        return result.response

    if mount_demo:
        app = gr.mount_gradio_app(app, build_demo(), path="/demo")
    return app


app = create_app()
