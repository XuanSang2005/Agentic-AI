from __future__ import annotations

from typing import Any, cast

import gradio as gr

from src.tasco_query.contracts import QueryUnderstandRequest, UserLocation
from src.tasco_query.service import get_service

EXAMPLES = [
    ["hoot vjt lộn trênn đg đbp", "", ""],
    ["atm vcb q7", "", ""],
    ["quán cf sống ảo mở tới 12 đêm", "Quận 3", "TP Hồ Chí Minh"],
    ["đường tới sân bay tsn", "", ""],
    ["cây xăng gần đây", "", ""],
    ["cf yên tĩnh để học q3", "", ""],
    ["quán ăn gia đình có phòng riêng", "", ""],
    ["circle k gần đây mở 24h", "", ""],
    ["địa chỉ bv chợ rẫy", "", ""],
    ["đi từ q1 tới sân bay", "", ""],
]


def understand_for_demo(query: str, area: str, city: str) -> dict[str, Any]:
    location = UserLocation(area=area or None, city=city or None) if area or city else None
    result = get_service().understand(QueryUnderstandRequest(query=query, location=location))
    return result.response.model_dump()


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Vietnamese Map Query Understanding") as demo:
        gr.Markdown("# Vietnamese Map Query Understanding\nLocal deterministic Release 1 demo")
        query = gr.Textbox(label="Noisy query", placeholder="quán cf sống ảo mở tới 12 đêm")
        with gr.Row():
            area = gr.Textbox(label="Area (optional)")
            city = gr.Textbox(label="City (optional)")
        output = gr.JSON(label="Understanding result")
        submit = gr.Button("Understand", variant="primary")
        submit.click(understand_for_demo, [query, area, city], output)
        query.submit(understand_for_demo, [query, area, city], output)
        gr.Examples(EXAMPLES, [query, area, city])
    return cast(gr.Blocks, demo)
