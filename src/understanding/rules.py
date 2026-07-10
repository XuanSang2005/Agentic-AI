"""Rule-based extractor: regex cho location (city/district/landmark theo gazetteer),
time ("sau 11 giờ tối", "24/7", "tối nay"), price ("giá rẻ", "miễn phí").

Chạy TRƯỚC và độc lập với LLM — cũng là planner chính ở deterministic mode.
"""

# TODO Ngày 1
