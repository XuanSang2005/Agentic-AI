"""Kết hợp 7 signal với trọng số ĐIỀU KIỆN theo query_category.

Tune bằng coordinate ascent trên 60 câu eval, leave-one-out, ít weight — tránh overfit.
Trả về score tổng + breakdown từng signal (explainability).
"""

# TODO Ngày 1-2
