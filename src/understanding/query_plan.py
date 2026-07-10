"""QueryPlan dataclass — output chuẩn của L1, input của L2/L3.

Slot: category, attributes[] (chỉ token trong whitelist 82 nhãn), 
location {city, district, landmark}, time, price, ranking_bias, language.
Nhớ bẫy P009: "trên đường đi Hạ Long" là ngữ cảnh, KHÔNG map vào location filter.
"""

# TODO Ngày 1 (shape trước, LLM planner Ngày 2)
