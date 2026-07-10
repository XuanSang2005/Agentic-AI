"""LLM planner: 1 call duy nhất/query → QueryPlan.

- Few-shot + WHITELIST từ lexicon/attribute_concepts.yaml: LLM chỉ được nhả token
  có thật trong 82 nhãn, không chế từ mới.
- "nổi tiếng"→ranking_bias=popularity, "giá rẻ"→price, "quán cà phê"→category, "gần X"→location.
- Fallback: rules.py (deterministic mode, không cần API key).
"""

# TODO Ngày 2
