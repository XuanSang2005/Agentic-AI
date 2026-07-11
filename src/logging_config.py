"""Structured logging (Phase 6): JSON ra stdout — 12-factor, log là stream.

- setup_logging() gắn 1 handler JSON vào root logger, level từ env LOG_LEVEL
  (default INFO). Idempotent — gọi nhiều lần không nhân đôi handler.
- Extra fields của record (logger.info("x", extra={...})) đi thẳng vào JSON.
- KHÔNG log secret/PII: middleware chỉ log path (không query string — có thể
  chứa lat/lon người dùng); client HTTP không bao giờ log URL chứa API key.
- Cố ý KHÔNG Prometheus/ELK/APM — quy mô nhỏ, stdout JSON là đủ quan sát.
"""
from __future__ import annotations

import json
import logging
import os
import sys

# Attr chuẩn của LogRecord — phần còn lại trong __dict__ là extra do caller đưa
_RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging() -> None:
    """Root → stdout JSON, level từ LOG_LEVEL. Idempotent (đánh dấu handler)."""
    root = logging.getLogger()
    if any(getattr(h, "_tasco_json", False) for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler._tasco_json = True  # cờ idempotent
    root.addHandler(handler)
    root.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
