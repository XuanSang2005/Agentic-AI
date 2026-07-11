-- Phase 5: data_version dùng chung cho đa-instance.
--
-- Nguồn sự thật về "data đã đổi chưa" là POSTGRES, không phải in-memory:
-- ingestion bump version TRONG CÙNG transaction với upsert POI (rollback →
-- version không tăng); mọi serve instance poll version, thấy mới hơn bản
-- đang giữ → reload qua atomic swap.

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value BIGINT NOT NULL
);

INSERT INTO meta (key, value) VALUES ('data_version', 1)
ON CONFLICT (key) DO NOTHING;
