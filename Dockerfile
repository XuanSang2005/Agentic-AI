# Tasco Semantic Search — image tự chứa 100%: model + embedding cache bake lúc BUILD,
# runtime offline hoàn toàn (HF_HUB_OFFLINE=1), đúng pitch deterministic/offline.
# Target: HF Spaces (Docker SDK, port 7860) — chạy được ở mọi platform container khác.
FROM python:3.11-slim

WORKDIR /app

# torch PHẢI cài từ CPU index trước — mặc định PyPI trên linux kéo CUDA wheels ~3GB.
# Pin đúng phiên bản trong requirements.txt; bước cài requirements sau sẽ thấy
# torch==2.8.0 đã thỏa mãn và bỏ qua.
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cache model/embedding nằm trong image (ghi được bởi user 1000 của HF Spaces)
ENV HF_HOME=/app/.cache/huggingface

COPY . .

# Bake lúc build: tải e5-small + encode 111 POI ra data/cache/*.npy + warmup.
# Sau bước này container KHÔNG cần network nữa.
RUN python -c "from src.search import SearchService; SearchService()" \
    && rm -rf /app/.cache/huggingface/xet 2>/dev/null; true

# Runtime ép offline: không stall check remote lúc startup, chứng minh self-contained
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# HF Spaces chạy container bằng uid 1000
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
