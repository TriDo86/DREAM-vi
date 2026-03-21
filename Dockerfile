# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir ".[api,demo]"


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/     src/
COPY api/     api/
COPY configs/ configs/

# Checkpoint mounted at runtime via -v, or baked in for HF Spaces
# ENV DREAM_CHECKPOINT=checkpoints/dream_best.pt
# ENV DREAM_BACKBONE=sentence-transformers/LaBSE
# ENV DREAM_DEVICE=cpu

EXPOSE 7860

# Default: Gradio demo.
# To run the FastAPI instead: docker run ... -e CMD="uvicorn api.app:app --host 0.0.0.0 --port 7860"
CMD ["python", "api/demo.py"]
