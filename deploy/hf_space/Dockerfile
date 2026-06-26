# Hugging Face Space (Docker) - trains + serves Shannon's Gambit agents.
FROM python:3.11-slim

# Serve-only by default: a CPU Space that also self-plays starves the web server,
# so /healthz times out and the Space restart-loops. Train on HF Jobs instead
# (deploy/hf_job). Override TRAIN_ENABLED=1 only on a Space you don't serve from.
ENV PYTHONUNBUFFERED=1 \
    MODELS_DIR=/data/models \
    TRAIN_ENABLED=0 \
    STOCKFISH_PATH=/usr/games/stockfish

WORKDIR /app
# git: pip-install the package from GitHub; stockfish: reliable Elo-matched opponents.
RUN apt-get update && apt-get install -y --no-install-recommends git stockfish && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# HF Spaces route to port 7860.
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
