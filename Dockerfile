# ─────────────────────────────────────────────
#  Dockerfile — uses pre-built frontend
# ─────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY configs/    ./configs/
COPY backend/    ./backend/
COPY training/   ./training/
COPY setup.py    .
RUN pip install -e .

# Copy pre-built React frontend
COPY frontend_dist/ ./frontend/dist/

EXPOSE 7860

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "7860"]