# OfflineForge — minimal CPU image
# Pure numpy + scikit-learn, no GPU required.
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt requirements.lock.txt ./
# Install locked deps for reproducible runs
RUN pip install --no-cache-dir -r requirements.lock.txt

COPY . .
RUN pip install --no-cache-dir -e ".[dev]"

# Default: run the full benchmark and print the summary
CMD ["python", "-m", "offlineforge.cli", "--out", "benchmark.json"]
