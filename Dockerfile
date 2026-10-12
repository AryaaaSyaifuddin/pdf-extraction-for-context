FROM python:3.11-slim

# Mencegah Python buffering output stdout & stderr agar log langsung tampil di Docker
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install library pendukung sistem untuk PDF dan kompilasi dasar
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy daftar dependensi dan install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh source code proyek
COPY . .

# Port untuk Web UI / FastAPI
EXPOSE 8000

CMD ["python", "src/main.py"]