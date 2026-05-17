FROM python:3.10.6-slim

WORKDIR /app

# libgomp1: required by faiss-cpu
# gcc: required to compile some pip packages from source
RUN apt-get update && apt-get install -y \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements before code so Docker can cache the install layer
# and skip reinstalling when only application code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY src/ ./src/
COPY samples/ ./samples/

EXPOSE 8000

# --workers 1 because multiple workers each load the models separately,
# would unnecessarily multiply memory usage beyond the ECS task allocation if there were more workers.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]