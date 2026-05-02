# -------- Base --------
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

# -------- Install only required system deps --------
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# -------- Install Python deps (optimized) --------
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# -------- Copy project --------
COPY . .

# -------- Expose --------
EXPOSE 8000

# -------- Run API --------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]