# ---- Base image ----
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=true

# ---- Set working directory ----
WORKDIR /app

# ---- Install dependencies first (layer caching) ----
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Copy project files ----
COPY . .

# ---- Expose FastAPI port ----
EXPOSE 7860

# ---- Run the app via server/app.py ----
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
