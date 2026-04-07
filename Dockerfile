FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

CMD ["sh", "-c", "python openenv_validate.py && python app.py"]
