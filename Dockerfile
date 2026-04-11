# Use lightweight slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/tmp/huggingface

# Set working directory
WORKDIR /app

# Install only essential lightweight dependencies
# Keeping it minimal for 2 vCPU / 8 GB RAM limits
RUN pip install --no-cache-dir \
    openai \
    pydantic \
    pyyaml \
    fastapi \
    gradio \
    uvicorn

# Copy project files
# We only need the core logic and inference script
COPY env.py .
COPY schemas.py .
COPY tasks.py .
COPY models.py .
COPY inference.py .
COPY openenv.yaml .
COPY app.py .
COPY server/ server/

# Create a non-root user for security (required by many HF evaluators)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Keeps the Space awake on Hugging Face
EXPOSE 7860
CMD ["python", "app.py"]
