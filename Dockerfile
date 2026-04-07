FROM python:3.11-slim

RUN useradd -m -u 1000 user

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chown -R user:user /app

USER user

EXPOSE 7860

ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
