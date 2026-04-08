FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: show help. Actual command passed via docker run / k8s args
ENTRYPOINT ["python", "cli.py"]
CMD ["--help"]
