FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir prometheus-client

# Copy application
COPY app.py .

# Expose metrics port
EXPOSE 8000

# Run the exporter
CMD ["python", "app.py"]
