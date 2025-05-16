# docker/dashboard/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-dashboard.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-dashboard.txt

# Copy application code
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Command to run Streamlit
CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
