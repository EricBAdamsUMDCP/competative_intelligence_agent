# docker/airflow/Dockerfile
FROM apache/airflow:2.6.3

USER root

# Install additional system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Install Python dependencies
COPY requirements-airflow.txt /requirements-airflow.txt
RUN pip install --no-cache-dir -r /requirements-airflow.txt

# Copy DAGs, plugins, etc.
COPY dags/ ${AIRFLOW_HOME}/dags/
COPY plugins/ ${AIRFLOW_HOME}/plugins/

# Set Python path to include project modules
ENV PYTHONPATH=${PYTHONPATH}:${AIRFLOW_HOME}