FROM apache/airflow:3.1.6

USER root
RUN mkdir -p /app/models

USER airflow
COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt

USER airflow