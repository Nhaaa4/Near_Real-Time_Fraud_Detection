FROM python:3.14.0-slim-bookworm
WORKDIR /app
COPY requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

COPY . .

EXPOSE 5500