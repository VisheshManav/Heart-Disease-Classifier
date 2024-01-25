FROM python:3.12-slim-bullseye

RUN pip install pipenv

WORKDIR /app

COPY Pipfile* ./

RUN pipenv install --system --deploy

COPY  predict.py *.bin ./
COPY templates/ ./templates/

EXPOSE 9696

ENTRYPOINT gunicorn --bind 0.0.0.0:9696 predict:app