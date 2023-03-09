FROM python:3.10.6-buster

COPY requirements.txt /requirements.txt
COPY api /api
COPY ml_logic /ml_logic
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --reload --host 0.0.0.0 --port $PORT
EXPOSE 8000
