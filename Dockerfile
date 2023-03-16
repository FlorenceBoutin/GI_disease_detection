# FROM python:3.10.6-buster

# COPY requirements.txt /requirements.txt
# COPY api /api
# COPY app_main.py /app_main.py
# COPY ml_logic /ml_logic
# COPY models /models
# RUN pip install --upgrade pip
# RUN pip install -r requirements.txt
# RUN pip install python-multipart
# # CMD uvicorn api.fast_main:app --reload --host 0.0.0.0
# CMD uvicorn api.fast_main:app --reload --host 0.0.0.0 --port $PORT
# EXPOSE 8000
