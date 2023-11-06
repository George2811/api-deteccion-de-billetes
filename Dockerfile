FROM python:3.10.6

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y libgl1

CMD uvicorn main:app --port=8000 --host=0.0.0.0
