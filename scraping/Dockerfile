FROM python:3.8.12-buster

WORKDIR /prod

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY scraper.py scraper.py

CMD python scraper.py
