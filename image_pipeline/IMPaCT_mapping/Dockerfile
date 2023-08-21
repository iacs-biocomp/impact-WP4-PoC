FROM python:3.9-slim-buster

COPY scripts /scripts

WORKDIR /scripts

RUN pip install pandas pytz openpyxl

CMD python imaging_occurrence.py && python imaging_feature.py && python measurement.py
