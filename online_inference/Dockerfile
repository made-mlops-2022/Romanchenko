ARG ARCH=arm64/v8
FROM --platform=linux/${ARCH} python:3.9.15-slim-bullseye
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./src /code/src
COPY ./models /code/models
EXPOSE 80
ENV USE_S3=False
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]