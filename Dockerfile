FROM python:3.8

RUN mkdir /app
WORKDIR /app

RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cpu

ADD requirements.txt /app
RUN pip3 install -r requirements.txt

ADD . /app

# number of workers
ENV WEB_CONCURRENCY=2

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 CMD curl -f http://localhost:8000/ping || exit 1

CMD ["gunicorn",  "-k", "uvicorn.workers.UvicornWorker",  "--bind", "0.0.0.0:8000",  "--timeout", "600",  "--access-logfile", "-",  "api-server:asgi_app"]