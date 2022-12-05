FROM python:3.10

RUN mkdir /app
WORKDIR /app

RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cpu
RUN wget -P ./lib/ https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz

ADD requirements.txt /app
RUN pip3 install -r requirements.txt

ADD . /app

# number of workers
ARG WEB_CONCURRENCY=1
ENV WEB_CONCURRENCY=${WEB_CONCURRENCY}

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 CMD curl -f http://localhost:8000/ping || exit 1

CMD ["gunicorn",  "-k", "uvicorn.workers.UvicornWorker",  "--bind", "0.0.0.0:8000",  "--timeout", "600",  "--access-logfile", "-",  "api-server:asgi_app"]