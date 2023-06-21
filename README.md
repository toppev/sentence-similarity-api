# sentence-similarity-api

A hobby project to learn new cool stuff.

## What is it?

Flask/Gunicorn API deployed as [Rapid API](https://rapidapi.com/) for comparing semantic similarity using [Sentence Transformers](https://sbert.net/).

The API is no longer available on Rapid API, but you can run it yourself.

## Features
- HTTP API for comparing semantic similarity of two sentences (also, multilingual)
- Redis Caching (per sentence)
- Dockerized
- CI/CD with Github Actions + Integration tests
- OpenAPI spec

## Getting Started

1. Install [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/).
2. Copy `.env.example` to `.env` and fill in the values.
3. Start with `docker-compose up --build`.