import os
import time
import ujson as json
import logging

from asgiref.wsgi import WsgiToAsgi
from flask import Flask, request, Response

import numpy as np
import fasttext
from sentence_transformers import SentenceTransformer, util

import redis.asyncio as async_redis

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

REDIS_NS = 'sentence-similarity'
redis = async_redis.Redis(host='redis', port=6379, socket_timeout=2)

rapid_proxy_secret = os.environ.get('RAPID_API_PROXY_SECRET', None)
app.logger.info('RAPID_API_PROXY_SECRET exists: %s', rapid_proxy_secret is not None)

app.logger.info('Loading model(s)...')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')  # scores: 68.70	50.82	59.76
multilingual_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')  # scores: 64.25	39.19	51.72

lang_detect = fasttext.load_model('lib/lid.176.ftz')
app.logger.info('Models loaded')


def get_model(lang_id):
    return model if lang_id == 'en' else multilingual_model


async def get_similarities(source, candidates, lang_id) -> list:
    source_embedding = (await encode_with_cache([source], lang_id))[0]
    embeddings = await encode_with_cache(candidates, lang_id)
    return [{
        'score': round(get_similarity(source_embedding, embeddings[i]), 3),
        'candidate': candidates[i]
    } for i in range(len(candidates))]


async def encode_with_cache(messages, lang_id) -> list:
    # Either the embeddings or None (len(cached)=len(messages))
    try:
        # TODO: append model name in the key? Helps with model changes
        cached = await redis.mget([f'{REDIS_NS}:embeddings:{lang_id}:{message}' for message in messages])
    except Exception as e:
        app.logger.error(e)
        cached = [None] * len(messages)

    new_embeddings = get_model(lang_id).encode([messages[i] for i in range(len(messages)) if not cached[i]])

    embeddings = []
    pipeline = redis.pipeline()
    for i in range(len(messages)):
        if cached[i] is not None:
            embeddings.append(np.array(json.loads(cached[i])))
        else:
            embedding = new_embeddings[0]
            embeddings.append(embedding)
            new_embeddings = new_embeddings[1:]
            pipeline.set(f'{lang_id}:{messages[i]}', json.dumps(embedding.tolist()), ex=60 * 60 * 12)

    try:
        await pipeline.execute()
    except Exception as e:
        app.logger.error(e)

    return embeddings


def get_similarity(msg_embeddings, embeddings) -> float:
    return util.cos_sim(msg_embeddings.reshape(1, -1), np.array(embeddings).reshape(1, -1))[0][0].item()


def predict_lang(sample):
    (label, prob) = lang_detect.predict(sample, k=1)
    return label[0].replace('__label__', ''), prob[0]


@app.route('/similarity', methods=["POST"])
async def similarity():
    start_time = time.perf_counter()

    rapid_key = request.headers.get('X-RapidAPI-Proxy-Secret')
    if rapid_proxy_secret and rapid_key != rapid_proxy_secret:
        return Response(status=403, response='Forbidden - Invalid X-RapidAPI-Proxy-Secret')

    source = request.json.get('source')
    candidates = request.json.get('candidates')

    app.logger.info(f'Source length {len(source)} and {len(candidates)} candidates')

    if not source or not candidates:
        return Response(status=400, response="Missing required fields")
    if len(source) > 512:
        return Response(status=400, response="Source text is too long (max 512 characters)")
    if len(candidates) > 100:
        return Response(status=400, response="Too many candidate sentences")
    if any(len(candidate) > 512 for candidate in candidates):
        return Response(status=400, response="One or more candidate sentences are too long (max 512 characters)")

    (lang_id, prob) = predict_lang(source + candidates[0] if len(candidates) > 0 else source)
    results = await get_similarities(source, candidates, lang_id if prob > 0.75 else None)
    results.sort(key=lambda x: x['score'], reverse=True)

    total_time = time.perf_counter() - start_time
    app.logger.info(f'Processed %s sentences in %s ms', len(candidates) + 1, round(total_time * 1000, 2))

    return dict({
        'results': results,
        'meta': {
            'lang': lang_id,
        }
    })


@app.route('/ping')
def ping():
    return dict({'status': 'ok'})


asgi_app = WsgiToAsgi(app)
