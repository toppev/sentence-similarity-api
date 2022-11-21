from asgiref.wsgi import WsgiToAsgi
from flask import Flask, request, Response
import logging
import ujson as json

import redis.asyncio as async_redis

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
redis = async_redis.Redis(host='redis', port=6379)
app = Flask(__name__)

app.logger.info('Loading model...')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
# model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
app.logger.info('Model loaded')


async def get_similarities(source, candidates) -> list:
    source_embedding = (await encode_with_cache([source]))[0]
    embeddings = await encode_with_cache(candidates)
    return [{
        'score': round(get_similarity(source_embedding, embeddings[i]), 3),
        'candidate': candidates[i]
    } for i in range(len(candidates))]


async def encode_with_cache(messages) -> list:
    # Either the embeddings or None (len(cached)=len(messages))
    cached = await redis.mget(messages)
    new_embeddings = model.encode([messages[i] for i in range(len(messages)) if not cached[i]])

    embeddings = []
    pipeline = redis.pipeline()
    for i in range(len(messages)):
        if cached[i]:
            embeddings.append(np.array(json.loads(cached[i])))
        else:
            embedding = new_embeddings[0]
            embeddings.append(embedding)
            new_embeddings = new_embeddings[1:]
            pipeline.set(messages[i], json.dumps(embedding.tolist()), ex=60 * 60 * 12)

    await pipeline.execute()

    return embeddings


def get_similarity(msg_embeddings, embeddings) -> float:
    return cosine_similarity(msg_embeddings.reshape(1, -1), np.array(embeddings).reshape(1, -1))[0][0].item()


@app.route('/similarity', methods=["POST"])
async def similarity():
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

    results = await get_similarities(source, candidates)
    results.sort(key=lambda x: x['score'], reverse=True)

    return dict({
        'results': results,
    })


@app.route('/ping')
def ping():
    return dict({'status': 'ok'})


asgi_app = WsgiToAsgi(app)
