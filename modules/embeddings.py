import os
import time
import hashlib
import json
from openai import OpenAI
from openai import RateLimitError, APIError
from sentence_transformers import SentenceTransformer
from llama_index.embeddings.openai import OpenAIEmbedding
import logging
from typing import List

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.expanduser('~'), '.chatbot_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

def get_local_embeddings(texts):
    logger.info("Using local embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(texts).tolist()

def get_openai_embeddings(client, list_of_text, engine, max_retries=5):
    logger.info(f"Attempting to get OpenAI embeddings using model: {engine}")
    models = ["text-embedding-3-small", "text-embedding-ada-002"]
    
    for attempt in range(max_retries):
        for model in models:
            try:
                logger.info(f"Trying to get embeddings with model: {model}")
                response = client.embeddings.create(input=list_of_text, model=model)
                logger.info(f"Successfully got OpenAI embeddings using model: {model}")
                return [item.embedding for item in response.data]
            except RateLimitError:
                logger.warning(f"Rate limit hit for model {model}. Attempt {attempt + 1}/{max_retries}")
                time.sleep(2 ** attempt)  # Exponential backoff
            except APIError as e:
                logger.error(f"API error for model {model}: {e}")
                if model == models[-1] and attempt == max_retries - 1:
                    break
                time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logger.error(f"Unexpected error in get_openai_embeddings for model {model}: {e}")
                if model == models[-1] and attempt == max_retries - 1:
                    break
                time.sleep(2 ** attempt)  # Exponential backoff
    
    logger.info("All embedding attempts failed. Falling back to local embeddings.")
    return get_local_embeddings(list_of_text)

def cache_embeddings(func):
    def wrapper(client, list_of_text, engine, **kwargs):
        logger.info("Checking embedding cache...")
        
        cached_results = []
        texts_to_embed = []
        
        for text in list_of_text:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            cache_file = os.path.join(CACHE_DIR, f"embed_{text_hash}.json")
            
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cached_results.append(json.load(f))
            else:
                texts_to_embed.append(text)
        
        if texts_to_embed:
            logger.info(f"Getting embeddings for {len(texts_to_embed)} texts...")
            new_embeddings = func(client, texts_to_embed, engine, **kwargs)
            for text, embedding in zip(texts_to_embed, new_embeddings):
                text_hash = hashlib.md5(text.encode()).hexdigest()
                cache_file = os.path.join(CACHE_DIR, f"embed_{text_hash}.json")
                with open(cache_file, 'w') as f:
                    json.dump(embedding, f)
                cached_results.append(embedding)
        
        logger.info("Finished getting/caching embeddings.")
        return cached_results
    return wrapper

@cache_embeddings
def get_embeddings(client, list_of_text, engine, **kwargs):
    return get_openai_embeddings(client, list_of_text, engine, **kwargs)

class CustomOpenAIEmbedding(OpenAIEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._primary_model = 'text-embedding-3-small'
        self._fallback_model = 'text-embedding-ada-002'
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._dimension = 1536
        logger.info(f"Initializing CustomOpenAIEmbedding with primary model: {self._primary_model}")

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return get_embeddings(self._client, texts, self._primary_model)

