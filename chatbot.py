import os
from dotenv import load_dotenv
import re
import time
import hashlib
import json
import sys
import glob
import signal
import nltk
import nltk.tokenize
from openai import OpenAI
from openai import RateLimitError, APIError
from functools import wraps
from typing import List
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import UnstructuredReader
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.agent.openai import OpenAIAgent
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import logging
import glob
from modules.embeddings import CustomOpenAIEmbedding
from modules.document_indexer import setup_vector_indices
from modules.query_engine import setup_query_engine
import chainlit as cl
from typing import Dict as ThreadDict
from llama_index.core.memory import ChatMemoryBuffer
from unstructured.partition.html import partition_html
import traceback  # Add this import at the top of the file
import iso639
from llama_index.core import Document
import asyncio
import functools
from modules.image_analysis import analyze_image

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("Starting script...")
sys.stdout.flush()

# Load environment variables from .env file
load_dotenv(override=True)

# Get OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
print(f"OpenAI API Key in chatbot.py: {openai_api_key}")

# Check if the API key is set
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")

logger.info("API key loaded from environment variable...")

OFFLINE_MODE = False
MAX_RETRIES = 5
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.chatbot_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

def rate_limit(max_per_minute):
    min_interval = 60.0 / max_per_minute
    def decorator(func):
        last_called = [0.0]
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not OFFLINE_MODE:
                elapsed = time.time() - last_called[0]
                left_to_wait = min_interval - elapsed
                if left_to_wait > 0:
                    time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

def get_local_embeddings(texts):
    logger.info("Using local embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(texts).tolist()

@rate_limit(max_per_minute=60)
def get_openai_embeddings(client, list_of_text, engine, **kwargs):
    logger.info(f"Attempting to get OpenAI embeddings using model: {engine}")
    models = ["text-embedding-3-small", "text-embedding-ada-002"]
    
    for attempt in range(MAX_RETRIES):
        for model in models:
            try:
                logger.info(f"Trying to get embeddings with model: {model}")
                response = client.embeddings.create(input=list_of_text, model=model)
                logger.info(f"Successfully got OpenAI embeddings using model: {model}")
                return [item.embedding for item in response.data]
            except RateLimitError:
                logger.warning(f"Rate limit hit for model {model}. Attempt {attempt + 1}/{MAX_RETRIES}")
                time.sleep(2 ** attempt)  # Exponential backoff
            except APIError as e:
                logger.error(f"API error for model {model}: {e}")
                if model == models[-1] and attempt == MAX_RETRIES - 1:
                    break
                time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logger.error(f"Unexpected error in get_openai_embeddings for model {model}: {e}")
                if model == models[-1] and attempt == MAX_RETRIES - 1:
                    break
                time.sleep(2 ** attempt)  # Exponential backoff
    
    logger.info("All embedding attempts failed. Falling back to local embeddings.")
    return get_local_embeddings(list_of_text)

def cache_embeddings(func):
    @wraps(func)
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
    logger.info(f"get_embeddings: Called with engine: {engine}")
    if OFFLINE_MODE:
        logger.info("get_embeddings: Using local embeddings due to OFFLINE_MODE")
        return get_local_embeddings(list_of_text)
    logger.info(f"get_embeddings: Calling get_openai_embeddings with engine: {engine}")
    return get_openai_embeddings(client, list_of_text, engine, **kwargs)

class CustomOpenAIEmbedding(OpenAIEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._primary_model = 'text-embedding-3-small'
        self._fallback_model = 'text-embedding-ada-002'
        self._client = OpenAI(api_key=openai_api_key)
        self._dimension = 1536
        logger.info(f"Initializing CustomOpenAIEmbedding with primary model: {self._primary_model}")

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"CustomOpenAIEmbedding: Getting embeddings using primary model: {self._primary_model}")
        
        for attempt in range(MAX_RETRIES):
            try:
                embeddings = self._try_get_embeddings(texts, self._primary_model)
                logger.info(f"CustomOpenAIEmbedding: Received {len(embeddings)} embeddings from primary model")
                return embeddings
            except Exception as e:
                logger.warning(f"Failed to get embeddings with primary model. Attempt {attempt + 1}/{MAX_RETRIES}. Error: {e}")
                if attempt == MAX_RETRIES - 1:
                    logger.info(f"Falling back to {self._fallback_model}")
                    try:
                        embeddings = self._try_get_embeddings(texts, self._fallback_model)
                        logger.info(f"CustomOpenAIEmbedding: Received {len(embeddings)} embeddings from fallback model")
                        return embeddings
                    except Exception as e:
                        logger.error(f"Failed to get embeddings with fallback model. Error: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error("All embedding attempts failed. Falling back to local embeddings.")
        return get_local_embeddings(texts)

    def _try_get_embeddings(self, texts: List[str], model: str) -> List[List[float]]:
        response = self._client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]

Settings.embed_model = CustomOpenAIEmbedding()

logger.info("Custom embedding model set...")

def simple_sentence_tokenize(text, language='english'):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [sent.strip() for sent in sentences if sent.strip()]

nltk.tokenize.sent_tokenize = simple_sentence_tokenize
nltk.tokenize.word_tokenize = lambda text: text.split()

logger.info("Tokenization overrides set...")

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('popular', quiet=True)

nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

logger.info("NLTK data downloaded and set up...")

def lazy_load_documents(blog_files):
    loader = UnstructuredReader()
    for blog_file in blog_files:
        logger.info(f"Loading data from {blog_file}...")
        blog_docs = loader.load_data(
            file=Path(blog_file), split_documents=False
        )
        for d in blog_docs:
            d.metadata = {"source": blog_file}
        yield blog_file, blog_docs

def setup_vector_indices(blog_files):
    logger.info("Setting up vector indices...")
    Settings.chunk_size = 512
    index_set = {}

    for blog_file in blog_files:
        try:
            logger.info(f"Loading data from {blog_file}...")
            with open(blog_file, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Use a custom function to partition HTML without language detection
            elements = custom_partition_html(content)
            
            # Convert elements to documents
            docs = [Document(text=str(element), metadata={"source": blog_file}) for element in elements]
            
            logger.info(f"Creating index for {blog_file}...")
            storage_context = StorageContext.from_defaults()
            cur_index = VectorStoreIndex.from_documents(
                docs,
                storage_context=storage_context,
            )
            index_set[blog_file] = cur_index
            storage_context.persist(persist_dir=f"./storage/{Path(blog_file).stem}")
            logger.info(f"Persisted storage for {blog_file}")
        except Exception as e:
            logger.error(f"Error processing {blog_file}: {str(e)}")
            logger.error(traceback.format_exc())
            continue

    if not index_set:
        logger.error("No files were successfully processed.")
        raise ValueError("No files were successfully processed. Please check your data files.")

    logger.info("Vector indices setup complete.")
    return index_set

def custom_partition_html(html_content):
    # Simple HTML parsing without language detection
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    elements = []
    for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        elements.append(element.get_text())
    return elements

def setup_query_engine(index_set, blog_files):
    logger.info("Setting up query engine...")
    individual_query_engine_tools = [
        QueryEngineTool(
            query_engine=index_set[blog_file].as_query_engine(),
            metadata=ToolMetadata(
                name=f"idx_{Path(blog_file).stem[:20]}",  # Shortened name
                description=f"useful for when you want to answer queries about the cybersecurity red team and bug bounty information in {Path(blog_file).stem}",
            ),
        )
        for blog_file in blog_files
    ]

    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=individual_query_engine_tools,
        llm=LlamaOpenAI(model="gpt-4o", api_key=openai_api_key),
    )

    logger.info("Query engine setup complete.")
    return query_engine, individual_query_engine_tools

def timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out")

def get_local_chat_response(prompt):
    logger.info("Using local chat response...")
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    context = "Cybersecurity red team and bug bounty programs involve ethical hacking to identify and fix security vulnerabilities. Red teams simulate real-world attacks to test the effectiveness of security measures, while bug bounty programs incentivize independent researchers to report security flaws."
    result = qa_pipeline(question=prompt, context=context)
    return f"Based on local analysis, a possible answer to your query '{prompt}' is: {result['answer']} (Confidence: {result['score']:.2f}). Please note that this is a fallback response and may not be fully accurate or up-to-date."

def cache_query_result(func):
    cache = {}
    @wraps(func)
    def wrapper(agent, query, *args, **kwargs):
        if query in cache:
            logger.info("Using cached query result.")
            return cache[query]
        result = func(agent, query, *args, **kwargs)
        cache[query] = result
        return result
    return wrapper

@functools.lru_cache(maxsize=100)
def cached_query(query: str) -> str:
    # This function will cache the results of up to 100 most recent unique queries
    return query  # We'll use this as a key for our streaming cache

class StreamingCache:
    def __init__(self):
        self.cache = {}

    async def get_or_set(self, key, generator):
        if key not in self.cache:
            self.cache[key] = []
            async for item in generator:
                self.cache[key].append(item)
                yield item
        else:
            for item in self.cache[key]:
                yield item

streaming_cache = StreamingCache()

@cache_query_result
async def process_query(agent, query, max_retries=1, base_delay=1):
    logger.info("Processing query...")
    for attempt in range(max_retries):
        try:
            if OFFLINE_MODE:
                return get_local_chat_response(query)
            
            response_message = cl.Message(content="")
            await response_message.send()

            full_response = ""
            cache_key = cached_query(query)
            async for token in streaming_cache.get_or_set(cache_key, agent.stream_chat(query)):
                full_response += token
                await response_message.stream_token(token)

            await response_message.update(content=full_response)
            return full_response

        except RateLimitError:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit. Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                return "I'm sorry, but I've reached the rate limit. Please try again in a few moments."
        except APIError as e:
            logger.error(f"API error: {e}")
            if attempt < max_retries - 1:
                logger.info("Retrying due to API error...")
            else:
                return "I'm experiencing some technical difficulties. Please try again later."
        except Exception as e:
            logger.error(f"Error in getting response: {e}")
            if attempt < max_retries - 1:
                logger.info("Retrying...")
            else:
                return "I encountered an unexpected issue. Could you please rephrase your question?"
    
    return "I'm sorry, but I'm having trouble accessing the information at the moment. Please try again later."

@cl.on_chat_start
async def on_chat_start():
    logger.info("Starting new chat session...")
    
    # Initialize chat memory with a slightly higher token limit
    memory = ChatMemoryBuffer.from_defaults(token_limit=4000)
    
    # Set up the agent with memory
    blog_files = glob.glob("./data/*.html")
    index_set = setup_vector_indices(blog_files)
    query_engine, individual_query_engine_tools = setup_query_engine(index_set, blog_files)
    
    query_engine_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="sub_question_query_engine",
            description="useful for when you want to answer queries that require analyzing multiple cybersecurity blog posts",
        ),
    )
    
    tools = individual_query_engine_tools + [query_engine_tool]
    agent = OpenAIAgent.from_tools(
        tools,
        verbose=True,
        streaming=True,
        memory=memory,
        llm=LlamaOpenAI(model="gpt-4o", api_key=openai_api_key),
    )
    
    # Store the agent in the user session
    cl.user_session.set("agent", agent)
    
    logger.info("Chat session initialized with agent and memory.")

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    logger.info(f"Resuming chat session for thread: {thread['id']}")
    
    # Retrieve the agent from the user session
    agent = cl.user_session.get("agent")
    if not agent:
        logger.error("Agent not found in user session. Initializing a new one.")
        await on_chat_start()
        agent = cl.user_session.get("agent")
    
    logger.info("Chat session resumed with updated memory.")

async def main():
    logger.info("Starting main function...")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(600)  # Set timeout to 10 minutes
    
    try:
        await on_chat_start()
        
        while True:
            user_message = await cl.AskUserMessage(content="What would you like to know?").send()
            if user_message.content.lower() == "exit":
                break
            
            logger.info(f"Processing user input: {user_message.content}")
            
            image_analysis_result = None
            # Handle image upload
            if user_message.elements:
                for element in user_message.elements:
                    if isinstance(element, cl.Image):
                        image_analysis_result = await analyze_image(element)
                        logger.info(f"Image analysis result: {image_analysis_result}")
                        
                        # Render the image
                        await cl.Message(content="Here's the image you uploaded:").send()
                        await cl.Message(content="", elements=[element]).send()
                        
                        await cl.Message(content=f"Image analysis: {image_analysis_result}").send()
            
            agent = cl.user_session.get("agent")
            
            # Combine user message and image analysis result
            full_query = user_message.content
            if image_analysis_result:
                full_query += f"\n\nImage analysis: {image_analysis_result}"
            else:
                full_query += "\n\nNo image analysis available."
            
            response = await cl.make_async(process_query)(agent, full_query)
            logger.info(f"Agent response: {response}")
            await cl.Message(content=response).send()

        logger.info("Chat ended.")

    except TimeoutError:
        logger.error("Setup process timed out. Please check your data and API limits.")
    except Exception as e:
        logger.error(f"An error occurred in main: {e}", exc_info=True)
    finally:
        signal.alarm(0)  # Disable the alarm

if __name__ == "__main__":
    cl.run(main)
