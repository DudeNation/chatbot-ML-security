import os
from pathlib import Path
import chainlit as cl
from modules.document_indexer import setup_vector_indices
from modules.query_engine import setup_query_engine
from llama_index.agent.openai import OpenAIAgent
import logging

logger = logging.getLogger(__name__)

def should_update_index(blog_file, storage_dir):
    index_file = os.path.join(storage_dir, "index_store.json")
    if not os.path.exists(index_file):
        return True
    return os.path.getmtime(blog_file) > os.path.getmtime(index_file)

async def update_indices_if_needed(blog_files):
    update_needed = any(should_update_index(blog_file, f"./storage/{Path(blog_file).stem}") for blog_file in blog_files)

    if update_needed:
        index_set = await cl.make_async(setup_vector_indices)(blog_files)
        if index_set:
            query_engine, tools = await cl.make_async(setup_query_engine)(index_set, blog_files)
            if query_engine and tools:
                new_agent = OpenAIAgent.from_tools(tools, verbose=True)
                cl.user_session.set("agent", new_agent)
                logger.info("Agent created successfully.")
                return True
            else:
                logger.error("Failed to create query engine or tools. Agent not created.")
        else:
            logger.error("No indices were created. Agent not created.")
        return False
    logger.info("No update needed for indices.")
    return True
