from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

def setup_query_engine(index_set, blog_files):
    logger.info("Setting up query engine...")
    individual_query_engine_tools = []

    for blog_file in blog_files:
        if blog_file in index_set:
            individual_query_engine_tools.append(
                QueryEngineTool(
                    query_engine=index_set[blog_file].as_query_engine(),
                    metadata=ToolMetadata(
                        name=f"idx_{Path(blog_file).stem[:20]}",
                        description=f"useful for when you want to answer queries about the cybersecurity red team and bug bounty information in {Path(blog_file).stem}",
                    ),
                )
            )
        else:
            logger.warning(f"Skipping {blog_file} as it was not successfully indexed.")

    if not individual_query_engine_tools:
        logger.error("No query engine tools created. Check if any files were successfully indexed.")
        return None, []

    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=individual_query_engine_tools,
        llm=LlamaOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY")),
    )

    logger.info("Query engine setup complete.")
    return query_engine, individual_query_engine_tools
