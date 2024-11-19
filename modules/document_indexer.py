from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from pathlib import Path
import logging
import traceback
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

def lazy_load_documents(blog_files):
    loader = CustomUnstructuredReader()
    for blog_file in blog_files:
        logger.info(f"Loading data from {blog_file}...")
        docs = loader.load_data(file=Path(blog_file), extra_info={"source": blog_file})
        if docs:
            yield blog_file, docs
        else:
            logger.warning(f"No documents loaded from {blog_file}")

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

def detect_language(text):
    try:
        return langdetect.detect(text)
    except:
        return "unknown"

def custom_partition_html(html_content):
    # Simple HTML parsing without language detection
    soup = BeautifulSoup(html_content, 'html.parser')
    elements = []
    for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        elements.append(element.get_text())
    return elements
