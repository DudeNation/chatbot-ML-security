from llama_index.readers.file import UnstructuredReader
from typing import Dict, List
from pathlib import Path
from llama_index.core import Document
from unstructured.partition.html import partition_html
import logging

logger = logging.getLogger(__name__)

class CustomUnstructuredReader(UnstructuredReader):
    def load_data(self, file: Path, extra_info: Dict = {}) -> List[Document]:
        """Load data from the file."""
        try:
            elements = partition_html(filename=str(file), include_metadata=False)
            metadata = {"file_name": file.name, **extra_info}
            return [Document(text=element.text, metadata=metadata) for element in elements]
        except Exception as e:
            logger.error(f"Error loading data from {file}: {str(e)}")
            return []
