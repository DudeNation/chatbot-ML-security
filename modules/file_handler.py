import os
import tempfile
import chainlit as cl
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import requests
from bs4 import BeautifulSoup
import logging
import shutil

logger = logging.getLogger(__name__)

async def handle_file_upload(file: cl.File) -> str:
    if not file or not file.path:
        logger.error(f"Invalid file or empty path: {file}")
        return "Error: Invalid file or empty path"

    try:
        file_extension = os.path.splitext(file.name)[1].lower()

        if file_extension in ['.py', '.js', '.html', '.css', '.txt']:
            with open(file.path, 'r', encoding='utf-8') as f:
                content = f.read()
        elif file_extension == '.pdf':
            content = extract_pdf_content(file.path)
        elif file_extension in ['.docx', '.doc']:
            content = extract_word_content(file.path)
        elif file_extension in ['.xlsx', '.xls']:
            content = extract_excel_content(file.path)
        else:
            content = f"Unsupported file type: {file_extension}"

        return content
    except Exception as e:
        logger.error(f"Error processing file {file.name}: {str(e)}")
        return f"Error processing file: {str(e)}"

def extract_pdf_content(file_path: str) -> str:
    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        content = ""
        for page in pdf_reader.pages:
            content += page.extract_text() + "\n"
    return content

def extract_word_content(file_path: str) -> str:
    doc = Document(file_path)
    content = ""
    for para in doc.paragraphs:
        content += para.text + "\n"
    return content

def extract_excel_content(file_path: str) -> str:
    df = pd.read_excel(file_path)
    return df.to_string()

async def handle_url(url: str) -> str:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()
