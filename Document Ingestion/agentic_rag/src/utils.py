import traceback
import os
import re
import shutil
import pandas as pd
import json
import uuid
from io import StringIO
from openai import AzureOpenAI
from datetime import datetime
from azure.storage.blob import BlobServiceClient, ContentSettings
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict
import fitz  # PyMuPDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from PIL import Image
from reportlab.lib.utils import ImageReader
from uuid import uuid4
import pdfplumber
import tiktoken
import nltk
from nltk.tokenize import sent_tokenize
from time import time
from azure.identity import (
    DefaultAzureCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider
)
from agentic_rag.src import (
    ACCOUNT_URL,
    ENV_NAME, 
    ENV_NAMES,
    STATUS_IN_PROGRESS,
    AZURE_OPENAI_SERVICE
)

base_dir = os.path.dirname(__file__)
nltk_path = os.path.join(base_dir, "..", "nltk_data")
nltk.data.path = [nltk_path]
print("FOUND DATA:", nltk.data.find("tokenizers/punkt"))

TOKENIZER_NAME = "o200k_base"
if ENV_NAME in ENV_NAMES:
    azure_credential = ManagedIdentityCredential()
else:
    azure_credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(azure_credential, "https://cognitiveservices.azure.com/.default")
blob_service_client = BlobServiceClient(account_url=ACCOUNT_URL, credential=azure_credential)
openai_client = AzureOpenAI(
    azure_endpoint=f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com",
    api_version="2024-02-15-preview",
    azure_ad_token_provider=token_provider,
)

def get_traceback(exception, message=""):
    return message + '\n' + ''.join(traceback.format_exception(None, exception, exception.__traceback__))

def contains_uuid4(text):
    # UUID4 pattern: 8-4-4-4-12 hexadecimal digits
    uuid4_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}'
    return bool(re.search(uuid4_pattern, text, re.IGNORECASE))

def download_blob(blobname, input_dir, container_name, save_as=None):
    blob_client_instance = blob_service_client.get_blob_client(container_name, blobname, snapshot=None)
    blob_data = blob_client_instance.download_blob()
    output_filename = save_as if save_as is not None else blobname
    with open(file=os.path.join(input_dir, output_filename), mode="wb") as sample_blob:
        sample_blob.write(blob_data.readall())

def get_processing_paths(base_dir, doc_id, blobname):
    input_dir = os.path.join(base_dir, doc_id)
    os.makedirs(input_dir, exist_ok=True)
    pdf_file_path = os.path.join(input_dir, blobname)
    return input_dir, pdf_file_path

def split_text_into_chunks(text: str, doc_name: str, doc_id: str, page: int, para_num: int, min_tokens: int = 250) -> List[Dict[str, Any]]:
    """
    Splits a single text string into smaller chunks, respecting sentence boundaries.
    This is used for recursively breaking down larger chunks.
    """
    sentences = sent_tokenize(text)
    tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
    chunks: List[Dict[str, Any]] = []
    current_sentences: List[str] = []
    current_token_count = 0
    for sentence in sentences:
        sentence_tokens = len(tokenizer.encode(sentence))
        if (current_token_count + sentence_tokens > min_tokens * 2) and (current_token_count >= min_tokens):
            chunk_text = " ".join(current_sentences)
            chunks.append({
                "documentId": doc_id,
                "chunkId": str(uuid.uuid4()), 
                "text": chunk_text, 
                "documentName": doc_name,
                "page": page,
                "paragraphNumber": para_num,
                "createdDate": datetime.now()
            })
            current_sentences = [sentence]
            current_token_count = sentence_tokens
        else:
            current_sentences.append(sentence)
            current_token_count += sentence_tokens   
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunks.append({
            "documentId": doc_id,
            "chunkId": str(uuid.uuid4()), 
            "text": chunk_text, 
            "documentName": doc_name,
            "page": page,
            "paragraphNumber": para_num,
            "createdDate": datetime.now()
        })
    return chunks

def get_chunks_obj(paragraphs: list, min_tokens: int = 500) -> list:
    """
    Splits each paragraph into chunks while preserving its page and paragraph number.
    """
    all_chunks = []
    if not paragraphs:
        return all_chunks
    for para in paragraphs:
        doc_name = para['document_name']
        doc_id = para['document_id']
        page = para['page']
        para_num = para['paragraph_number']
        text = para['text']
        chunks = split_text_into_chunks(
            text,
            doc_name,
            doc_id,
            page,
            para_num,
            min_tokens
        )
        all_chunks.extend(chunks)
    return all_chunks

def clear_dir(folder="images/", suffix=None, cur_dir=False):
    if cur_dir:
        shutil.rmtree(folder)
    else:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if suffix:
                    if (os.path.isfile(file_path) or os.path.islink(file_path)) and file_path.endswith(suffix):
                        os.unlink(file_path)
                else:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))

def check_incomplete_unicode(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                fonts = defaultdict(set)
                for char in page.chars:
                    font_name = char.get('fontname', '')
                    fonts[font_name].add(char.get('text', ''))
                for font_name in fonts:
                    if not font_name:  # If font name is missing
                        return True
                    problematic_indicators = [
                        'Symbol', 
                        'ZapfDingbats',
                        'MT Extra',
                        'Wingdings',
                        'Type3'
                    ]
                    if any(indicator in font_name for indicator in problematic_indicators):
                        return True
                    chars = fonts[font_name]
                    if any('\\u' in c.encode('unicode_escape').decode('ascii') for c in chars if c):
                        return True
                    if any(any(ord(c) in range(0xE000, 0xF8FF + 1) for c in char) 
                           for char in chars if char):
                        return True
            return False
    except Exception as e:
        err_msg = get_traceback(e, "Error processing PDF:")
        return True  # Return True to indicate potential issues if an error occurs

def check_encoding_and_convert_pdf_2_image(pdf_path):
    doc_dir = os.path.dirname(pdf_path)
    has_unicode_issues = check_incomplete_unicode(pdf_path)
    if not has_unicode_issues:
        return pdf_path, False
    try:
        pdf_document = fitz.open(pdf_path)
        output_pdf = os.path.join(doc_dir, os.path.basename(pdf_path).rsplit('.', 1)[0] + '_images.pdf')
        c = canvas.Canvas(output_pdf, pagesize=letter)
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            pix = page.get_pixmap(dpi=150)
            img_data = pix.tobytes("png")
            img = Image.open(BytesIO(img_data))
            img_width, img_height = img.size
            aspect = img_height / float(img_width)
            width = letter[0]
            height = width * aspect
            img_reader = ImageReader(BytesIO(img_data))
            c.drawImage(img_reader, 0, letter[1] - height, width=width, height=height)
            c.showPage()
        c.save()
        pdf_document.close()
        return output_pdf, True
    except Exception as e:
        err_msg = get_traceback(e, "Error converting PDF to images:")
        raise

def is_image_pdf(file_path):
    doc = fitz.open(file_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text().strip()
        if text:  # If text is found on any page
            return False
    return True

def normalize_pattern(s: str) -> str:
    """Remove spaces, parentheses, and other special characters."""
    original = s
    normalized = re.sub(r"[ ()-/]", "", s)
    return normalized
        
def get_processed_form_numbers(csv_tables, form_number_table_cols=['NAME', 'FORM NUMBER', 'EDITION DATE']):
    form_numbers = {
        "form_numbers": [],
        "normalized_form_numbers": []
    }
    for csv_table in csv_tables:
        df = pd.read_csv(StringIO(csv_table))
        if list(df.columns) == form_number_table_cols:
            for form_no, date_str in zip(df["FORM NUMBER"], df["EDITION DATE"]):
                date_splitter = [ch for ch in date_str if not ch.isdigit()]
                splitted_date = date_str.split(date_splitter[0])
                form_number = form_no + " (" + splitted_date[0] + "/" + splitted_date[2] + ")"
                form_numbers["form_numbers"].append(form_number)
                form_numbers["normalized_form_numbers"].append(normalize_pattern(form_number))
            return form_numbers

def get_merge_doc_obj(
        doc_id, 
        doc_name, 
        client_name,
        doc_type,
        uploaded_at=datetime.now(),
        file_path=None,
        is_active=True,
        policy_name=None,
        client_id=None,
        user_id=None
    ):
    merge_obj = {
        "useCase": "policy lookup",
        "uniqueId": None,
        "documentId": doc_id,
        "userId": user_id,
        "clientName": client_name,
        "clientID": client_id,
        "documentTag": doc_type,
        "documentSubTag": policy_name,
        "documentName": doc_name,
        "filePath": file_path,
        "adobeExtractPath": None,
        "uploadDate": uploaded_at,
        "processedDate": None,
        "processStatus": STATUS_IN_PROGRESS,
        "retryCount": 0,
        "failure": {
            "reason": None,
            "code": []
        },
        "isActive": is_active,
        "splitIndices": None,
        "splitDocumentList": []
    }
    return merge_obj

def get_document_obj(
        doc_id,
        doc_name, 
        client_name, 
        doc_type, 
        file_path=None, 
        uploaded_at=datetime.now(), 
        policy_name=None,
        client_id=None,
        user_id=None
    ):
    print("\n\nINSIDE DOC ID:", doc_id)
    doc_obj = {
        "useCase": "policy lookup",
        "userId": user_id,
        "uniqueId": None,
        "documentId": doc_id,
        "prompt": {
                    "qnAPrompId":None,
                    "globalComparisonPromptId":None,
                    "globalQnAPromptId":None,
                    "comparsionPromptId":None
                    },
        "documentName": doc_name,
        "clientName": client_name,
        "clientID": client_id,
        "documentTag": doc_type,
        "documentSubTag": policy_name,
        "filePath": file_path,
        "declarationType": None,
        "processStatus": STATUS_IN_PROGRESS, #
        "adobeExtractPath": None,
        "uploadDate": uploaded_at,
        "processedDate": None, #
        "policyStatus": None,
        "retryCount": 0,
        "failure": {
            "reason": None, #
            "code": []
        },
        "formNumber": {
            "formNumber": None,
            "normalizedFormNumber": None
        },
        "extractedFormNumbers": [],
        "isDelete": False,
        "isActive": True,
        "jsonId": None,
        "metaData": None,
        "childDocumentIds": []
    }
    return doc_obj

def get_metadata_obj():
    return {   
        "doc_id": None,
        "client_name": None,
        "policy_name": None,
        "doc_type": None,
        "file_path": None,
        "app_insight_logger": None,
        "transformed_doc_name": None,
        "paths": {
            "input_dir": None,
            "pdf_file_path": None,
        },
        "status": {
            "code": None,
            "message": None,
            "state": None
        },
        "result": {
            "chunks_obj": None,
            "declaration_metadata": {
                "version": 1
            },
            "form_numbers": None
        },
        "logs": {
            "error_log": [],
            "info_log": []
        },
        "original_message": None
    }