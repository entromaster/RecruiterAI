import fitz
import base64
from PIL import Image
from io import BytesIO
import json
import os
import re
import traceback
from datetime import datetime
from PyPDF2 import PdfReader

from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
from adobe.pdfservices.operation.io.stream_asset import StreamAsset
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.pdfjobs.jobs.ocr_pdf_job import OCRPDFJob
from adobe.pdfservices.operation.pdfjobs.result.ocr_pdf_result import OCRPDFResult

from agentic_rag.src import (
    AZURE_OPENAI_CHATGPT_DEPLOYMENT,
    ADOBE_CLIENT_ID,
    ADOBE_CLIENT_SECRET
)
from agentic_rag.src.utils import (
    openai_client,
    get_traceback
)

def extract_pypdf_text(pdf_path: BytesIO, doc_name: str, doc_id: str) -> list:
    """
    Loads a PDF from bytes, extracts text, and splits it into paragraphs.
    Each paragraph is annotated with its source document name and page number.
    """
    pdf_reader = PdfReader(pdf_path)
    total_pages = pdf_reader.pages
    paragraphs = []
    full_text = ""
    for i, page in enumerate(total_pages):
        page_text = page.extract_text()
        if not page_text:
            continue
        else:
            full_text += page_text + "\n"
        page_paragraphs = [p.strip() for p in re.split(r'\n\s*\n', page_text) if p.strip()]
        for p_index, p_text in enumerate(page_paragraphs):
            paragraphs.append({
                "document_id": doc_id,
                "document_name": doc_name,
                "page": i + 1,
                "paragraph_number": p_index + 1,
                "text": p_text
            })
    return paragraphs, total_pages, full_text

def pdf_pages_to_images(pdf_file_path, max_pages):
    try:
        pdf_document = fitz.open(pdf_file_path)
        num_pages = min(len(pdf_document), max_pages)
        base64_images = []
        for page_num in range(num_pages):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_byte_array = BytesIO()
            img.save(img_byte_array, format='JPEG')
            img_base64 = base64.b64encode(img_byte_array.getvalue()).decode('ascii')
            base64_images.append(f"data:image/jpeg;base64,{img_base64}")
        pdf_document.close()
        return base64_images, None
    except Exception as e:
        error_msg = get_traceback(e, "Failed to extract policy name images:")
        return None, error_msg

def openai_vision_response_generator(extraction_type, base64_images=None, target_text=None):
    try:
        user_content = []
        with open(f'./agentic_rag/config/prompts/{extraction_type}.prompt', 'r', encoding="utf-8") as file:
            prompt_type = file.read()
        system_message = {
            "role": "system",
            "content": [{
                "type": "text",
                "text": prompt_type
            }]
        }
        if extraction_type in ["policy_name", "cancellation", "reconciliation_doc_type", "tables", "declaration_metadata"] and base64_images:
            for base64_image in base64_images:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image
                    }
                })
            messages = [
                system_message,
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        elif extraction_type == "declaration_metadata" and target_text:
            user_content.append({
                    "type": "text",
                    "text": target_text
                })
            messages = [
                system_message,
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        temperature = 0.3  # Lower temperature for more deterministic results
        top_p = 0.95
        max_tokens = 1000  # Reduced as we expect shorter responses
        response = openai_client.chat.completions.create(
            model=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return response, None
    except Exception as e:
        try:
            response = openai_client.chat.completions.create(
                model=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return response, None
        except Exception as retry_error:
            return None, retry_error

def extract_policy_name_from_response(response_text):
    try:
        lines = response_text.strip().split('\n')
        policy_name = "Not Found"
        for line in lines:
            line = line.strip()
            if line.startswith("Policy Name:"):
                policy_name = line.replace("Policy Name:", "").strip()
                break
        return policy_name, None
    except Exception as e:
        error_msg = f"Failed to extract policy name from response: {str(e)}\n{traceback.format_exc()}"
        return None, error_msg

def vision_extractor(pdf_path=None, max_pages=2, extraction_type="policy_name", target_text=None):
    result = {
        "status": "success",
        "policy_name": None,
        "error_message": None,
        "error_stage": None
    }
    try:
        # if error:
        #     result["status"] = "failed"
        #     result["error_message"] = error
        #     result["error_stage"] = "image_extraction"
        #     return result
        # if error:
        #     result["status"] = "failed"
        #     result["error_message"] = error
        #     result["error_stage"] = "api_response"
        #     return result
        if "policy_name" == extraction_type:
            try:
                base64_images, error = pdf_pages_to_images(pdf_path, max_pages=max_pages)
                response, error = openai_vision_response_generator(extraction_type, base64_images=base64_images)
                response_text = response.choices[0].message.content
                policy_name, error = extract_policy_name_from_response(response_text)
                if error:
                    return None, error
                else:
                    return policy_name, None
            except Exception as error:
                err_msg = ''.join(traceback.format_exception(None, error, error.__traceback__))
                return None, err_msg
        elif "cancellation" in extraction_type or "rescission" in extraction_type:
            try:
                base64_images, error = pdf_pages_to_images(pdf_path, max_pages=max_pages)
                response, error = openai_vision_response_generator(extraction_type, base64_images=base64_images)
                response_text = response.choices[0].message.content
                cancel_data = json.loads(response_text)
                cancel_data["cancellation_date"] = datetime.strptime(cancel_data["cancellation_date"], '%Y-%m-%dT%I:%M%p')
                return cancel_data, None
            except Exception as error:
                err_msg = ''.join(traceback.format_exception(None, error, error.__traceback__))
                return None, err_msg
        elif "reconciliation_doc_type" in extraction_type:
            try:
                base64_images, error = pdf_pages_to_images(pdf_path, max_pages=max_pages)
                response, error = openai_vision_response_generator(extraction_type, base64_images=base64_images)
                response_text = response.choices[0].message.content
                reconciliation_type = response_text.split(":")[-1].strip()
                return reconciliation_type, None
            except Exception as error:
                err_msg = ''.join(traceback.format_exception(None, error, error.__traceback__))
                return None, err_msg  
        elif "declaration_metadata" in extraction_type:
            try:
                if target_text:
                    response, error = openai_vision_response_generator(extraction_type, target_text=target_text)
                elif pdf_path:
                    base64_images, error = pdf_pages_to_images(pdf_path, max_pages=max_pages)
                    response, error = openai_vision_response_generator(extraction_type, base64_images=base64_images)
                if response:
                    response_text = response.choices[0].message.content
                    declaration_metadata = json.loads(response_text)
                    return declaration_metadata, None
                else:
                    return None, get_traceback(error, "Failed while extracting metadata using openAI.")
            except Exception as error:
                err_msg = ''.join(traceback.format_exception(None, error, error.__traceback__))
                return None, err_msg 
        elif "tables" in extraction_type:
            try:
                base64_images, error = pdf_pages_to_images(pdf_path, max_pages=max_pages)
                response, error = openai_vision_response_generator(extraction_type, base64_images=base64_images)
                response_text = response.choices[0].message.content
                csv_tables = json.loads(response_text)
                return csv_tables, None
            except Exception as error:
                err_msg = ''.join(traceback.format_exception(None, error, error.__traceback__))
                return None, err_msg  
    except Exception as e:
        result["status"] = "failed"
        result["error_message"] = f"Unexpected error in policy name extraction: {str(e)}\n{traceback.format_exc()}"
        result["error_stage"] = "unexpected_error"
        return result
    
def scanned_2_text(pdf_path):
    try:
        pdf_name, ext = os.path.splitext(os.path.basename(pdf_path))
        output_file_path = os.path.join(os.path.dirname(pdf_path), pdf_name + "-scanned2text" + ext)
        file = open(pdf_path, 'rb')
        input_stream = file.read()
        file.close()
        credentials = ServicePrincipalCredentials(
            client_id=ADOBE_CLIENT_ID,
            client_secret=ADOBE_CLIENT_SECRET
        )
        pdf_services = PDFServices(credentials=credentials)
        input_asset = pdf_services.upload(input_stream=input_stream,
                                            mime_type=PDFServicesMediaType.PDF)
        ocr_pdf_job = OCRPDFJob(input_asset=input_asset)
        location = pdf_services.submit(ocr_pdf_job)
        pdf_services_response = pdf_services.get_job_result(location, OCRPDFResult)
        result_asset: CloudAsset = pdf_services_response.get_result().get_asset()
        stream_asset: StreamAsset = pdf_services.get_content(result_asset)
        with open(output_file_path, "wb") as file:
            file.write(stream_asset.get_input_stream())
        return output_file_path, None
    except (ServiceApiException, ServiceUsageException, SdkException) as e:
        err_msg = get_traceback(e, "Failed while conversion of the image type of the document to text pdf:")
        return None, err_msg