import os
import re
import shutil
import asyncio
from asyncio import Queue
from concurrent.futures import ProcessPoolExecutor
from azure.servicebus import ServiceBusMessage
from threading import Thread
from flask import Flask, request, jsonify
import json
import logging
import traceback
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS
from src.log_setup import setup_logger, azure_log
from src.extract_text_table_info_with_styling_from_pdf import run_extract_pdf
from src.documentNumber import extractDocumentNumber
from src.formNumber import extractFormNumber
from src.documentsplitter import DocumentSplitProcessor
from datetime import datetime, timezone
from src.policydetails import extractPolicyDetails
from src.policydetailsvision import visionpolicydetails
from src.tableextractionfromadoberesponse import process_ordering, normalize_pattern
from src import (
    LOGGER_NAME,
    ENV_NAME,
    SWAGGER_STATIC_PATH,
    SWAGGER_URL,
    API_URL,
    APP_NAME,
    WEBSITE_PORT,
    AZURE_OPENAI_CHATGPT_DEPLOYMENT,
    BASE_ENDORSEMENT_DIR,
    DECLARATION_DIR,
    MERGED_DIR,  # Added directory for merged documents
    FULLY_QUALIFIED_NAME,
    OCRFULLY_QUALIFIED_NAME,
    BASE_ENDO_QUEUE_NAME,
    QUEUE_PROCESS_BATCH_SIZE,
    MAX_WORKERS,
    MONGO_DOCUMENT_COLLECTION,
    MONGO_POLICIES_COLLECTION,
    MONGO_DECLARATION_COLLECTION,
    MONGO_CLIENT_COLLECTION,
    MONGO_REGEX_COLLECTION,
    MONGO_MERGED_COLLECTION,
    OCR_CONTAINER_NAME,
    BLOB_ACCOUNT_NAME,
    QUEUE_NAME,
    ADOBE_RESULT_FILENAME,
    ENV_NAMES,
    MSALREC,
    MSALNotify
)

from src.utils import (
    openai_client,
    create_declaration,
    create_document,
    create_policy,
    create_merge,
    asyto_mongo,
    update_status_mongo,
    get_document, get_document_async,
    upload_blob,
    download_blob,
    validate_doc_id,
    extractZipNew,
    map_form_numbers_to_declaration,
    match_document_numbers,
    fetch_carriers_with_base_policy,
    update_endorsement_document_id,
    to_mongo,
    log_ingestion_status_toDB,
    checkPDFencodingerrors,
    update_status_mongo_async,
    update_endorsement_document_id_async,
    merge_declarations,
    process_document_splitsforapi,
    convert_date_format,
    cleanup_error_entries,
    fetch_document_config,
    check_document_completion,
    check_merged_constituents,
    check_form_number_exists,
    cleanup_document_entries,
    pull_carrier_config,
    get_highest_declaration_version,
    remove_spaces,
    check_form_number_existskey,
    cleanup_document_entries_async,
    check_form_number_exists_async,
    map_form_numbers_to_declaration_async,
    get_highest_declaration_version_async,
    transform_frontend_blob_name,
    upload_blob_with_transform,
    get_processing_paths

    
)
import uuid
from azure.identity.aio import DefaultAzureCredential, ManagedIdentityCredential
from azure.servicebus.aio import ServiceBusClient
from flask import Flask, make_response, redirect, request
from flask_talisman import Talisman

# Global variables
declaration_dependencies = {}  # key: declaration_id, value: {'required_docs': set(), 'received_docs': set(), 'message': msg}
merged_documents_tracking = {}
received_documents = set()  # Set of document IDs that have been processed
stop_event = asyncio.Event()
receive_messages_started = False
main_pid = os.getpid()
background_loop = None

# Queues
base_endo_queue = Queue()
declaration_queue = Queue()
#ready_declaration_queue = Queue()
merged_queue = Queue()  # Queue for merged documents
extraction_retry_queue = Queue()
adobe_retry_queue = Queue()
manual_split_queue = Queue()
declaration_temp_queue = Queue()
MSALeaseQueue = Queue()

waiting_endorsements = {}

# Lock related variables
declaration_lock = asyncio.Event()  # Initially clear (locked state)
unlock_timer = None
timer_cancelled = False

msa_priority_lock = asyncio.Event()  # Similar to declaration_lock
msa_unlock_timer = None
msa_timer_cancelled = False
temp_base_endo_storage = []  # Temporary storage for base/endorsement items
temp_merged_storage = []  # Temporary storage for merged items
temp_adobe_retry_storage = []  # Temporary storage for adobe retry items  
temp_extraction_retry_storage = []  # Temporary storage for extraction retry items
temp_manual_split_storage = []  # Temporary storage for manual split items
temp_declaration_storage = []  # Temporary storage for declaration items

app_logger = setup_logger(LOGGER_NAME)
mopenaiclientres = openai_client(ENV_NAME)
client, token_provider = mopenaiclientres.unwrap()

app = Flask(__name__, static_url_path=SWAGGER_STATIC_PATH)
Talisman(app)
# Set global cookie settings
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Strict'
CORS(app)
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': APP_NAME},
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

app_insight_logger = azure_log(app)
properties = {'custom_dimensions': {'ApplicationName': "Document Ingestion Application"}}


@app.route('/api/ping', methods=['GET'])
def ping():
    return jsonify({'success': True, "statusCode": 200}), 200

@app.route('/api/retryadobe', methods=['POST'])
def retry_adobe_extract():
    """
    Retry Adobe Extract API for a specific document.
    ---
    tags:
      - Document Processing
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - doc_id
          properties:
            doc_id:
              type: string
              description: The ID of the document to retry Adobe Extract processing
    responses:
      200:
        description: Retry process initiated successfully
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: true
            message:
              type: string
              example: "Adobe retry initiated for document"
            requestId:
              type: string
              example: "f47ac10b-58cc-4372-a567-0e02b2c3d479"
            statusCode:
              type: integer
              example: 200
      400:
        description: Invalid request or processing error
    """
    try:
        request_data = request.get_json()
        if not request_data or 'doc_id' not in request_data:
            return jsonify({
                "success": False,
                "message": "Missing doc_id in request body",
                "statusCode": 400
            }), 400

        doc_id = request_data['doc_id']
        request_id = str(uuid.uuid4())
        
        if background_loop:
            message_tuple = ("adobe_retry", (doc_id, request_id))
            future = asyncio.run_coroutine_threadsafe(
                adobe_retry_queue.put(message_tuple), 
                background_loop
            )
            future.result()
            
            return jsonify({
                "success": True,
                "message": f"Adobe retry initiated for document {doc_id}",
                "requestId": request_id,
                "statusCode": 200
            }), 200
        else:
            return jsonify({
                "success": False,
                "message": "Background processing not available",
                "statusCode": 500
            }), 500
            
    except Exception as e:
        err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
        app_insight_logger.error(f"Error in retry_adobe_extract: {err_msg}", extra=properties)
        return jsonify({
            "success": False,
            "message": f"Error processing request: {str(e)}",
            "statusCode": 400
        }), 400

@app.route('/api/retryextraction', methods=['POST'])
def retry_extraction():
    """
    Retry document number and form number extraction with new carrier configuration.
    ---
    tags:
      - Document Processing
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - doc_id
            - config
          properties:
            doc_id:
              type: string 
              description: The ID of the document to retry extraction
            config:
              type: object
              description: Carrier configuration for extraction
              required:
                - CarrierName
                - Mode
                - FormNumberPattern
              properties:
                CarrierName:
                  type: string
                  description: Name of the carrier
                Mode:
                  type: string
                  enum: ["Table", "Text"]
                  description: Processing mode for the extraction
                FormNumberPattern:
                  type: string
                  description: Regex pattern for form number extraction
                TableConfig:
                  type: object
                  description: Required when Mode is "Table"
                  required:
                    - TablesFolder
                    - FileExtensions
                    - Encoding
                    - ColumnRequirements
                    - BasePolicyCondition
                    - DeclarationCondition
    """
    try:
        request_data = request.get_json()
        
        # Validate required fields
        if not request_data or 'doc_id' not in request_data or 'config' not in request_data:
            return jsonify({
                "success": False,
                "message": "Missing required fields in request body",
                "statusCode": 400
            }), 400
            
        # Validate config object has required fields
        config = request_data['config']
        required_config_fields = ['CarrierName', 'Mode', 'FormNumberPattern']
        missing_fields = [field for field in required_config_fields if field not in config]
        if missing_fields:
            return jsonify({
                "success": False,
                "message": f"Config missing required fields: {', '.join(missing_fields)}",
                "statusCode": 400
            }), 400

        # Validate Mode
        if config['Mode'] not in ['Table', 'Text']:
            return jsonify({
                "success": False,
                "message": "Invalid Mode value. Must be either 'Table' or 'Text'",
                "statusCode": 400
            }), 400

        # Mode-specific validation - only validate Table mode config
        if config['Mode'] == 'Table':
            if 'TableConfig' not in config:
                return jsonify({
                    "success": False,
                    "message": "TableConfig is required when Mode is 'Table'",
                    "statusCode": 400
                }), 400

            table_config = config['TableConfig']
            required_table_fields = [
                'TablesFolder', 
                'FileExtensions', 
                'Encoding', 
                'ColumnRequirements',
                'BasePolicyCondition',
                'DeclarationCondition'
            ]
            missing_table_fields = [field for field in required_table_fields if field not in table_config]
            if missing_table_fields:
                return jsonify({
                    "success": False,
                    "message": f"TableConfig missing required fields: {', '.join(missing_table_fields)}",
                    "statusCode": 400
                }), 400

            if not isinstance(table_config['FileExtensions'], list):
                return jsonify({
                    "success": False,
                    "message": "FileExtensions must be an array",
                    "statusCode": 400
                }), 400

        doc_id = request_data['doc_id']
        request_id = str(uuid.uuid4())
        
        if background_loop:
            message_tuple = ("extraction_retry", (doc_id, config, request_id))
            future = asyncio.run_coroutine_threadsafe(
                extraction_retry_queue.put(message_tuple), 
                background_loop
            )
            future.result()
            
            return jsonify({
                "success": True,
                "message": f"Extraction retry initiated for document {doc_id}",
                "requestId": request_id,
                "statusCode": 200
            }), 200
        else:
            return jsonify({
                "success": False,
                "message": "Background processing not available",
                "statusCode": 500
            }), 500
            
    except Exception as e:
        err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
        app_insight_logger.error(f"Error in retry_extraction: {err_msg}", extra=properties)
        return jsonify({
            "success": False,
            "message": f"Error processing request: {str(e)}",
            "statusCode": 400
        }), 400
    
@app.route('/api/reprocess', methods=['POST'])
def reprocess():
    """
    Reprocess multiple documents using their configurations from the database.
    ---
    tags:
      - Document Processing
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - documentIds
          properties:
            documentIds:
              type: array
              items:
                type: string
              description: List of document IDs to reprocess
    """
    try:
        request_data = request.get_json()

        app_insight_logger.info("Re-process api payload : ", str(request_data), extra=properties)
        
        # Validate request body exists and has documentIds
        if not request_data or 'documentIds' not in request_data:
            return jsonify({
                "success": False,
                "message": "Missing required field: documentIds",
                "statusCode": 400
            }), 400
            
        # Validate documentIds is a non-empty array
        if not isinstance(request_data['documentIds'], list) or not request_data['documentIds']:
            return jsonify({
                "success": False,
                "message": "documentIds must be a non-empty array",
                "statusCode": 400
            }), 400

        if not background_loop:
            return jsonify({
                "success": False,
                "message": "Background processing not available",
                "statusCode": 500
            }), 500

        # Get collection names from environment variables
        document_collection = MONGO_DOCUMENT_COLLECTION
        client_collection = MONGO_CLIENT_COLLECTION
        regex_collection = MONGO_REGEX_COLLECTION

        # Process each document ID
        request_ids = {}
        failed_docs = {}
        
        for doc_id in request_data['documentIds']:
            try:
                # Fetch config for this document
                config = fetch_document_config(
                    doc_id,
                    document_collection,
                    client_collection,
                    regex_collection
                )
                
                # Generate request ID and queue the task
                request_id = str(uuid.uuid4())
                request_ids[doc_id] = request_id
                
                message_tuple = ("extraction_retry", (doc_id, config, request_id))
                future = asyncio.run_coroutine_threadsafe(
                    extraction_retry_queue.put(message_tuple), 
                    background_loop
                )
                future.result()
                
            except ValueError as ve:
                # Handle validation errors for individual documents
                failed_docs[doc_id] = str(ve)
            except Exception as e:
                # Handle other errors for individual documents
                failed_docs[doc_id] = f"Unexpected error: {str(e)}"
        
        # Prepare response based on results
        if not request_ids:
            # If all documents failed
            return jsonify({
                "success": False,
                "message": "Failed to process all documents",
                "failedDocuments": failed_docs,
                "statusCode": 400
            }), 400
            
        response = {
            "success": True,
            "message": f"Reprocess initiated for {len(request_ids)} documents",
            "requestIds": request_ids,
            "statusCode": 200
        }
        
        # Include failed documents in response if any
        if failed_docs:
            response["failedDocuments"] = failed_docs
            response["message"] += f" ({len(failed_docs)} documents failed)"
            
        return jsonify(response), 200
            
    except Exception as e:
        err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
        app_insight_logger.error(f"Error in reprocess: {err_msg}", extra=properties)
        return jsonify({
            "success": False,
            "message": f"Error processing request: {str(e)}",
            "statusCode": 400
        }), 400
    
@app.route('/api/manualsplit', methods=['POST'])
def manual_split():
    """
    Manually split a merged document according to provided specifications.
    ---
    tags:
      - Document Processing
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - doc_id
            - splitList
          properties:
            doc_id:
              type: string
              description: The ID of the document to split
            splitList:
              type: array
              items:
                type: object
                required:
                  - formNumber
                  - pageSpan
                  - documentType
                properties:
                  formNumber:
                    type: string
                    description: Form number for the split
                  pageSpan:
                    type: array
                    items:
                      type: integer
                    minItems: 2
                    maxItems: 2
                    description: Start and end page numbers [start, end]
                  documentType:
                    type: string
                    enum: [Base, Declaration, Endorsement]
                    description: Type of document for this split
    responses:
      200:
        description: Split process initiated successfully
      400:
        description: Invalid request
    """
    try:
        request_data = request.get_json()
        if not request_data or 'doc_id' not in request_data or 'splitList' not in request_data:
            return jsonify({
                "success": False,
                "message": "Missing required fields in request body",
                "statusCode": 400
            }), 400

        # Validate splitList format
        for split in request_data["splitList"]:
            if not all(key in split for key in ["formNumber", "pageSpan", "documentType"]):
                return jsonify({
                    "success": False,
                    "message": "Invalid split format. Each split must contain formNumber, pageSpan, and documentType",
                    "statusCode": 400
                }), 400
            
            if split["documentType"] not in ["Base", "Declaration", "Endorsement"]:
                return jsonify({
                    "success": False,
                    "message": f"Invalid documentType: {split['documentType']}",
                    "statusCode": 400
                }), 400
            
            if not isinstance(split["pageSpan"], (list, tuple)) or len(split["pageSpan"]) != 2:
                return jsonify({
                    "success": False,
                    "message": "pageSpan must be an array of two integers",
                    "statusCode": 400
                }), 400

            # Additional validation for pageSpan values
            if not all(isinstance(x, int) for x in split["pageSpan"]) or split["pageSpan"][0] > split["pageSpan"][1]:
                return jsonify({
                    "success": False,
                    "message": "pageSpan must contain valid integer ranges where start page <= end page",
                    "statusCode": 400
                }), 400

        doc_id = request_data['doc_id']
        request_id = str(uuid.uuid4())
        
        if background_loop:
            message_tuple = ("manual_split", (doc_id, request_data, request_id))
            future = asyncio.run_coroutine_threadsafe(
                manual_split_queue.put(message_tuple), 
                background_loop
            )
            future.result()
            
            return jsonify({
                "success": True,
                "message": f"Manual split initiated for document {doc_id}",
                "requestId": request_id,
                "statusCode": 200
            }), 200
        else:
            return jsonify({
                "success": False,
                "message": "Background processing not available",
                "statusCode": 500
            }), 500
            
    except Exception as e:
        err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
        app_insight_logger.error(f"Error in manual_split: {err_msg}", extra=properties)
        return jsonify({
            "success": False,
            "message": f"Error processing request: {str(e)}",
            "statusCode": 400
        }), 400
    

class Timer:
    def __init__(self, timeout, callback):
        self._timeout = timeout
        self._callback = callback
        self._task = asyncio.ensure_future(self._job())
        self._cancelled = False

    async def _job(self):
        try:
            await asyncio.sleep(self._timeout)
            await self._callback()
        except asyncio.CancelledError:
            self._cancelled = True
            app_insight_logger.info("Timer cancelled: Declaration lock will remain active", extra=properties)
            raise

    def cancel(self):
        self._task.cancel()

    @property
    def was_cancelled(self):
        return self._cancelled

"""
Responsible for reading the message from the azure service bus queue
Adds the basic placeholder entry into the database for the document collection, declaration collection and the polcies collection
And calls the functio that re-directs them to the queues
"""
async def generate_new_doc_id_and_policy(blobname, folder_name, carrier_name, id_valid=False, policy_name=None):
    """
    Generates new document ID and creates necessary policy/declaration objects
    Returns (doc_id, filename_with_doc_id, doc_type)
    """
    try:
        doc_id = str(uuid.uuid4())
        doc_type = None
        policy_no = str(uuid.uuid4())
        file_name, ext = os.path.splitext(blobname)
        filename_with_doc_id = file_name + "_" + doc_id + ext

        if "base" in folder_name.lower():
            doc_type = "BasePolicy"
            policy_result = create_policy(
                policy_number=policy_no, 
                base_doc_id=doc_id, 
                carriername=carrier_name
            )
            policy_obj = policy_result.unwrap()
            
            # Add PolicyName if it's a frontend upload
            if policy_name:
                policy_obj['PolicyName'] = policy_name
                
            await asyto_mongo([policy_obj], MONGO_POLICIES_COLLECTION)

        elif "endo" in folder_name.lower():
            doc_type = "Endorsement"

        elif "declaration" in folder_name.lower():
            doc_type = "Declaration"
            dec_result = create_declaration(
                dec_doc_id=doc_id,
                carriername=carrier_name,
                sample_declaration=False
            )
            dec_obj = dec_result.unwrap()
            await asyto_mongo([dec_obj], MONGO_DECLARATION_COLLECTION)

        elif "merged" in folder_name.lower():
            doc_type = "Merged Document"

        return doc_id, filename_with_doc_id, doc_type

    except Exception as e:
        app_insight_logger.error(f"Error in generate_new_doc_id_and_policy: {str(e)}", extra=properties)
        raise


async def receive_messages():
    app_insight_logger.info("Inside receive_messages function", extra=properties)
    
    try:
        # Initialize Azure credentials
        try:
            if ENV_NAME in ENV_NAMES:
                credential = ManagedIdentityCredential()
            else:
                credential = DefaultAzureCredential()
        except Exception as e:
            err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
            app_insight_logger.error(f"Error while creating Azure Credentials: {err_msg}", extra=properties)
            raise

        # Create ServiceBusClient
        try:
            servicebus_client = ServiceBusClient(
                fully_qualified_namespace=FULLY_QUALIFIED_NAME,
                credential=credential,
                logging_enable=True,
            )
        except Exception as e:
            err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
            app_insight_logger.error(f"Error in creating service bus client: {err_msg}", extra=properties)
            raise

        async with credential, servicebus_client:
            receiver = servicebus_client.get_queue_receiver(
                queue_name=BASE_ENDO_QUEUE_NAME, 
                max_wait_time=5
            )
            
            async with receiver:
                while not stop_event.is_set():
                    try:
                        received_msgs = await receiver.receive_messages(
                            max_message_count=QUEUE_PROCESS_BATCH_SIZE, 
                            max_wait_time=5
                        )
                        
                        for msg in received_msgs:
                            try:
                                global unlock_timer, timer_cancelled                              

                                # Clear lock and log
                                if declaration_lock.is_set():
                                    declaration_lock.clear()
                                    app_insight_logger.info("LOCK ACQUIRED: Declaration processing locked due to new message arrival", extra=properties)
                                
                                # Cancel any existing timer
                                if unlock_timer:
                                    unlock_timer.cancel()
                                    timer_cancelled = True
                                    unlock_timer = None
                                    app_insight_logger.info("Timer cancelled due to new message arrival", extra=properties)

                                app_insight_logger.info(f"Received message:\n {msg}", extra=properties)    
                                
                                for body in msg.body:
                                    try:
                                        j_body = json.loads(body)
                                        document_path = j_body["data"]["url"]
                                        blobname = document_path.split("/")[-1]
                                        carrier_name = document_path.split("/")[-3]
                                        doc_id = ""
                                        filename_with_doc_id = None
                                        user_id = None
                                        policy_name = ""
                                        splitdictinfo = {'frontendflag': 'N'}
                                        splitdictinfo["FormNumberExist"] = "Y"

                                        # Check if this is a frontend upload and get transformed name and user_id
                                        transformed_name = blobname
                                        app_insight_logger.info(f"Original Blobname {transformed_name}", extra=properties)
                                        if "==" in blobname:
                                            # Split by == to get all parts
                                            filename_parts = blobname.split("==")
                                            if len(filename_parts) == 4:  # Original name, UserID, PolicyName.pdf
                                                original_name = filename_parts[0]
                                                doc_id = filename_parts[1]
                                                user_id = filename_parts[2]
                                                app_insight_logger.info(f"User ID:  {user_id}", extra=properties)
                                                policy_name = filename_parts[3].rsplit('.', 1)[0]  # Remove .pdf extension
                                                transformed_name = transform_frontend_blob_name(blobname)
                                                splitdictinfo['frontendflag'] = 'Y'
                                                app_insight_logger.info(
                                                    f"Frontend upload detected. UserID: {user_id}, PolicyName: {policy_name}, Transformed name: {transformed_name}", 
                                                    extra=properties
                                                )

                                        folder_name = document_path.split("/")[-2]
                                        carrier_name = document_path.split("/")[-3]
                                        container_name = document_path.split("/")[3] + "/document-ingestion"
                                        request_id = str(uuid.uuid4())
                                        # Generate new doc_id and create necessary objects
                                        document_exist = await get_document_async({"ID": doc_id}, MONGO_DOCUMENT_COLLECTION)
                                        policy_exist = await get_document_async({"PolicyName": policy_name, "CarrierName": carrier_name}, MONGO_POLICIES_COLLECTION)
                                        # document_exist = document_exist.unwrap()
                                        
                                        app_insight_logger.info(f"DOC ID {doc_id}\nDOCUMENT EXIST:\n{document_exist}")
                                        if document_exist: # Document exist for no form number
                                            splitdictinfo["FormNumberExist"] = "N"

                                            file_name, ext = os.path.splitext(blobname)
                                            app_insight_logger.info(f"CHANGED THE VALUE OF splitdictinfo['FormNumberExist']: {splitdictinfo['FormNumberExist']}")
                                            filename_with_doc_id = ""
                                            if "base" in folder_name.lower():
                                                doc_type = "BasePolicy"
                                            elif "endo" in folder_name.lower():
                                                doc_type = "Endorsement"
                                            elif "declaration" in folder_name.lower():
                                                doc_type = "Declaration"
                                            elif "merged" in folder_name.lower():
                                                doc_type = "Merged Document"

                                            await update_status_mongo_async(
                                                {"ID": doc_id},
                                                {
                                                    "$set": {
                                                        "DocumentName": transformed_name
                                                    }
                                                },
                                                MONGO_DOCUMENT_COLLECTION
                                            )
                                        elif policy_exist: # for adding more endorsement policy exist in policies collection
                                            # splitdictinfo["FormNumberExist"] = "Y"
                                            file_name, ext = os.path.splitext(blobname)
                                            app_insight_logger.info(f"CHANGED THE VALUE OF splitdictinfo['FormNumberExist']: {splitdictinfo['FormNumberExist']}")
                                            filename_with_doc_id = ""
                                            if "base" in folder_name.lower():
                                                doc_type = "BasePolicy"
                                            elif "endo" in folder_name.lower():
                                                doc_type = "Endorsement"
                                            elif "declaration" in folder_name.lower():
                                                doc_type = "Declaration"
                                                doc_id = str(uuid.uuid4())
                                                dec_result = create_declaration(
                                                    dec_doc_id=doc_id,
                                                    carriername=carrier_name,
                                                    sample_declaration=False
                                                )
                                                dec_obj = dec_result.unwrap()
                                                dec_obj["PolicyName"] = policy_name
                                                await asyto_mongo([dec_obj], MONGO_DECLARATION_COLLECTION)
                                            elif "merged" in folder_name.lower():
                                                doc_type = "Merged Document"

                                            await update_status_mongo_async(
                                                {"ID": doc_id},
                                                {
                                                    "$set": {
                                                        "DocumentName": transformed_name
                                                    }
                                                },
                                                MONGO_DOCUMENT_COLLECTION
                                            )
                                            if doc_type != "Merged Document":
                                                doc_obj = create_document(
                                                    doc_id=doc_id,
                                                    doc_name=transformed_name,  # Use transformed name in DB
                                                    carrier_name=carrier_name,
                                                    doc_type=doc_type,
                                                ).unwrap()
                                                
                                                # Add UserID if present
                                                if user_id:
                                                    doc_obj['UserID'] = user_id
                                                    #doc_obj['original_filename'] = blobname  # Store original name for reference
                                                
                                                mongo_result = await asyto_mongo(
                                                    [doc_obj], 
                                                    MONGO_DOCUMENT_COLLECTION
                                                )
                                                mongo_result.unwrap()
                                            
                                        else:
                                            try:
                                                gen_result = await generate_new_doc_id_and_policy(
                                                    blobname, 
                                                    folder_name, 
                                                    carrier_name,
                                                    id_valid=False,  # Always generate new ID for frontend uploads
                                                    policy_name=policy_name if "==" in blobname else None
                                                )
                                                doc_id, filename_with_doc_id, doc_type = gen_result

                                                if doc_type != "Merged Document":
                                                    doc_obj = create_document(
                                                        doc_id=doc_id,
                                                        doc_name=transformed_name,  # Use transformed name in DB
                                                        carrier_name=carrier_name,
                                                        doc_type=doc_type,
                                                    ).unwrap()
                                                    
                                                    # Add UserID if present
                                                    if user_id:
                                                        doc_obj['UserID'] = user_id
                                                        #doc_obj['original_filename'] = blobname  # Store original name for reference
                                                    
                                                    mongo_result = await asyto_mongo(
                                                        [doc_obj], 
                                                        MONGO_DOCUMENT_COLLECTION
                                                    )
                                                    mongo_result.unwrap()
                                                else:
                                                    merge_obj = create_merge(
                                                            doc_id=doc_id,
                                                            doc_name=transformed_name,
                                                            file_path=document_path,
                                                            uploaded_at=datetime.now(),
                                                            carriername=carrier_name
                                                        ).unwrap()
                                                    
                                                    # mongo_result = await asyto_mongo(
                                                    #     [merge_obj],
                                                    #     MONGO_MERGED_COLLECTION
                                                    # ).unwrap()
                                            except Exception as e:
                                                err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
                                                app_insight_logger.error(f"Error generating new doc id and policy: {err_msg}", extra=properties)
                                                raise

                                        app_insight_logger.info(
                                            f"\n\n\nDOCUMENT ID: {doc_id} (Frontend Upload: {splitdictinfo['frontendflag']})\n\n\n",
                                            extra=properties
                                        )
                                        
                                        message_data = (
                                            blobname,  # Keep original blobname for downstream processing
                                            doc_id,
                                            filename_with_doc_id,
                                            container_name,
                                            carrier_name,
                                            folder_name,
                                            app_insight_logger,
                                            request_id,
                                            splitdictinfo
                                        )

                                        # Process message
                                        try:
                                            await process_message(message_data)
                                        except Exception as e:
                                            err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
                                            app_insight_logger.error(f"Error processing message: {err_msg}", extra=properties)
                                            raise

                                        await asyncio.sleep(0.1)
                                        
                                    except json.JSONDecodeError as e:
                                        app_insight_logger.error(f"Error decoding message body: {str(e)}", extra=properties)
                                        continue
                                    except Exception as e:
                                        err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
                                        app_insight_logger.error(f"Error processing message body: {err_msg}", extra=properties)
                                        continue
                                        
                                await receiver.complete_message(msg)
                                
                            except Exception as e:
                                err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
                                app_insight_logger.error(f"Error handling received message: {err_msg}", extra=properties)
                                continue
                                
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
                        app_insight_logger.error(f"Error in receive_messages loop: {err_msg}", extra=properties)
                        await asyncio.sleep(1)
                        
    except asyncio.CancelledError:
        pass
    except Exception as e:
        err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
        app_insight_logger.error(f"Error in receive_messages: {err_msg}", extra=properties)
    finally:
        app_insight_logger.info("Closing received messages!", extra=properties)
        if credential:
            await credential.close()

async def async_unlock_callback():
    """Callback for the unlock timer. Verifies conditions before unlocking."""
    global timer_cancelled
    
    # Double check conditions before unlocking
    if base_endo_queue.empty():
        declaration_lock.set()
        timer_cancelled = False  # Timer completed naturally
        app_insight_logger.info("LOCK RELEASED: All conditions met, declaration processing now allowed", extra=properties)
    else:
        declaration_lock.clear()
        timer_cancelled = True
        app_insight_logger.info("LOCK MAINTAINED: Queue not empty, declaration processing remains locked", extra=properties)

async def async_msa_unlock_callback():
    """Callback for the MSA unlock timer. Releases MSA priority after timeout."""
    global msa_timer_cancelled, msa_unlock_timer
    
    # Double check conditions before unlocking
    if MSALeaseQueue.empty():
        msa_priority_lock.clear()
        msa_timer_cancelled = False  # Timer completed naturally
        msa_unlock_timer = None  # Reset timer reference
        app_insight_logger.info("MSA PRIORITY RELEASED: MSA queue empty for 3 seconds, resuming normal processing", extra=properties)
    else:
        # Queue is not empty, maintain priority and reset timer
        msa_priority_lock.set()
        msa_timer_cancelled = True
        msa_unlock_timer = None  # Reset timer reference
        app_insight_logger.info("MSA PRIORITY MAINTAINED: MSA queue not empty, continuing priority processing", extra=properties)                    

"""
Redirects the message into the proper queue
These messages can then be picked by other co-routines doing the processing
Those co-routines check if the queue is non empty
and if they are indeed non empty, they will form batches and use process pool executor to process those batches
therby offloading the cpu bound work
"""

async def process_message(message_data):
    blobname, doc_id, filename_with_doc_id, container_name, carrier_name, folder_name, appinslogger, request_id, split_indices_dict = message_data
    app_insight_logger.info(f"{folder_name}, {folder_name.lower()}", extra=properties)
    
    # Handle MSA Lease documents
    if "msalease" in folder_name.lower():
        await MSALeaseQueue.put(("msa_lease_doc", message_data))
        app_insight_logger.info(f"Added MSA Lease document to MSALeaseQueue: {blobname}", extra=properties)
    # Handle existing document types
    elif ("endorsement" in folder_name.lower()) or ("base" in folder_name.lower()):
        await base_endo_queue.put(("base_endo_doc", message_data))
    elif "declaration" in folder_name.lower():
        await declaration_queue.put(("dec_doc", message_data))
    elif "merged" in folder_name.lower():
        await merged_queue.put(("merged_doc", message_data))
    else:
        app_insight_logger.info("Unknown Document", extra=properties)


async def cleanup_directory():
    """
    Cleanup processed documents from all directories
    """
    try:
        #app_insight_logger.info("Starting directory cleanup check")
        
        for dir_path in [BASE_ENDORSEMENT_DIR, DECLARATION_DIR, MERGED_DIR]:
            try:
                # Get all document directories
                doc_dirs = os.listdir(dir_path)
                cleaned_dirs = []
                
                for doc_id in doc_dirs:
                    try:
                        doc_dir_path = os.path.join(dir_path, doc_id)
                        
                        # Skip if not a directory
                        if not os.path.isdir(doc_dir_path):
                            continue
                            
                        # For merged directory, check all constituent documents
                        if dir_path == MERGED_DIR:
                            is_complete = await check_merged_constituents(doc_id, MONGO_DOCUMENT_COLLECTION)
                        else:
                            is_complete = await check_document_completion(doc_id, MONGO_DOCUMENT_COLLECTION)
                        
                        if is_complete:
                            try:
                                shutil.rmtree(doc_dir_path)
                                cleaned_dirs.append(doc_id)
                                app_insight_logger.info(f"Cleaned up directory for document {doc_id} from {os.path.basename(dir_path)}", extra=properties)
                            except Exception as e:
                                app_insight_logger.error(f"Error removing directory for document {doc_id}: {str(e)}", extra=properties)
                                
                    except Exception as e:
                        app_insight_logger.error(f"Error processing directory {doc_id}: {str(e)}", extra=properties)
                
                if cleaned_dirs:
                    app_insight_logger.info(f"Cleaned {len(cleaned_dirs)} directories from {os.path.basename(dir_path)}: {cleaned_dirs}", extra=properties)
                    
            except Exception as e:
                app_insight_logger.error(f"Error processing {dir_path}: {str(e)}", extra=properties)
                
        await asyncio.sleep(0.5)
        
    except Exception as e:
        app_insight_logger.error(f"Error in cleanup_directory: {str(e)}", extra=properties)

async def receiveMSALeaseMessages():
    """
    Coroutine to receive messages from MSA Lease queue.
    Runs continuously to receive messages and add them to internal processing queue.
    """
    app_insight_logger.info("Inside receiveMSALeaseMessages function", extra=properties)
    
    try:
        # Initialize Azure credentials
        try:
            if ENV_NAME in ENV_NAMES:
                credential = ManagedIdentityCredential()
            else:
                credential = DefaultAzureCredential()
        except Exception as e:
            err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
            app_insight_logger.error(f"Error while creating Azure Credentials for MSA Lease queue: {err_msg}", extra=properties)
            raise

        # Create ServiceBusClient
        try:
            servicebus_client = ServiceBusClient(
                fully_qualified_namespace=FULLY_QUALIFIED_NAME,
                credential=credential,
                logging_enable=True,
            )
        except Exception as e:
            err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
            app_insight_logger.error(f"Error in creating service bus client for MSA Lease queue: {err_msg}", extra=properties)
            raise

        async with credential, servicebus_client:
            # Use a placeholder queue name - replace with actual queue name
            receiver = servicebus_client.get_queue_receiver(
                queue_name=MSALREC,  # Replace with actual queue name constant
                max_wait_time=5
            )
            
            async with receiver:
                while not stop_event.is_set():
                    try:
                        received_msgs = await receiver.receive_messages(
                            max_message_count=QUEUE_PROCESS_BATCH_SIZE, 
                            max_wait_time=5
                        )

                        if received_msgs:
                            global msa_unlock_timer, msa_timer_cancelled
                            
                            # Set the MSA priority lock
                            msa_priority_lock.set()
                            app_insight_logger.info("MSA PRIORITY ACTIVATED: Prioritizing MSA lease processing", extra=properties)
                            
                            # Cancel any existing MSA unlock timer
                            if msa_unlock_timer:
                                msa_unlock_timer.cancel()
                                msa_timer_cancelled = True
                                msa_unlock_timer = None
                                app_insight_logger.info("MSA timer cancelled due to new MSA message arrival", extra=properties)
                        
                        for msg in received_msgs:
                            try:
                                app_insight_logger.info(f"Received MSA Lease message:\n {msg}", extra=properties)    
                                
                                for body in msg.body:
                                    try:
                                        j_body = json.loads(body)
                                        document_path = j_body["data"]["url"]
                                        
                                        # Extract blobname and parse document ID
                                        blobname = document_path.split("/")[-1]
                                        doc_id = blobname.split("_")[-1].split(".")[0]  # Get ID between last underscore and .pdf
                                        
                                        # Log the Document ID
                                        app_insight_logger.info(f"Processing Document ID: {doc_id}", extra=properties)
                                        
                                        container_name = document_path.split("/")[3]
                                        folder_name = "MsaLease"  # This is fixed based on the URL format
                                        request_id = str(uuid.uuid4())
                                        
                                        # Create message tuple similar to existing format
                                        message_data = (
                                            blobname,
                                            doc_id,  # Now we have the correct doc_id
                                            None,  # No filename_with_doc_id needed
                                            container_name,
                                            None,  # No carrier_name needed
                                            folder_name,
                                            app_insight_logger,
                                            request_id,
                                            {'frontendflag': 'N'}  # Default flags
                                        )

                                        # Process message
                                        try:
                                            await process_message(message_data)
                                        except Exception as e:
                                            err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
                                            app_insight_logger.error(f"Error processing MSA Lease message: {err_msg}", extra=properties)
                                            raise

                                        await asyncio.sleep(0.1)
                                        
                                    except json.JSONDecodeError as e:
                                        app_insight_logger.error(f"Error decoding MSA Lease message body: {str(e)}", extra=properties)
                                        continue
                                    except Exception as e:
                                        err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
                                        app_insight_logger.error(f"Error processing MSA Lease message body: {err_msg}", extra=properties)
                                        continue
                                        
                                await receiver.complete_message(msg)
                                
                            except Exception as e:
                                err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
                                app_insight_logger.error(f"Error handling received MSA Lease message: {err_msg}", extra=properties)
                                continue
                                
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
                        app_insight_logger.error(f"Error in receiveMSALeaseMessages loop: {err_msg}", extra=properties)
                        await asyncio.sleep(1)
                        
    except asyncio.CancelledError:
        pass
    except Exception as e:
        err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
        app_insight_logger.error(f"Error in receiveMSALeaseMessages: {err_msg}", extra=properties)
    finally:
        app_insight_logger.info("Closing MSA Lease message receiver!", extra=properties)
        if credential:
            await credential.close()

def process_msa_lease_item(msg):
    """
    Process a MSA Lease document message with comprehensive logging.
    Returns: (doc_id, status, metadata, original_message, logs_data)
    """
    # Initialize logging collections
    info_logs = []
    error_logs = []
    current_step = "initialization"
    
    def add_info_log(step, message, extra_metadata=None):
        """Helper function to add info logs"""
        log_metadata = {
            "document_name": blobname if 'blobname' in locals() else None,
            "document_type": "Lease",
            "processing_path": metadata["paths"].get("adobe_extract_path") if metadata.get("paths") else None
        }
        if extra_metadata:
            log_metadata.update(extra_metadata)
            
        info_logs.append({
            "document_id": metadata.get("doc_id"),
            "processor_type": "msa_lease",
            "request_id": request_id if 'request_id' in locals() else None,
            "timestamp": datetime.now(),
            "step": step,
            "message": message,
            "metadata": log_metadata
        })

    def add_error_log(step, error, error_type, extra_metadata=None):
        """Helper function to add error logs"""
        log_metadata = {
            "document_name": blobname if 'blobname' in locals() else None,
            "document_type": "Lease",
            "processing_path": metadata["paths"].get("pdf_path") if metadata.get("paths") else None
        }
        if extra_metadata:
            log_metadata.update(extra_metadata)
            
        error_logs.append({
            "document_id": metadata.get("doc_id"),
            "processor_type": "msa_lease",
            "request_id": request_id if 'request_id' in locals() else None,
            "timestamp": datetime.now(),
            "error_type": error_type,
            "error_message": str(error),
            "traceback": ''.join(traceback.format_exception(None, error, error.__traceback__)),
            "step": step,
            "metadata": log_metadata
        })

    # Initialize metadata structure
    metadata = {
        "doc_id": None,
        "paths": {
            "adobe_extract_path": None,
            "input_dir": None,
            "pdf_file_path": None,
            "adobe_output_dir": None,
            "adobe_blob_name": None
        },
        "status": {
            "message": None,
            "state": None
        },
        "document_info": {
            "uploaded_at": datetime.now()
        }
    }

    try:
        # Message unpacking step
        current_step = "message_unpacking"
        add_info_log(current_step, "Starting to unpack message data")
        
        msg_type, msg_data = msg
        app_insight_logger = msg_data[6]
        app_insight_logger.info(f"Processing {msg_type}: {msg_data}", extra=properties)
        
        # Extract doc_id early for error tracking
        if len(msg_data) >= 2:
            doc_id = metadata["doc_id"] = msg_data[1]
            add_info_log(current_step, "Successfully extracted document ID", {
                "doc_id": doc_id
            })
        else:
            error_msg = "Invalid message data: missing doc_id"
            add_error_log(current_step, error_msg, "MessageValidationError")
            metadata["status"].update({
                "message": error_msg,
                "state": "Failed"
            })
            return None, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}

        # Unpack remaining message data
        if len(msg_data) == 9:
            (blobname, _, _, container_name, _, folder_name, appisnlogger, request_id, split_indicesdict) = msg_data
            folder_name = "MSALeaseAgreementInput"
        else:
            error_msg = "Invalid message data format"
            add_error_log(current_step, error_msg, "MessageValidationError")
            return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}

        add_info_log(current_step, "Successfully unpacked message data", {
            "folder_name": folder_name
        })

        # Path setup step
        current_step = "path_setup"
        add_info_log(current_step, "Setting up processing paths")
        
        input_dir = os.path.join("./", "MSALease", doc_id)
        pdf_file_path = os.path.join(input_dir, blobname)
        adobe_output_dir = os.path.join(input_dir, "adobe_output")

        metadata["paths"].update({
            "input_dir": input_dir,
            "pdf_file_path": pdf_file_path,
            "adobe_output_dir": adobe_output_dir
        })
        
        add_info_log(current_step, "Path setup completed", {
            "input_dir": input_dir,
            "pdf_file_path": pdf_file_path
        })

        # Create directories if they don't exist
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(adobe_output_dir, exist_ok=True)

        # PDF processing step
        current_step = "pdf_processing"
        if not os.path.isfile(pdf_file_path):
            add_info_log(current_step, f"Downloading PDF from blob storage: {blobname}")
            try:
                download_result = download_blob(
                    blobname,
                    input_dir=input_dir,
                    container_name=f"{container_name}/{folder_name}"
                )
                download_result.unwrap()
                add_info_log(current_step, "PDF download completed successfully")
            except Exception as e:
                add_error_log(current_step, e, "BlobDownloadError")
                metadata["status"].update({
                    "message": f"Failed to download blob: {str(e)}",
                    "state": "Failed"
                })
                return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}

        # PDF encoding check
        add_info_log(current_step, "Checking PDF encoding")
        try:
            check_encoding_result = checkPDFencodingerrors(pdf_file_path)
            pdf_processing_path = check_encoding_result.unwrap()
            add_info_log(current_step, "PDF encoding check completed")
        except Exception as e:
            add_error_log(current_step, e, "PDFEncodingError")
            return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}

        # Adobe API processing step
        current_step = "adobe_processing"
        adobe_api_filename = f"adobe-api-{blobname.split('.')[0]}-{doc_id}.zip"
        adobe_api_zip_path = os.path.join(adobe_output_dir, adobe_api_filename)

        try:
            if not os.path.isfile(adobe_api_zip_path):
                add_info_log(current_step, "Starting Adobe API extraction")
                extract_result = run_extract_pdf(
                    filename=pdf_processing_path,
                    adobe_dir=adobe_api_zip_path,
                    logger_name=appisnlogger,
                    request_id=request_id
                )
                extract_result.unwrap()
                
                add_info_log(current_step, "Extracting Adobe API results")
                extract_zip_result = extractZipNew(zip_file_path=adobe_api_zip_path)
                extract_zip_result.unwrap()

                add_info_log(current_step, "Uploading Adobe extraction results")
                upload_result = upload_blob(
                    adobe_api_filename,
                    filepath=adobe_api_zip_path,
                    container_name=f"{OCR_CONTAINER_NAME}/MSALeaseAgreementOuput",
                    content_type="application/zip"
                )
                upload_result.unwrap()  # Unwrap the monad
                transformed_adobe_filename = adobe_api_filename  # Since we're not transforming the name anymore
                add_info_log(current_step, "Adobe upload completed successfully", {
                    "original_name": adobe_api_filename,
                    "transformed_name": transformed_adobe_filename
                })

            adobe_api_path = f'https://{BLOB_ACCOUNT_NAME}.blob.core.windows.net/{OCR_CONTAINER_NAME}/MSALeaseAgreementOuput/{adobe_api_filename}'
            metadata["paths"].update({
                "adobe_extract_path": adobe_api_path,
                "adobe_blob_name": adobe_api_filename
            })
            add_info_log(current_step, "Adobe processing completed successfully")

        except Exception as e:
            add_error_log(current_step, e, "AdobeAPIError")
            metadata["status"].update({
                "message": f"Adobe API processing failed: {str(e)}",
                "state": "Failed"
            })
            return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}

        # Final success status
        current_step = "completion"
        metadata["status"].update({
            "message": "Successfully processed document",
            "state": "Ingested"
        })
        add_info_log(current_step, "Document processing completed successfully", {
            "final_status": "Ingested"
        })
        return doc_id, 'Ingested', metadata, msg, {"info": info_logs, "error": error_logs}

    except Exception as e:
        add_error_log(current_step, e, "UnhandledError")
        metadata["status"].update({
            "message": f"Unhandled exception in process_msa_lease_item: {str(e)}",
            "state": "Failed"
        })
        return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}
    
async def process_msa_lease_batch(batch):
    """
    Process a batch of MSA Lease documents with comprehensive logging.
    Handles parallel processing, DB operations, and Azure Service Bus integration.
    """
    app_insight_logger.info(f"Processing MSA Lease batch: {batch}", extra=properties)
    loop = asyncio.get_running_loop()
    credential = None

    async def store_logs(doc_id, logs_data):
        """Store info and error logs in their respective collections"""
        try:
            if logs_data.get("info"):
                await asyto_mongo(logs_data["info"], "DataIngestionInfoLog")
                app_insight_logger.info(f"Stored {len(logs_data['info'])} info logs for document {doc_id}", extra=properties)
                
            if logs_data.get("error"):
                await asyto_mongo(logs_data["error"], "DataIngestionErrorLog")
                app_insight_logger.info(f"Stored {len(logs_data['error'])} error logs for document {doc_id}", extra=properties)
                
        except Exception as e:
            app_insight_logger.error(f"Failed to store logs for document {doc_id}: {str(e)}", extra=properties)

    async def add_batch_log(log_type, step, message, error=None, metadata=None):
        """Add a log entry for batch-level operations"""
        try:
            log_entry = {
                "document_id": "batch_operation",
                "processor_type": "msa_lease_batch",
                "timestamp": datetime.now(),
                "step": step,
                "metadata": metadata or {}
            }
            
            if log_type == "info":
                log_entry["message"] = message
                await asyto_mongo([log_entry], "DataIngestionInfoLog")
            else:  # error
                log_entry.update({
                    "error_type": "BatchError",
                    "error_message": message,
                    "traceback": ''.join(traceback.format_exception(None, error, error.__traceback__)) if error else None
                })
                await asyto_mongo([log_entry], "DataIngestionErrorLog")
        except Exception as e:
            app_insight_logger.error(f"Failed to add batch log: {str(e)}", extra=properties)

    async def cleanup_directory(doc_id):
        """Clean up the document directory after successful processing"""
        try:
            doc_dir = os.path.join("./", "MSALease", doc_id)
            if os.path.exists(doc_dir):
                shutil.rmtree(doc_dir)
                await add_batch_log("info", "cleanup", 
                    f"Successfully cleaned up directory for document {doc_id}", 
                    metadata={"cleaned_path": doc_dir})
        except Exception as e:
            await add_batch_log("error", "cleanup", 
                f"Error cleaning up directory for document {doc_id}", 
                error=e,
                metadata={"attempted_path": doc_dir})
            raise

    try:
        # Initialize Azure credentials
        try:
            credential = ManagedIdentityCredential() if ENV_NAME in ENV_NAMES else DefaultAzureCredential()
            await add_batch_log("info", "azure_init", "Successfully initialized Azure credentials")
        except Exception as e:
            await add_batch_log("error", "azure_init", "Failed to initialize Azure credentials", e)
            raise

        # Initialize tracking collections
        notify_messages = []
        await add_batch_log("info", "initialization", "Initialized batch processing collections")

        try:
            # Initialize ServiceBusClient for notifications
            async with ServiceBusClient(
                fully_qualified_namespace=FULLY_QUALIFIED_NAME,
                credential=credential,
                logging_enable=True
            ) as servicebus_client:
                sender = servicebus_client.get_queue_sender(queue_name=MSALNotify)
                
                try:
                    async with sender:
                        # Process items in parallel
                        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                            tasks = [loop.run_in_executor(executor, process_msa_lease_item, msg) for msg in batch]
                            
                            for future in asyncio.as_completed(tasks):
                                try:
                                    doc_id, status, metadata, original_message, logs_data = await future
                                    
                                    # Store processing logs
                                    await store_logs(doc_id, logs_data)

                                    if not doc_id:
                                        await add_batch_log("error", "validation", "Invalid document result - missing doc_id")
                                        continue

                                    try:
                                        # Update MongoDB based on processing status
                                        update_data = {
                                            "$set": {
                                                "Status": status,
                                                "ProcessedAt": datetime.now(),
                                                "AdobeExtractPath": metadata["paths"]["adobe_extract_path"]
                                            }
                                        }

                                        if status == "Failed":
                                            update_data["$set"].update({
                                                "FailureReason": metadata["status"]["message"]
                                            })
                                            update_data["$inc"] = {"RetryCount": 1}

                                        # Update document in master_lease_documents collection
                                        await update_status_mongo_async(
                                            {"ID": doc_id},
                                            update_data,
                                            "master_lease_documents"
                                        )

                                        msadoc = await get_document_async({"ID":doc_id},"master_lease_documents")

                                        DocumentPath = msadoc.get("DocumentPath")

                                        # If processing was successful
                                        if status == "Ingested":
                                            # Prepare notification message
                                            notify_message = {
                                                "eventTime": datetime.strftime(datetime.now(), "%Y-%m-%dT%H:%M:%S"),
                                                "data": {
                                                    "adobe_blob_name": metadata["paths"]["adobe_extract_path"],
                                                    "doc_id": doc_id,
                                                    "url": DocumentPath
                                                }
                                            }
                                            notify_messages.append(json.dumps([notify_message]))
                                            await add_batch_log("info", "notification_prep", 
                                                f"Prepared notification message for document {doc_id}")

                                            # Clean up directory for successful processing
                                            try:
                                                await cleanup_directory(doc_id)
                                            except Exception as e:
                                                app_insight_logger.error(f"Directory cleanup failed for {doc_id}: {str(e)}", extra=properties)
                                                # Continue processing even if cleanup fails

                                    except Exception as e:
                                        await add_batch_log("error", "document_update", 
                                            f"Error updating document {doc_id}", e)
                                        continue

                                except Exception as e:
                                    await add_batch_log("error", "future_processing", 
                                        "Error processing future result", e)
                                    continue

                            # Send all notification messages
                            if notify_messages:
                                try:
                                    await add_batch_log("info", "notification_send", 
                                        f"Sending {len(notify_messages)} notification messages")
                                    service_bus_msgs = [ServiceBusMessage(msg) for msg in notify_messages]
                                    await sender.send_messages(service_bus_msgs)
                                    await add_batch_log("info", "notification_send", 
                                        f"Successfully sent {len(notify_messages)} notification messages")
                                except Exception as e:
                                    await add_batch_log("error", "notification_send", 
                                        "Error sending notification messages", e)
                                    raise

                except Exception as e:
                    await add_batch_log("error", "sender_context", "Error in sender context", e)
                    raise

        except Exception as e:
            await add_batch_log("error", "servicebus_client", "Error in ServiceBusClient", e)
            raise

    except Exception as e:
        err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
        await add_batch_log("error", "critical_error", 
            "Critical error in batch processing", e, 
            {"error_details": err_msg})
        raise

    finally:
        if credential:
            try:
                await credential.close()
                await add_batch_log("info", "cleanup", "Closed Azure credentials")
            except Exception as e:
                await add_batch_log("error", "cleanup", "Error closing Azure credentials", e)        

async def process_batches():
    """Main coroutine to check queues and process batches until stop_event is set."""
    global unlock_timer, timer_cancelled, declaration_lock
    global msa_unlock_timer, msa_timer_cancelled, msa_priority_lock
    global temp_base_endo_storage, temp_merged_storage, temp_adobe_retry_storage
    global temp_extraction_retry_storage, temp_manual_split_storage, temp_declaration_storage
    
    try:
        # Fetch carriers with base policy
        fetch_result = await fetch_carriers_with_base_policy(MONGO_POLICIES_COLLECTION)
        carriers_with_base = fetch_result.unwrap()
        
        while not stop_event.is_set():
            try:
                # Check MSA queue first - highest priority
                if not MSALeaseQueue.empty():
                    # Activate MSA priority if not already active
                    if not msa_priority_lock.is_set():
                        msa_priority_lock.set()
                        app_insight_logger.info("MSA PRIORITY ACTIVATED: MSA Lease queue has messages", extra=properties)
                        
                        # Cancel any existing MSA unlock timer
                        if msa_unlock_timer:
                            msa_unlock_timer.cancel()
                            msa_timer_cancelled = True
                            msa_unlock_timer = None
                
                # MSA Lease queue processing (always processed regardless of priority)
                work_done = False
                if not MSALeaseQueue.empty():
                   try:
                     msa_lease_batches = form_batches(MSALeaseQueue, QUEUE_PROCESS_BATCH_SIZE)
                     for batch in msa_lease_batches:
                         await process_msa_lease_batch(batch)
                         work_done = True
                         
                         # Check if MSA queue is now empty after this batch
                         if MSALeaseQueue.empty() and msa_priority_lock.is_set() and msa_unlock_timer is None:
                             app_insight_logger.info("Starting MSA unlock timer: Normal processing will resume in 3 seconds if MSA queue remains empty", extra=properties)
                             msa_unlock_timer = Timer(3, async_msa_unlock_callback)  # 3 second timer
                             msa_timer_cancelled = False
                         
                   except Exception as e:
                     err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
                     app_insight_logger.error(f"Error processing MSA Lease queue: {err_msg}", extra=properties)

                   # Reset declaration lock if work was done
                   if work_done and declaration_lock.is_set():
                         declaration_lock.clear()
                         app_insight_logger.info("LOCK ACQUIRED: Lock was reset after MSA Lease processing", extra=properties)
                         if unlock_timer:
                             unlock_timer.cancel()
                             unlock_timer = None
                
                # Only process other queues if MSA priority is not active
                if not msa_priority_lock.is_set():
                    # Process base and endorsement queue
                    work_done = False
                    if not base_endo_queue.empty():
                        try:
                            all_messages = form_batches(base_endo_queue, QUEUE_PROCESS_BATCH_SIZE)
                            base_messages = []
                            current_endorsements = []
                            
                            # Sort messages into base and endorsement categories
                            for batch in all_messages:
                                for msg in batch:
                                    folder_name = msg[1][5]
                                    carrier_name = msg[1][4]
                                    work_done = True  # Messages were found and sorted
                                    
                                    if "base" in folder_name.lower():
                                        base_messages.append(msg)
                                    elif "endorsement" in folder_name.lower():
                                        if carrier_name in carriers_with_base:
                                            current_endorsements.append(msg)
                                        else:
                                            if carrier_name not in waiting_endorsements:
                                                waiting_endorsements[carrier_name] = []
                                            waiting_endorsements[carrier_name].append(msg)
                                            app_insight_logger.info(
                                                f"Stored endorsement for carrier {carrier_name} in waiting queue",
                                                extra=properties
                                            )
                            
                            # Process base messages batch by batch with MSA priority check
                            if base_messages:
                                try:
                                    base_batches = [
                                        base_messages[i:i + QUEUE_PROCESS_BATCH_SIZE] 
                                        for i in range(0, len(base_messages), QUEUE_PROCESS_BATCH_SIZE)
                                    ]
                                    
                                    for batch_idx, batch in enumerate(base_batches):
                                        # Check MSA priority before processing each batch
                                        if msa_priority_lock.is_set():
                                            # MSA priority activated, push remaining batches back to queue
                                            app_insight_logger.info(
                                                f"MSA priority detected during base processing. Returning {len(base_batches) - batch_idx} batches to queue.",
                                                extra=properties
                                            )
                                            # Return unprocessed messages to queue
                                            remaining_messages = []
                                            for remaining_batch in base_batches[batch_idx:]:
                                                remaining_messages.extend(remaining_batch)
                                            for msg in remaining_messages:
                                                await base_endo_queue.put(msg)
                                            break
                                        
                                        # Process this batch
                                        await process_base_endo_batch(batch)
                                        
                                        # If this was the last batch, update carriers list
                                        if batch_idx == len(base_batches) - 1:
                                            # Update carriers with base policy
                                            fetch_result = await fetch_carriers_with_base_policy(MONGO_POLICIES_COLLECTION)
                                            carriers_with_base = fetch_result.unwrap()
                                            
                                            # Process any waiting endorsements if MSA priority not set
                                            if not msa_priority_lock.is_set():
                                                for carrier in carriers_with_base:
                                                    if carrier in waiting_endorsements:
                                                        carrier_endorsements = waiting_endorsements.pop(carrier)
                                                        current_endorsements.extend(carrier_endorsements)
                                                        app_insight_logger.info(
                                                            f"Processing {len(carrier_endorsements)} waiting endorsements for carrier {carrier}", 
                                                            extra=properties
                                                        )
                                    
                                except Exception as e:
                                    err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
                                    app_insight_logger.error(f"Error processing base messages: {err_msg}", extra=properties)
                            
                            # Process current endorsements with MSA priority check
                            if current_endorsements and not msa_priority_lock.is_set():
                                try:
                                    endo_batches = [
                                        current_endorsements[i:i + QUEUE_PROCESS_BATCH_SIZE] 
                                        for i in range(0, len(current_endorsements), QUEUE_PROCESS_BATCH_SIZE)
                                    ]
                                    
                                    for batch_idx, batch in enumerate(endo_batches):
                                        # Check MSA priority before processing each batch
                                        if msa_priority_lock.is_set():
                                            # MSA priority activated, push remaining batches back to queue
                                            app_insight_logger.info(
                                                f"MSA priority detected during endorsement processing. Returning {len(endo_batches) - batch_idx} batches to queue.",
                                                extra=properties
                                            )
                                            # Return unprocessed messages to queue
                                            remaining_messages = []
                                            for remaining_batch in endo_batches[batch_idx:]:
                                                remaining_messages.extend(remaining_batch)
                                            for msg in remaining_messages:
                                                await base_endo_queue.put(msg)
                                            break
                                            
                                        # Process this batch
                                        await process_base_endo_batch(batch)
                                        
                                except Exception as e:
                                    err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
                                    app_insight_logger.error(f"Error processing endorsement messages: {err_msg}", extra=properties)
                                    
                        except Exception as e:
                            err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
                            app_insight_logger.error(f"Error processing base/endorsement queue: {err_msg}", extra=properties)

                        # Check if lock needs to be reset after base/endo processing
                        if work_done and declaration_lock.is_set():
                            declaration_lock.clear()
                            app_insight_logger.info("LOCK ACQUIRED: Lock was reset after base/endo processing", extra=properties)
                            if unlock_timer:
                                unlock_timer.cancel()
                                unlock_timer = None
                    
                    # Check MSA priority again before continuing to next queue
                    if msa_priority_lock.is_set():
                        app_insight_logger.info("MSA priority activated. Skipping remaining queue processing.", extra=properties)
                        continue
                
                    # Process merged queue
                    work_done = False
                    if not merged_queue.empty():
                        try:
                            merged_batches = form_batches(merged_queue, QUEUE_PROCESS_BATCH_SIZE)
                            
                            for batch_idx, batch in enumerate(merged_batches):
                                # Check MSA priority before processing each batch
                                if msa_priority_lock.is_set():
                                    # MSA priority activated, push remaining batches back to queue
                                    app_insight_logger.info(
                                        f"MSA priority detected during merged processing. Returning {len(merged_batches) - batch_idx} batches to queue.",
                                        extra=properties
                                    )
                                    # Return unprocessed messages to queue
                                    remaining_messages = []
                                    for remaining_batch in merged_batches[batch_idx:]:
                                        remaining_messages.extend(remaining_batch)
                                    for msg in remaining_messages:
                                        await merged_queue.put(msg)
                                    break
                                    
                                # Process this batch
                                await process_merged_batch(batch)
                                work_done = True
                                            
                        except Exception as e:
                            err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
                            app_insight_logger.error(f"Error processing merged queue: {err_msg}", extra=properties)

                        # Check if lock needs to be reset after merged processing
                        if work_done and declaration_lock.is_set():
                            declaration_lock.clear()
                            app_insight_logger.info("LOCK ACQUIRED: Lock was reset after merged processing", extra=properties)
                            if unlock_timer:
                                unlock_timer.cancel()
                                unlock_timer = None
                
                    # Check MSA priority again before continuing to next queue
                    if msa_priority_lock.is_set():
                        app_insight_logger.info("MSA priority activated. Skipping remaining queue processing.", extra=properties)
                        continue
                
                    # Process adobe retry queue with similar MSA priority checking
                    work_done = False
                    if not adobe_retry_queue.empty():
                        try:
                            adobe_retry_batches = form_batches(adobe_retry_queue, QUEUE_PROCESS_BATCH_SIZE)
                            
                            for batch_idx, batch in enumerate(adobe_retry_batches):
                                # Check MSA priority before processing each batch
                                if msa_priority_lock.is_set():
                                    # Return unprocessed messages to queue
                                    remaining_messages = []
                                    for remaining_batch in adobe_retry_batches[batch_idx:]:
                                        remaining_messages.extend(remaining_batch)
                                    for msg in remaining_messages:
                                        await adobe_retry_queue.put(msg)
                                    app_insight_logger.info(
                                        f"MSA priority detected. Returned {len(remaining_messages)} adobe retry messages to queue.",
                                        extra=properties
                                    )
                                    break
                                
                                await process_adobe_retry_batch(batch)
                                work_done = True
                                
                        except Exception as e:
                            err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
                            app_insight_logger.error(f"Error processing adobe retry queue: {err_msg}", extra=properties)

                        # Check if lock needs to be reset
                        if work_done and declaration_lock.is_set():
                            declaration_lock.clear()
                            app_insight_logger.info("LOCK ACQUIRED: Lock was reset after adobe retry processing", extra=properties)
                            if unlock_timer:
                                unlock_timer.cancel()
                                unlock_timer = None
                    
                    # Check MSA priority again before continuing
                    if msa_priority_lock.is_set():
                        app_insight_logger.info("MSA priority activated. Skipping remaining queue processing.", extra=properties)
                        continue
                
                    # Process extraction retry queue with similar MSA priority checking
                    work_done = False
                    if not extraction_retry_queue.empty():
                        try:
                            extraction_retry_batches = form_batches(extraction_retry_queue, QUEUE_PROCESS_BATCH_SIZE)
                            
                            for batch_idx, batch in enumerate(extraction_retry_batches):
                                # Check MSA priority before processing each batch
                                if msa_priority_lock.is_set():
                                    # Return unprocessed messages to queue
                                    remaining_messages = []
                                    for remaining_batch in extraction_retry_batches[batch_idx:]:
                                        remaining_messages.extend(remaining_batch)
                                    for msg in remaining_messages:
                                        await extraction_retry_queue.put(msg)
                                    app_insight_logger.info(
                                        f"MSA priority detected. Returned {len(remaining_messages)} extraction retry messages to queue.",
                                        extra=properties
                                    )
                                    break
                                
                                await process_extraction_retry_batch(batch)
                                work_done = True
                                
                        except Exception as e:
                            err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
                            app_insight_logger.error(f"Error processing extraction retry queue: {err_msg}", extra=properties)

                        # Check if lock needs to be reset
                        if work_done and declaration_lock.is_set():
                            declaration_lock.clear()
                            app_insight_logger.info("LOCK ACQUIRED: Lock was reset after extraction retry processing", extra=properties)
                            if unlock_timer:
                                unlock_timer.cancel()
                                unlock_timer = None
                
                    # Check MSA priority again before continuing
                    if msa_priority_lock.is_set():
                        app_insight_logger.info("MSA priority activated. Skipping remaining queue processing.", extra=properties)
                        continue
                
                    # Process manual split queue with similar MSA priority checking
                    work_done = False
                    if not manual_split_queue.empty():
                        try:
                            manual_split_batches = form_batches(manual_split_queue, QUEUE_PROCESS_BATCH_SIZE)
                            
                            for batch_idx, batch in enumerate(manual_split_batches):
                                # Check MSA priority before processing each batch
                                if msa_priority_lock.is_set():
                                    # Return unprocessed messages to queue
                                    remaining_messages = []
                                    for remaining_batch in manual_split_batches[batch_idx:]:
                                        remaining_messages.extend(remaining_batch)
                                    for msg in remaining_messages:
                                        await manual_split_queue.put(msg)
                                    app_insight_logger.info(
                                        f"MSA priority detected. Returned {len(remaining_messages)} manual split messages to queue.",
                                        extra=properties
                                    )
                                    break
                                
                                await process_manual_split_batch(batch)
                                work_done = True
                                
                        except Exception as e:
                            err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
                            app_insight_logger.error(f"Error processing manual split queue: {err_msg}", extra=properties)

                        # Check if lock needs to be reset
                        if work_done and declaration_lock.is_set():
                            declaration_lock.clear()
                            app_insight_logger.info("LOCK ACQUIRED: Lock was reset after manual split processing", extra=properties)
                            if unlock_timer:
                                unlock_timer.cancel()
                                unlock_timer = None
                
                    await cleanup_directory()
                    
                    # Check MSA priority again before processing declarations
                    if msa_priority_lock.is_set():
                        app_insight_logger.info("MSA priority activated. Skipping declaration queue processing.", extra=properties)
                        continue
                
                    # Process declarations last
                    if not declaration_queue.empty():
                        while not declaration_queue.empty():
                            # Check MSA priority before moving items
                            if msa_priority_lock.is_set():
                                app_insight_logger.info("MSA priority activated. Skipping declaration queue processing.", extra=properties)
                                break
                                
                            try:
                                item = declaration_queue.get_nowait()
                                await declaration_temp_queue.put(item)
                            except asyncio.QueueEmpty:
                                break
                        app_insight_logger.info(f"Items in declaration_temp_queue: {declaration_temp_queue.qsize()}", extra=properties)

                    if not declaration_temp_queue.empty():
                        if declaration_lock.is_set():
                            try:
                                dec_batches = form_batches(declaration_temp_queue, QUEUE_PROCESS_BATCH_SIZE)
                                
                                for batch_idx, batch in enumerate(dec_batches):
                                    # Check MSA priority before processing each batch
                                    if msa_priority_lock.is_set():
                                        # Return unprocessed batches to queue
                                        remaining_messages = []
                                        for remaining_batch in dec_batches[batch_idx:]:
                                            remaining_messages.extend(remaining_batch)
                                        for msg in remaining_messages:
                                            await declaration_temp_queue.put(msg)
                                        app_insight_logger.info(
                                            f"MSA priority detected during declaration processing. Returning {len(remaining_messages)} messages to temp queue.",
                                            extra=properties
                                        )
                                        break
                                    
                                    await process_declarations(batch)
                                
                            except Exception as e:
                                declaration_lock.clear()
                                err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
                                app_insight_logger.error(f"Error processing declaration queue: {err_msg}", extra=properties)
                                app_insight_logger.info("LOCK ACQUIRED: Declaration processing locked due to processing error", extra=properties)
                                raise
                        else:
                            app_insight_logger.info("Declaration queue not empty but processing is locked", extra=properties)
                
                    # Check if we can start timer after EVERYTHING
                    if base_endo_queue.empty() and merged_queue.empty() and extraction_retry_queue.empty() and adobe_retry_queue.empty() and manual_split_queue.empty() and unlock_timer is None:
                        app_insight_logger.info("Starting unlock timer: Declaration processing will be allowed in 40 seconds if conditions remain met", extra=properties)
                        unlock_timer = Timer(40, async_unlock_callback)
                        timer_cancelled = False

                await asyncio.sleep(3.5)        
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                app_insight_logger.error(f"Error in process_batches main loop: {str(e)}", extra=properties)
                traceback.print_exc()
                await asyncio.sleep(1)
                
    except Exception as e:
        app_insight_logger.error(f"Fatal error in process_batches: {str(e)}", extra=properties)
        traceback.print_exc()
    finally:
        app_insight_logger.info("Closing process_batches", extra=properties)

def form_batches(queue, batch_size):
    """Forms multiple batches from a queue up to the specified batch size per batch."""
    batches = []
    
    while not queue.empty():  # Continue as long as there are items in the queue
        current_batch = []
        while not queue.empty() and len(current_batch) < batch_size:
            try:
                item = queue.get_nowait()
                current_batch.append(item)
            except asyncio.QueueEmpty:
                break
        if current_batch:
            batches.append(current_batch)
    
    return batches

async def process_merged_batch(batch):
    """
    Process a batch of merged documents with enhanced logging.
    Handles parallel processing and database operations for both documents and logs.
    """
    app_insight_logger.info(f"Processing merged document batch: {batch}", extra=properties)
    loop = asyncio.get_running_loop()

    async def store_logs(doc_id, logs_data):
        """Store info and error logs in their respective collections"""
        try:
            if logs_data.get("info"):
                await asyto_mongo(logs_data["info"], "DataIngestionInfoLog")
                app_insight_logger.info(f"Stored {len(logs_data['info'])} info logs for document {doc_id}", extra=properties)
                
            if logs_data.get("error"):
                await asyto_mongo(logs_data["error"], "DataIngestionErrorLog")
                app_insight_logger.info(f"Stored {len(logs_data['error'])} error logs for document {doc_id}", extra=properties)
                
        except Exception as e:
            app_insight_logger.error(f"Failed to store logs for document {doc_id}: {str(e)}", extra=properties)

    try:
        # First create placeholder entries for all documents in batch
        for msg in batch:
            try:
                _, msg_data = msg
                blobname, doc_id, _, _, carrier_name, _, _, _, split_indicesdict = msg_data if len(msg_data) == 9 else (*msg_data, {})

                # Only create placeholder if not from frontend and document doesn't exist
                #if 'frontendflag' not in split_indicesdict or split_indicesdict.get('frontendflag') != 'Y':
                # Check if document already exists
                existing_doc = await get_document_async({"ID": doc_id}, MONGO_DOCUMENT_COLLECTION)
                app_insight_logger.info(f"\nEXISTING DOC: \n\n{existing_doc}\n{blobname}\n{split_indicesdict.get('frontendflag')}")
                if "==" in blobname:
                    filename_parts = blobname.split("==")
                    if len(filename_parts) == 4:  # Original name, UserID, PolicyName.pdf
                        user_id = filename_parts[2]
                else:
                    user_id = None
                if not existing_doc:
                    # For frontend uploads, use transformed name in the document
                        #user_id = blobname.split('==')[1].split('.')[0] if '==' in blobname else None
                    if split_indicesdict.get('frontendflag') == 'Y':
                        transformed_name = transform_frontend_blob_name(blobname)
                        
                        doc_result = create_document(
                            doc_id=doc_id,
                            doc_name=transformed_name,  # Use transformed name
                            carrier_name=carrier_name,
                            doc_type="Merged Document"
                        )
                        doc_obj = doc_result.unwrap()
                        
                        # Add UserID and original filename for frontend uploads
                        doc_obj['UserID'] = user_id
                        #doc_obj['original_filename'] = blobname
                        
                        await asyto_mongo([doc_obj], MONGO_DOCUMENT_COLLECTION)
                        app_insight_logger.info(f"Created frontend placeholder entry for merged document {doc_id}", extra=properties)
                    else:
                        # Regular non-frontend document creation
                        doc_result = create_document(
                            doc_id=doc_id,
                            doc_name=blobname,
                            carrier_name=carrier_name,
                            doc_type="Merged Document"
                        )
                        doc_obj = doc_result.unwrap()
                        await asyto_mongo([doc_obj], MONGO_DOCUMENT_COLLECTION)
                        app_insight_logger.info(f"Created placeholder entry for merged document {doc_id}", extra=properties)
                else:
                    app_insight_logger.info(f"Document {doc_id} already exists, skipping placeholder creation", extra=properties)
            except Exception as e:
                app_insight_logger.error(f"Failed to create placeholder for document: {str(e)}", extra=properties)
                continue

        # Process documents in parallel
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            tasks = [loop.run_in_executor(executor, process_merged_item, msg) for msg in batch]
            
            for future in asyncio.as_completed(tasks):
                try:
                    doc_id, status, metadata, original_message, logs_data = await future
                    
                    # Store processing logs
                    await store_logs(doc_id, logs_data)

                    if not doc_id or not metadata:
                        app_insight_logger.error("Invalid document result - missing doc_id or metadata", extra=properties)
                        continue

                    # Handle failures
                    if metadata["status"]["state"] == "Failed":
                        try:
                            doc = await get_document_async({"ID": doc_id}, MONGO_DOCUMENT_COLLECTION)
                            retry_count = doc.get('RetryCount', 0) if doc else 0

                            if retry_count < 3:
                                new_retry_count = retry_count + 1
                                
                                # Update document status for retry - using $set explicitly
                                await update_status_mongo_async(
                                    {"ID": doc_id},
                                    {
                                        "$set": {
                                            "Status": "RETRYING_INGESTION",
                                            "RetryCount": new_retry_count,
                                            "FailureReason": f"Attempt {new_retry_count}: {metadata['status']['message']}",
                                            "FailureStatusCode": [metadata['status']['code']],
                                            "DocumentPath": metadata["paths"].get("pdf_path"),
                                            "AdobeExtractPath": metadata["paths"].get("adobe_extract_path")
                                        }
                                    },
                                    MONGO_DOCUMENT_COLLECTION
                                )
                                
                                # Log retry attempt
                                retry_log = {
                                    "document_id": doc_id,
                                    "processor_type": "merged",
                                    "timestamp": datetime.now(),
                                    "step": "retry_processing",
                                    "message": f"Queuing document for retry attempt {new_retry_count}",
                                    "metadata": {
                                        "retry_count": new_retry_count,
                                        "failure_reason": metadata['status']['message'],
                                        "status_code": metadata['status']['code']
                                    }
                                }
                                await asyto_mongo([retry_log], "DataIngestionInfoLog")
                                
                                await merged_queue.put(original_message)
                                app_insight_logger.info(f"Requeued document {doc_id} for retry attempt {new_retry_count}", extra=properties)
                            else:
                                # Update document status for final failure
                                await update_status_mongo_async(
                                    {"ID": doc_id},
                                    {
                                        "$set": {
                                            "Status": "Failed",
                                            "FailureReason": f"All retry attempts exhausted. Final error: {metadata['status']['message']}",
                                            "FailureStatusCode": [metadata['status']['code']],
                                            "DocumentPath": metadata["paths"].get("pdf_path"),
                                            "AdobeExtractPath": metadata["paths"].get("adobe_extract_path")
                                        }
                                    },
                                    MONGO_DOCUMENT_COLLECTION
                                )
                                
                                # Log final failure
                                final_failure_log = {
                                    "document_id": doc_id,
                                    "processor_type": "merged",
                                    "timestamp": datetime.now(),
                                    "step": "final_failure",
                                    "message": "Document processing failed after all retry attempts",
                                    "metadata": {
                                        "total_retries": retry_count,
                                        "final_error": metadata['status']['message'],
                                        "status_code": metadata['status']['code']
                                    }
                                }
                                await asyto_mongo([final_failure_log], "DataIngestionInfoLog")
                                
                        except Exception as e:
                            app_insight_logger.error(f"Error handling document failure for {doc_id}: {str(e)}", extra=properties)
                        continue

                    # Handle successful processing
                    try:
                        # First, update the document with new fields
                        update_fields = {
                            "$set": {
                                "Status": "Processing",
                                "DocumentPath": metadata["paths"].get("pdf_path"),
                                "AdobeExtractPath": metadata["paths"].get("adobe_extract_path"),
                                "UploadedAt": metadata["processing_info"]["uploaded_at"],
                                "JsonData": metadata["processing_info"]["split_result"]
                            },
                            "$unset": {
                                "FailureReason": "",
                                "FailureStatusCode": "",
                                "RetryCount": ""
                            }
                        }
                        
                        # Update merged document status
                        await update_status_mongo_async(
                            {"ID": doc_id},
                            update_fields,  # This will be passed through without additional wrapping
                            MONGO_DOCUMENT_COLLECTION
                        )

                        # Log successful processing
                        success_log = {
                            "document_id": doc_id,
                            "processor_type": "merged",
                            "timestamp": datetime.now(),
                            "step": "processing_complete",
                            "message": "Document successfully processed and status updated",
                            "metadata": {
                                "total_parts": metadata["processing_info"]["total_parts"],
                                "upload_time": metadata["processing_info"]["uploaded_at"].isoformat(),
                                "pdf_path": metadata["paths"].get("pdf_path"),
                                "adobe_path": metadata["paths"].get("adobe_extract_path")
                            }
                        }
                        await asyto_mongo([success_log], "DataIngestionInfoLog")

                        # Set up tracking for frontend uploads
                        if metadata["processing_info"]["split_indices"].get('frontendflag') == 'Y':
                            merged_documents_tracking[doc_id] = {
                                'total_parts': metadata["processing_info"]["total_parts"],
                                'processed_parts': set(),
                                'failed_parts': {},
                                'status': 'processing',
                                'DocumentName': original_message[1][0],
                                'CarrierName': metadata["document_info"]["carrier_name"],
                                'DocumentPath': metadata["paths"].get("pdf_path"),
                                'AdobeExtractPath': metadata["paths"].get("adobe_extract_path"),
                                'ProcessedAt': metadata["processing_info"]["uploaded_at"],
                                'JsonData': metadata["processing_info"]["split_result"],
                                'MetaData': None,
                                'WaitingList': None,
                                'RetryCount': 0,
                                'FailureReason': None,
                                'FormNumber': None,
                                'NormalizedFormNumber': None
                            }

                            # Log frontend tracking setup
                            frontend_log = {
                                "document_id": doc_id,
                                "processor_type": "merged",
                                "timestamp": datetime.now(),
                                "step": "frontend_tracking_setup",
                                "message": "Frontend tracking initialized for document",
                                "metadata": {
                                    "total_parts": metadata["processing_info"]["total_parts"],
                                    "carrier_name": metadata["document_info"]["carrier_name"]
                                }
                            }
                            await asyto_mongo([frontend_log], "DataIngestionInfoLog")

                        # Process split documents
                        if metadata["processing_info"]["processed_parts"]:
                            for part in metadata["processing_info"]["processed_parts"]:
                                split_doc_id = part["doc_id"]
                                folder_name = part["folder_name"]

                                # Get split data from JsonData
                                if isinstance(metadata["processing_info"]["split_result"], dict) and split_doc_id in metadata["processing_info"]["split_result"]:
                                    split_data = metadata["processing_info"]["split_result"][split_doc_id]
                                    # pages = {
                                    #     "StartPage": split_data["page_num"][0][0],
                                    #     "EndPage": split_data["page_num"][0][1]
                                    #     }
                                    # Construct message data
                                    split_msg = [
                                        original_message[1][0],
                                        split_doc_id,
                                        original_message[1][2],
                                        original_message[1][3],
                                        metadata["document_info"]["carrier_name"],
                                        folder_name,
                                        original_message[1][6],
                                        original_message[1][7],
                                        split_data
                                    ]
                                    
                                    try:
                                        if folder_name in ("Base", "Endorsement"):
                                            # Handle base policy creation
                                            if folder_name == "Base":
                                                policy_no = str(uuid.uuid4())
                                                if metadata["processing_info"]["split_indices"].get('frontendmergedupload') != 'Y':
                                                    policy_result = create_policy(
                                                        policy_number=policy_no,
                                                        carriername=metadata["document_info"]["carrier_name"],
                                                        base_doc_id=split_doc_id
                                                    )
                                                    policy_obj = policy_result.unwrap()
                                                    policy_obj['MergedDocumentID'] = doc_id
                                                    await asyto_mongo([policy_obj], MONGO_POLICIES_COLLECTION)

                                                    # Log policy creation with MergedDocumentID
                                                    policy_log = {
                                                        "document_id": split_doc_id,
                                                        "processor_type": "merged",
                                                        "timestamp": datetime.now(),
                                                        "step": "policy_creation",
                                                        "message": "Created new policy for base document",
                                                        "metadata": {
                                                            "policy_number": policy_no,
                                                            "carrier_name": metadata["document_info"]["carrier_name"],
                                                            "merged_document_id": doc_id
                                                        }
                                                    }
                                                    await asyto_mongo([policy_log], "DataIngestionInfoLog")

                                            # Create document entry for split document
                                            if metadata["processing_info"]["split_indices"].get('frontendflag') == 'Y':
                                                # For frontend uploads, use transformed name and include UserID
                                                transformed_name = transform_frontend_blob_name(split_msg[0])
                                                #user_id = split_msg[0].split('==')[1].split('.')[0] if '==' in split_msg[0] else None
                                                
                                                doc_result = create_document(
                                                    doc_id=split_doc_id,
                                                    doc_name=transformed_name,
                                                    carrier_name=metadata["document_info"]["carrier_name"],
                                                    doc_type="BasePolicy" if folder_name == "Base" else "Endorsement"
                                                )
                                                doc_obj = doc_result.unwrap()
                                                doc_obj['UserID'] = user_id
                                                #doc_obj['original_filename'] = split_msg[0]
                                            else:
                                                doc_result = create_document(
                                                    doc_id=split_doc_id,
                                                    doc_name=split_msg[0],
                                                    carrier_name=metadata["document_info"]["carrier_name"],
                                                    doc_type="BasePolicy" if folder_name == "Base" else "Endorsement"
                                                )
                                            doc_obj = doc_result.unwrap()
                                            # doc_obj["PageNumber"] = pages
                                            await asyto_mongo([doc_obj], MONGO_DOCUMENT_COLLECTION)
                                            
                                            # Queue for base/endorsement processing
                                            new_msg = ("base_endo_doc", split_msg)
                                            await base_endo_queue.put(new_msg)
                                            
                                            # Log document queuing
                                            queue_log = {
                                                "document_id": split_doc_id,
                                                "processor_type": "merged",
                                                "timestamp": datetime.now(),
                                                "step": "document_queuing",
                                                "message": f"Queued {folder_name} document for processing",
                                                "metadata": {
                                                    "queue_type": "base_endo",
                                                    "document_type": folder_name
                                                }
                                            }
                                            await asyto_mongo([queue_log], "DataIngestionInfoLog")

                                        elif folder_name == "Declaration":
                                            # Create declaration entry
                                            dec_result = create_declaration(
                                                dec_doc_id=split_doc_id,
                                                carriername=metadata["document_info"]["carrier_name"],
                                                sample_declaration=False
                                            )
                                            dec_obj = dec_result.unwrap()
                                            await asyto_mongo([dec_obj], MONGO_DECLARATION_COLLECTION)

                                            if metadata["processing_info"]["split_indices"].get('frontendflag') == 'Y':
                                                # For frontend uploads, use transformed name and include UserID
                                                transformed_name = transform_frontend_blob_name(split_msg[0])
                                                #user_id = split_msg[0].split('==')[1].split('.')[0] if '==' in split_msg[0] else None
                                                
                                                doc_result = create_document(
                                                    doc_id=split_doc_id,
                                                    doc_name=transformed_name,
                                                    carrier_name=metadata["document_info"]["carrier_name"],
                                                    doc_type="Declaration"
                                                )
                                                doc_obj = doc_result.unwrap()
                                                doc_obj['UserID'] = user_id
                                                #doc_obj['original_filename'] = split_msg[0]
                                            else:
                                                doc_result = create_document(
                                                    doc_id=split_doc_id,
                                                    doc_name=split_msg[0],
                                                    carrier_name=metadata["document_info"]["carrier_name"],
                                                    doc_type="Declaration"
                                                )
                                            doc_obj = doc_result.unwrap()
                                            # doc_obj["PageNumber"] = pages
                                            # # Create document entry
                                            # doc_result = create_document(
                                            #     doc_id=split_doc_id,
                                            #     doc_name=split_msg[0],
                                            #     carrier_name=metadata["document_info"]["carrier_name"],
                                            #     doc_type="Declaration"
                                            # )
                                            # doc_obj = doc_result.unwrap()
                                            await asyto_mongo([doc_obj], MONGO_DOCUMENT_COLLECTION)
                                            
                                            # Queue for declaration processing
                                            new_msg = ("dec_doc", split_msg)
                                            await declaration_queue.put(new_msg)
                                            
                                            # Log declaration processing
                                            dec_log = {
                                                "document_id": split_doc_id,
                                                "processor_type": "merged",
                                                "timestamp": datetime.now(),
                                                "step": "declaration_processing",
                                                "message": "Created and queued declaration document",
                                                "metadata": {
                                                    "carrier_name": metadata["document_info"]["carrier_name"],
                                                    "queue_type": "declaration"
                                                }
                                            }
                                            await asyto_mongo([dec_log], "DataIngestionInfoLog")

                                    except Exception as e:
                                        error_log = {
                                            "document_id": split_doc_id,
                                            "processor_type": "merged",
                                            "timestamp": datetime.now(),
                                            "error_type": "SplitDocumentProcessingError",
                                            "error_message": str(e),
                                            "traceback": ''.join(traceback.format_exception(None, e, e.__traceback__)),
                                            "step": "split_document_processing",
                                            "metadata": {
                                                "folder_name": folder_name,
                                                "carrier_name": metadata["document_info"]["carrier_name"]
                                            }
                                        }
                                        await asyto_mongo([error_log], "DataIngestionErrorLog")
                                        app_insight_logger.error(f"Error processing split document {split_doc_id}: {str(e)}", extra=properties)
                                        continue
                                else:
                                    error_log = {
                                        "document_id": split_doc_id,
                                        "processor_type": "merged",
                                        "timestamp": datetime.now(),
                                        "error_type": "SplitDataMissingError",
                                        "error_message": f"Could not find split data for document {split_doc_id} in JsonData",
                                        "step": "split_document_processing",
                                        "metadata": {
                                            "folder_name": folder_name
                                        }
                                    }
                                    await asyto_mongo([error_log], "DataIngestionErrorLog")
                                    app_insight_logger.error(f"Could not find split data for document {split_doc_id} in JsonData", extra=properties)
                                    continue

                    except Exception as e:
                        error_log = {
                            "document_id": doc_id,
                            "processor_type": "merged",
                            "timestamp": datetime.now(),
                            "error_type": "SuccessProcessingError",
                            "error_message": str(e),
                            "traceback": ''.join(traceback.format_exception(None, e, e.__traceback__)),
                            "step": "success_processing",
                            "metadata": {
                                "carrier_name": metadata["document_info"]["carrier_name"]
                            }
                        }
                        await asyto_mongo([error_log], "DataIngestionErrorLog")
                        app_insight_logger.error(f"Error processing successful document {doc_id}: {str(e)}", extra=properties)
                        continue

                except Exception as e:
                    app_insight_logger.error(f"Future task failed: {str(e)}", extra=properties)
                    error_log = {
                        "document_id": "unknown",
                        "processor_type": "merged",
                        "timestamp": datetime.now(),
                        "error_type": "FutureTaskError",
                        "error_message": str(e),
                        "traceback": ''.join(traceback.format_exception(None, e, e.__traceback__)),
                        "step": "future_task",
                        "metadata": {}
                    }
                    await asyto_mongo([error_log], "DataIngestionErrorLog")
                    continue

    except Exception as e:
        err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
        app_insight_logger.error(f"Error in process_merged_batch: {err_msg}", extra=properties)
        error_log = {
            "document_id": "batch_error",
            "processor_type": "merged",
            "timestamp": datetime.now(),
            "error_type": "BatchProcessingError",
            "error_message": str(e),
            "traceback": err_msg,
            "step": "batch_processing",
            "metadata": {}
        }
        await asyto_mongo([error_log], "DataIngestionErrorLog")
        raise

async def process_base_endo_batch(batch):
    """
    Process a batch of base/endorsement documents with comprehensive logging.
    Handles parallel processing, DB operations, and Azure Service Bus integration.
    """
    app_insight_logger.info(f"Processing base endorsement batch: {batch}", extra=properties)
    loop = asyncio.get_running_loop()
    credential = None

    async def store_logs(doc_id, logs_data):
        """Store info and error logs in their respective collections"""
        try:
            if logs_data.get("info"):
                await asyto_mongo(logs_data["info"], "DataIngestionInfoLog")
                app_insight_logger.info(f"Stored {len(logs_data['info'])} info logs for document {doc_id}", extra=properties)
                
            if logs_data.get("error"):
                await asyto_mongo(logs_data["error"], "DataIngestionErrorLog")
                app_insight_logger.info(f"Stored {len(logs_data['error'])} error logs for document {doc_id}", extra=properties)
                
        except Exception as e:
            app_insight_logger.error(f"Failed to store logs for document {doc_id}: {str(e)}", extra=properties)

    # Batch-level log function
    async def add_batch_log(log_type, step, message, error=None, metadata=None):
        """Add a log entry for batch-level operations"""
        log_entry = {
            "document_id": "batch_operation",
            "processor_type": "base_endorsement_batch",
            "timestamp": datetime.now(),
            "step": step,
            "metadata": metadata or {}
        }
        
        if log_type == "info":
            log_entry["message"] = message
            await asyto_mongo([log_entry], "DataIngestionInfoLog")
        else:  # error
            log_entry.update({
                "error_type": "BatchError",
                "error_message": message,
                "traceback": ''.join(traceback.format_exception(None, error, error.__traceback__)) if error else None
            })
            await asyto_mongo([log_entry], "DataIngestionErrorLog")

    try:
        # Initialize Azure credentials
        try:
            credential = ManagedIdentityCredential() if ENV_NAME in ENV_NAMES else DefaultAzureCredential()
            await add_batch_log("info", "azure_init", "Successfully initialized Azure credentials")
        except Exception as e:
            await add_batch_log("error", "azure_init", "Failed to initialize Azure credentials", e)
            raise

        # Initialize tracking collections
        ocr_trigger_messages = []
        endorsement_messages = []
        await add_batch_log("info", "initialization", "Initialized batch processing collections")

        async with ServiceBusClient(
            fully_qualified_namespace=OCRFULLY_QUALIFIED_NAME,
            credential=credential,
            logging_enable=True
        ) as servicebus_client:
            sender = servicebus_client.get_queue_sender(queue_name=QUEUE_NAME)
            async with sender:
                # Process items in parallel
                with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    tasks = [loop.run_in_executor(executor, process_base_endo_item, msg) for msg in batch]
                    
                    for future in asyncio.as_completed(tasks):
                        try:
                            doc_id, status, metadata, original_message, logs_data = await future
                            
                            # Store processing logs
                            await store_logs(doc_id, logs_data)

                            if not doc_id or not metadata:
                                await add_batch_log("error", "validation", "Invalid document result - missing doc_id or metadata")
                                continue

                            # First, check if this is part of a merged document
                            is_merged_doc = metadata["processing_info"]["split_indices"].get('frontendmergedupload') == 'Y'
                            parent_id = doc_id.split('_')[0] if is_merged_doc else None

                            # Handle document failures
                            if metadata["status"]["state"] == "Failed":
                                if metadata["status"]["code"] not in [101, 102, 111, 112]:
                                    try:
                                        doc = await get_document_async({"ID": doc_id}, MONGO_DOCUMENT_COLLECTION)
                                        retry_count = doc.get('RetryCount', 0) if doc else 0

                                        if retry_count < 3:
                                            # Document can be retried
                                            new_retry_count = retry_count + 1
                                            update_data = {
                                                "$set": {
                                                    "Status": "RETRYING_INGESTION",
                                                    "RetryCount": new_retry_count,
                                                    "FailureReason": f"Attempt {new_retry_count}: {metadata['status']['message']}",
                                                    "FailureStatusCode": [metadata['status']['code']],
                                                    "DocumentPath": metadata["paths"].get("pdf_path"),
                                                    "AdobeExtractPath": metadata["paths"].get("adobe_extract_path")
                                                }
                                            }

                                            await update_status_mongo_async(
                                                {"ID": doc_id},
                                                update_data,
                                                MONGO_DOCUMENT_COLLECTION
                                            )

                                            # Update merged document tracking if applicable
                                            if is_merged_doc and parent_id in merged_documents_tracking:
                                                merged_documents_tracking[parent_id]['failed_parts'][doc_id] = {
                                                    'retry_count': new_retry_count,
                                                    'latest_error': metadata['status']['message']
                                                }
                                                await check_merged_document_status(parent_id)

                                            # Log retry attempt
                                            retry_log = {
                                                "document_id": doc_id,
                                                "processor_type": "base_endorsement",
                                                "timestamp": datetime.now(),
                                                "step": "retry_processing",
                                                "message": f"Queuing document for retry attempt {new_retry_count}",
                                                "metadata": {
                                                    "retry_count": new_retry_count,
                                                    "failure_reason": metadata['status']['message'],
                                                    "status_code": metadata['status']['code'],
                                                    "is_merged_part": is_merged_doc
                                                }
                                            }
                                            await asyto_mongo([retry_log], "DataIngestionInfoLog")

                                            await base_endo_queue.put(original_message)
                                            continue
                                        else:
                                            # Final failure
                                            update_data = {
                                                "$set": {
                                                    "Status": "Failed",
                                                    "FailureReason": f"All retry attempts exhausted. Final error: {metadata['status']['message']}",
                                                    "FailureStatusCode": [metadata['status']['code']],
                                                    "DocumentPath": metadata["paths"].get("pdf_path"),
                                                    "AdobeExtractPath": metadata["paths"].get("adobe_extract_path")
                                                }
                                            }

                                            await update_status_mongo_async(
                                                {"ID": doc_id},
                                                update_data,
                                                MONGO_DOCUMENT_COLLECTION
                                            )

                                            # Update merged document tracking for final failure
                                            if is_merged_doc and parent_id in merged_documents_tracking:
                                                merged_documents_tracking[parent_id]['failed_parts'][doc_id] = {
                                                    'retry_count': retry_count,
                                                    'latest_error': metadata['status']['message']
                                                }
                                                await check_merged_document_status(parent_id)

                                            # Log final failure
                                            final_failure_log = {
                                                "document_id": doc_id,
                                                "processor_type": "base_endorsement",
                                                "timestamp": datetime.now(),
                                                "step": "final_failure",
                                                "message": "Document processing failed after all retry attempts",
                                                "metadata": {
                                                    "total_retries": retry_count,
                                                    "final_error": metadata['status']['message'],
                                                    "status_code": metadata['status']['code'],
                                                    "is_merged_part": is_merged_doc
                                                }
                                            }
                                            await asyto_mongo([final_failure_log], "DataIngestionInfoLog")
                                            continue

                                    except Exception as e:
                                        await add_batch_log("error", "retry_handling", f"Error handling document failure for {doc_id}", e)
                                        continue

                                elif metadata["status"]["code"] in [101, 102]:
                                    # Form number extraction failed but continue processing
                                    await add_batch_log("info", "form_number_handling", 
                                        f"Form number extraction failed for {doc_id} but continuing processing",
                                        metadata={"status_code": metadata['status']['code']})
                                else:
                                    # Duplicate form number failure
                                    update_data = {
                                        "$set": {
                                            "Status": "Failed",
                                            "FailureReason": metadata['status']['message'],
                                            "FailureStatusCode": [metadata['status']['code']],
                                            "DocumentPath": metadata["paths"].get("pdf_path"),
                                            "AdobeExtractPath": metadata["paths"].get("adobe_extract_path")
                                        }
                                    }
                                    
                                    await update_status_mongo_async(
                                        {"ID": doc_id},
                                        update_data,
                                        MONGO_DOCUMENT_COLLECTION
                                    )

                                    # Update merged document tracking for form number failure
                                    if is_merged_doc and parent_id in merged_documents_tracking:
                                        merged_documents_tracking[parent_id]['failed_parts'][doc_id] = {
                                            'retry_count': 0,
                                            'latest_error': f"Duplicate form number: {metadata['status']['message']}"
                                        }
                                        await check_merged_document_status(parent_id)
                                    continue

                            # Check for duplicate form numbers
                            try:
                                if metadata["document_info"].get("normalized_form_number"):
                                    exists, error = await check_form_number_exists_async(
                                        metadata["document_info"]["normalized_form_number"],
                                        MONGO_DOCUMENT_COLLECTION,
                                        metadata["document_info"]["carrier_name"],
                                        doc_id
                                    )
                                    
                                    if error:
                                        await add_batch_log("error", "form_number_check", 
                                            f"Error checking form number for {doc_id}: {error}")
                                        continue

                                    if exists:
                                        status_code = 111 if metadata["document_info"]["is_base_policy"] else 112

                                        # Cleanup for base policy documents
                                        if metadata["document_info"]["is_base_policy"]:
                                            try:
                                                cleanup_success, cleanup_error = await cleanup_document_entries_async(
                                                    doc_id,
                                                    metadata["document_info"]["folder_name"],
                                                    MONGO_POLICIES_COLLECTION
                                                )
                                                if cleanup_error:
                                                    await add_batch_log("error", "cleanup", 
                                                        f"Cleanup error for {doc_id}: {cleanup_error}")
                                            except Exception as e:
                                                await add_batch_log("error", "cleanup", 
                                                    f"Exception during cleanup for {doc_id}", e)

                                        update_data = {
                                            "$set": {
                                                "Status": "Failed",
                                                "FailureStatusCode": [status_code],
                                                "FailureReason": f"Duplicate form number: {metadata['document_info']['normalized_form_number']}",
                                                "DocumentPath": metadata["paths"].get("pdf_path"),
                                                "AdobeExtractPath": metadata["paths"].get("adobe_extract_path"),
                                                "FormNumber": metadata["document_info"].get("form_number"),
                                                "NormalizedFormNumber": metadata["document_info"].get("normalized_form_number")
                                            }
                                        }

                                        await update_status_mongo_async(
                                            {"ID": doc_id},
                                            update_data,
                                            MONGO_DOCUMENT_COLLECTION
                                        )

                                        # Update merged document tracking for duplicate form number
                                        if is_merged_doc and parent_id in merged_documents_tracking:
                                            merged_documents_tracking[parent_id]['failed_parts'][doc_id] = {
                                                'retry_count': 0,
                                                'latest_error': f"Duplicate form number: {metadata['document_info']['normalized_form_number']}"
                                            }
                                            await check_merged_document_status(parent_id)
                                        continue

                            except Exception as e:
                                await add_batch_log("error", "form_number_check", 
                                    f"Error in duplicate check for {doc_id}", e)
                                continue

                            # Process successful document
                            try:
                                # Update document collection
                                doc_update = {
                                    "$set": {
                                        "Status": "Ingested",
                                        "DocumentPath": metadata["paths"].get("pdf_path"),
                                        "AdobeExtractPath": metadata["paths"].get("adobe_extract_path"),
                                        "UploadedAt": metadata["document_info"]["uploaded_at"],
                                        "FormNumber": metadata["document_info"].get("form_number"),
                                        "NormalizedFormNumber": metadata["document_info"].get("normalized_form_number"),
                                        "JsonData": metadata["processing_info"]["split_indices"].get('split', [])
                                    }
                                }

                                # Add FailureStatusCode if it's a form extraction failure
                                if metadata["status"]["code"] in [101, 102]:
                                    doc_update["$set"].update({
                                        "FailureStatusCode": [metadata["status"]["code"]],
                                        "FailureReason": metadata["status"]["message"]
                                    })
                                else:
                                    # Clear failure fields on success
                                    doc_update["$unset"] = {
                                        "FailureStatusCode": "",
                                        "FailureReason": "",
                                        "RetryCount": ""
                                    }

                                await update_status_mongo_async(
                                    {"ID": doc_id},
                                    doc_update,
                                    MONGO_DOCUMENT_COLLECTION
                                )

                                # Update merged document tracking for successful processing
                                if is_merged_doc and parent_id in merged_documents_tracking:
                                    merged_documents_tracking[parent_id]['processed_parts'].add(doc_id)
                                    if doc_id in merged_documents_tracking[parent_id]['failed_parts']:
                                        del merged_documents_tracking[parent_id]['failed_parts'][doc_id]
                                    await check_merged_document_status(parent_id)

                                # Handle base policy updates
                                if metadata["document_info"]["is_base_policy"]:
                                    try:
                                        if not metadata["processing_info"]["frontend_merged"]:
                                            mongo_query = {"BasePolicyDocumentID": doc_id}
                                            doc_number_update = {
                                                "$set": {
                                                    "FormNumber": metadata["document_info"].get("form_number"),
                                                    "NormalizedFormNumber": metadata["document_info"].get("normalized_form_number")
                                                }
                                            }
                                            if not metadata["processing_info"]["frontend_upload"]:
                                                doc_number_update["$set"]["PolicyName"] = metadata["document_info"].get("normalized_form_number")
                                        else:
                                            mongo_query = {"MergedDocumentID": metadata["document_info"]["base_doc_id"]}
                                            doc_number_update = {
                                                "$set": {
                                                    "BasePolicyDocumentID": doc_id,
                                                    "FormNumber": metadata["document_info"].get("form_number"),
                                                    "NormalizedFormNumber": metadata["document_info"].get("normalized_form_number")
                                                }
                                            }
                                            if not metadata["processing_info"]["frontend_upload"]:
                                                doc_number_update["$set"]["PolicyName"] = metadata["document_info"].get("normalized_form_number")

                                        await update_status_mongo_async(
                                            mongo_query,
                                            doc_number_update,
                                            MONGO_POLICIES_COLLECTION
                                        )

                                        await add_batch_log("info", "policy_update", 
                                            f"Updated policy information for {doc_id}")

                                    except Exception as e:
                                        await add_batch_log("error", "policy_update", 
                                            f"Error updating base policy for {doc_id}", e)
                                        continue

                                # Log ingestion status
                                await log_ingestion_status_toDB(doc_id, status='Ingested')

                                # Collect OCR trigger message
                                ocr_message = {
                                    "eventTime": datetime.strftime(metadata["document_info"]["uploaded_at"], "%Y-%m-%dT%H:%M:%S"),
                                    "data": {"url": metadata["paths"].get("pdf_path")},
                                    "doc_id": doc_id,
                                    "adobe_blob_name": metadata["paths"].get("adobe_blob_name")
                                }
                                ocr_trigger_messages.append(json.dumps([ocr_message]))

                                # Collect endorsement message if applicable
                                if "endorsement" in original_message[1][5].lower():
                                    endorsement_messages.append(original_message)
                                    await add_batch_log("info", "endorsement_collection", 
                                        f"Collected endorsement message for {doc_id}")

                            except Exception as e:
                                await add_batch_log("error", "success_processing", 
                                    f"Error processing successful document {doc_id}", e)
                                continue

                        except Exception as e:
                            await add_batch_log("error", "document_processing", 
                                "Future task failed", e)
                            continue

                    # Process batch-level operations
                    try:
                        # Update policy collection for endorsements
                        if endorsement_messages:
                            await add_batch_log("info", "endorsement_processing", 
                                f"Processing {len(endorsement_messages)} endorsement messages")
                            try:
                                update_endo_res = await update_endorsement_document_id_async(
                                    MONGO_POLICIES_COLLECTION,
                                    endorsement_messages
                                )
                                if not update_endo_res:
                                    await add_batch_log("error", "endorsement_update", 
                                        "Failed to update endorsements in Policies collection")
                                else:
                                    await add_batch_log("info", "endorsement_update", 
                                        f"Successfully updated {len(endorsement_messages)} endorsements")
                            except Exception as e:
                                await add_batch_log("error", "endorsement_update", 
                                    "Error updating endorsements", e)

                        # Send OCR trigger messages
                        if ocr_trigger_messages:
                            await add_batch_log("info", "ocr_trigger", 
                                f"Sending {len(ocr_trigger_messages)} OCR trigger messages")
                            try:
                                service_bus_msgs = [ServiceBusMessage(msg) for msg in ocr_trigger_messages]
                                await sender.send_messages(service_bus_msgs)
                                await add_batch_log("info", "ocr_trigger", 
                                    f"Successfully sent {len(ocr_trigger_messages)} OCR trigger messages")
                            except Exception as e:
                                await add_batch_log("error", "ocr_trigger", 
                                    "Error sending OCR trigger messages", e)

                    except Exception as e:
                        await add_batch_log("error", "batch_operations", 
                            "Error in batch-level operations", e)

    except Exception as e:
        err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
        await add_batch_log("error", "critical_error", 
            "Critical error in batch processing", e, 
            {"error_details": err_msg})
        raise

    finally:
        if credential:
            await credential.close()
            await add_batch_log("info", "cleanup", 
                "Closed Azure credentials")

async def process_declarations(batch):
    """
    Process a batch of declaration documents with comprehensive logging.
    Handles Service Bus integration, frontend tracking, and complex DB operations.
    """
    app_insight_logger.info(f"Processing declaration batch: {batch}", extra=properties)
    loop = asyncio.get_running_loop()
    credential = None

    async def store_logs(doc_id, logs_data):
        """Store info and error logs in their respective collections"""
        try:
            if logs_data.get("info"):
                await asyto_mongo(logs_data["info"], "DataIngestionInfoLog")
                app_insight_logger.info(f"Stored {len(logs_data['info'])} info logs for document {doc_id}", extra=properties)
                
            if logs_data.get("error"):
                await asyto_mongo(logs_data["error"], "DataIngestionErrorLog")
                app_insight_logger.info(f"Stored {len(logs_data['error'])} error logs for document {doc_id}", extra=properties)
                
        except Exception as e:
            app_insight_logger.error(f"Failed to store logs for document {doc_id}: {str(e)}", extra=properties)

    # Batch-level log function
    async def add_batch_log(log_type, step, message, error=None, metadata=None):
        """Add a log entry for batch-level operations"""
        log_entry = {
            "document_id": "batch_operation",
            "processor_type": "declaration_batch",
            "timestamp": datetime.now(),
            "step": step,
            "metadata": metadata or {}
        }
        
        if log_type == "info":
            log_entry["message"] = message
            await asyto_mongo([log_entry], "DataIngestionInfoLog")
        else:  # error
            log_entry.update({
                "error_type": "BatchError",
                "error_message": message,
                "traceback": ''.join(traceback.format_exception(None, error, error.__traceback__)) if error else None
            })
            await asyto_mongo([log_entry], "DataIngestionErrorLog")

    try:
        # Initialize Azure credentials
        try:
            credential = ManagedIdentityCredential() if ENV_NAME in ENV_NAMES else DefaultAzureCredential()
            await add_batch_log("info", "azure_init", "Successfully initialized Azure credentials")
        except Exception as e:
            await add_batch_log("error", "azure_init", "Failed to initialize Azure credentials", e)
            raise

        async with ServiceBusClient(
            fully_qualified_namespace=OCRFULLY_QUALIFIED_NAME,
            credential=credential,
            logging_enable=True
        ) as servicebus_client:
            sender = servicebus_client.get_queue_sender(queue_name=QUEUE_NAME)
            async with sender:
                # Process items in parallel
                with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    tasks = [loop.run_in_executor(executor, process_dec_item, msg) for msg in batch]
                    
                    for future in asyncio.as_completed(tasks):
                        try:
                            doc_id, status, metadata, original_message, logs_data = await future
                            
                            # Store processing logs
                            await store_logs(doc_id, logs_data)

                            if not doc_id or not metadata:
                                await add_batch_log("error", "validation", "Invalid document result - missing doc_id or metadata")
                                continue

                            try:
                                # Handle failures that need retries (non 103 status codes)
                                if metadata["status"]["state"] == "Failed":
                                    if metadata["status"]["code"] != 103:  # Form extraction failures don't need retry
                                        try:
                                            doc = await get_document_async({"ID": doc_id}, MONGO_DOCUMENT_COLLECTION)
                                            retry_count = doc.get('RetryCount', 0) if doc else 0

                                            if retry_count < 3:
                                                # Document can be retried
                                                new_retry_count = retry_count + 1
                                                update_data = {
                                                    "$set": {
                                                        "Status": "RETRYING_INGESTION",
                                                        "RetryCount": new_retry_count,
                                                        "FailureReason": f"Attempt {new_retry_count}: {metadata['status']['message']}",
                                                        "FailureStatusCode": [metadata['status']['code']],
                                                        "DocumentPath": metadata["paths"].get("pdf_path"),
                                                        "AdobeExtractPath": metadata["paths"].get("adobe_extract_path")
                                                    }
                                                }

                                                await update_status_mongo_async(
                                                    {"ID": doc_id},
                                                    update_data,
                                                    MONGO_DOCUMENT_COLLECTION
                                                )

                                                # Log retry attempt
                                                retry_log = {
                                                    "document_id": doc_id,
                                                    "processor_type": "declaration",
                                                    "timestamp": datetime.now(),
                                                    "step": "retry_processing",
                                                    "message": f"Queuing document for retry attempt {new_retry_count}",
                                                    "metadata": {
                                                        "retry_count": new_retry_count,
                                                        "failure_reason": metadata['status']['message'],
                                                        "status_code": metadata['status']['code']
                                                    }
                                                }
                                                await asyto_mongo([retry_log], "DataIngestionInfoLog")

                                                await declaration_queue.put(original_message)
                                                continue
                                            else:
                                                # Final failure
                                                update_data = {
                                                    "$set": {
                                                        "Status": "Failed",
                                                        "FailureReason": f"All retry attempts exhausted. Final error: {metadata['status']['message']}",
                                                        "FailureStatusCode": [metadata['status']['code']],
                                                        "DocumentPath": metadata["paths"].get("pdf_path"),
                                                        "AdobeExtractPath": metadata["paths"].get("adobe_extract_path")
                                                    }
                                                }

                                                await update_status_mongo_async(
                                                    {"ID": doc_id},
                                                    update_data,
                                                    MONGO_DOCUMENT_COLLECTION
                                                )

                                                # Log final failure
                                                final_failure_log = {
                                                    "document_id": doc_id,
                                                    "processor_type": "declaration",
                                                    "timestamp": datetime.now(),
                                                    "step": "final_failure",
                                                    "message": "Document processing failed after all retry attempts",
                                                    "metadata": {
                                                        "total_retries": retry_count,
                                                        "final_error": metadata['status']['message'],
                                                        "status_code": metadata['status']['code']
                                                    }
                                                }
                                                await asyto_mongo([final_failure_log], "DataIngestionInfoLog")
                                                continue
                                        except Exception as e:
                                            await add_batch_log("error", "retry_handling", 
                                                f"Error handling document failure for {doc_id}", e)
                                            continue

                                # Handle merged document tracking
                                if metadata["processing_info"]["split_indices"].get('frontendmergedupload') == 'Y':
                                    parent_id = doc_id.split('_')[0]
                                    
                                    if parent_id in merged_documents_tracking:
                                        await add_batch_log("info", "frontend_tracking", 
                                            f"Updating frontend tracking for parent document {parent_id}")
                                            
                                        if status == 'Failed':
                                            # Track the failure
                                            if doc_id not in merged_documents_tracking[parent_id]['failed_parts']:
                                                merged_documents_tracking[parent_id]['failed_parts'][doc_id] = {
                                                    'retry_count': retry_count if 'retry_count' in locals() else 0,
                                                    'latest_error': metadata['status']['message']
                                                }
                                            else:
                                                merged_documents_tracking[parent_id]['failed_parts'][doc_id]['latest_error'] = metadata['status']['message']
                                                if 'retry_count' in locals():
                                                    merged_documents_tracking[parent_id]['failed_parts'][doc_id]['retry_count'] = retry_count
                                        else:
                                            # Success case
                                            merged_documents_tracking[parent_id]['processed_parts'].add(doc_id)
                                            if doc_id in merged_documents_tracking[parent_id]['failed_parts']:
                                                del merged_documents_tracking[parent_id]['failed_parts'][doc_id]
                                        
                                        await check_merged_document_status(parent_id)

                                # Update declaration details if policy number exists
                                if metadata["declaration_info"]["policy_number"]:
                                    try:
                                        await add_batch_log("info", "declaration_update", 
                                            f"Updating declaration details for {doc_id}")
                                            
                                        # Get next version
                                        next_version = await get_highest_declaration_version_async(
                                            metadata["declaration_info"]["policy_number"],
                                            MONGO_DECLARATION_COLLECTION,
                                            metadata["document_info"]["carrier_name"]
                                        )
                                        metadata["declaration_info"]["next_version"] = next_version

                                        # Update declaration collection
                                        declaration_update = {
                                            "$set": {
                                                "DeclarationNumber": metadata["declaration_info"]["policy_number"],
                                                "HolderName": metadata["declaration_info"]["holder_name"],
                                                "StartDate": metadata["declaration_info"]["start_date"],
                                                "ExpiryDate": metadata["declaration_info"]["end_date"],
                                                "Version": next_version
                                            }
                                        }

                                        await update_status_mongo_async(
                                            {"DeclarationDocumentID": [doc_id]},
                                            declaration_update,
                                            MONGO_DECLARATION_COLLECTION
                                        )
                                        
                                    except Exception as e:
                                        await add_batch_log("error", "declaration_update", 
                                            f"Error updating declaration details for {doc_id}", e)

                                # Process form number mapping if form numbers exist
                                if metadata["form_numbers"]["final"] and original_message[1][8].get("FormNumberExist", "Y") == "Y":
                                    try:
                                        await add_batch_log("info", "form_mapping", 
                                            f"Mapping form numbers for {doc_id}")
                                            
                                        mapping_success = await map_form_numbers_to_declaration_async(
                                            doc_id,
                                            metadata["form_numbers"]["final"],
                                            MONGO_POLICIES_COLLECTION,
                                            MONGO_DECLARATION_COLLECTION,
                                            MONGO_DOCUMENT_COLLECTION,
                                            metadata["document_info"]["carrier_name"]
                                        )
                                    except Exception as e:
                                        error_msg = str(e)
                                        status_code = [104] if any(msg in error_msg.lower() for msg in ["multiple base policy", "No matching base policy found"]) else [105]
                                        metadata["status"]["code"] = 104 if any(msg in error_msg.lower() for msg in ["multiple base policy", "No matching base policy found"]) else 105
                                        metadata["status"]["message"] = error_msg
                                        metadata["status"]["state"] = "Ingested"
                                        update_data = {
                                            "$set": {
                                                "Status": "Ingested",  # Changed to Ingested
                                                "FailureStatusCode": status_code,
                                                "FailureReason": f"Mapping failed: {error_msg}",
                                                "DocumentPath": metadata["paths"].get("pdf_path"),
                                                "AdobeExtractPath": metadata["paths"].get("adobe_extract_path")
                                            }
                                        }

                                        await update_status_mongo_async(
                                            {"ID": doc_id},
                                            update_data,
                                            MONGO_DOCUMENT_COLLECTION
                                        )
                                        
                                        await add_batch_log("error", "form_mapping", 
                                            f"Form number mapping failed for {doc_id}", e, {
                                                "status_code": status_code[0],
                                                "error_msg": error_msg
                                            })
                                        #continue

                                # Handle successful processing
                                try:
                                    # Update document collection
                                    doc_update = {
                                        "$set": {
                                            "Status": metadata["status"]["state"],
                                            "DocumentPath": metadata["paths"].get("pdf_path"),
                                            "AdobeExtractPath": metadata["paths"].get("adobe_extract_path"),
                                            "UploadedAt": metadata["processing_info"]["uploaded_at"],
                                            "FormNumber": metadata["document_info"].get("form_number"),
                                            "NormalizedFormNumber": metadata["document_info"].get("normalized_form_number"),
                                            "JsonData": metadata["processing_info"]["split_indices"].get('split', [])
                                        }
                                    }

                                    # Add FailureStatusCode if it's a form extraction failure (103)
                                    if metadata["status"]["state"] == "Failed" or metadata["status"]["code"] in [103,104,105]:
                                        doc_update["$set"].update({
                                            "FailureStatusCode": [metadata["status"]["code"]],
                                            "FailureReason": metadata["status"]["message"]
                                        })
                                    else:
                                        if metadata["status"]["code"] not in [103,104,105]:
                                            # Clear failure fields on success
                                            doc_update["$unset"] = {
                                                "FailureStatusCode": "",
                                                "FailureReason": "",
                                                "RetryCount": ""
                                            }

                                    print("checking the values in doc_update",doc_update)        

                                    await update_status_mongo_async(
                                        {"ID": doc_id},
                                        doc_update,
                                        MONGO_DOCUMENT_COLLECTION
                                    )

                                    # Log ingestion status
                                    await log_ingestion_status_toDB(
                                        doc_id,
                                        status=metadata["status"]["state"]
                                    )

                                    # Send OCR trigger if needed
                                    if status.startswith('Ingested'):
                                        ocr_message = {
                                            "eventTime": datetime.strftime(metadata["processing_info"]["uploaded_at"], "%Y-%m-%dT%H:%M:%S"),
                                            "data": {"url": metadata["paths"].get("pdf_path")},
                                            "doc_id": doc_id,
                                            "adobe_blob_name": metadata["paths"].get("adobe_blob_name")
                                        }
                                        message = ServiceBusMessage(json.dumps([ocr_message]))
                                        await sender.send_messages(message)
                                        
                                        await add_batch_log("info", "ocr_trigger", 
                                            f"Sent OCR trigger message for document {doc_id}")

                                except Exception as e:
                                    await add_batch_log("error", "success_processing", 
                                        f"Error processing successful document {doc_id}", e)
                                    continue

                            except Exception as e:
                                await add_batch_log("error", "document_processing", 
                                    f"Error processing document result for {doc_id}", e)
                                continue

                        except Exception as e:
                            await add_batch_log("error", "future_task", 
                                "Future task failed", e)
                            continue

    except Exception as e:
        err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
        await add_batch_log("error", "critical_error", 
            "Critical error in batch processing", e, 
            {"error_details": err_msg})
        raise

    finally:
        if credential:
            await credential.close()
            await add_batch_log("info", "cleanup", 
                "Closed Azure credentials")

async def process_adobe_retry_batch(batch):
    """
    Process a batch of Adobe retry documents, handling all DB operations with comprehensive logging.
    """
    app_insight_logger.info(f"Processing Adobe retry batch: {batch}", extra=properties)
    loop = asyncio.get_running_loop()
    
    async def store_logs(doc_id, logs_data):
        """Store info and error logs in their respective collections."""
        try:
            if logs_data.get("info"):
                await asyto_mongo(logs_data["info"], "DataIngestionInfoLog")
                app_insight_logger.info(f"Stored {len(logs_data['info'])} info logs for document {doc_id}", extra=properties)
            if logs_data.get("error"):
                await asyto_mongo(logs_data["error"], "DataIngestionErrorLog")
                app_insight_logger.info(f"Stored {len(logs_data['error'])} error logs for document {doc_id}", extra=properties)
        except Exception as e:
            app_insight_logger.error(f"Failed to store logs for document {doc_id}: {str(e)}", extra=properties)
    
    async def add_batch_log(log_type, step, message, error=None, extra_metadata=None):
        """Add a log entry for batch-level operations."""
        log_entry = {
            "document_id": "batch_operation",
            "processor_type": "adobe_retry",
            "timestamp": datetime.now(),
            "step": step,
            "metadata": extra_metadata or {}
        }
        if log_type == "info":
            log_entry["message"] = message
            await asyto_mongo([log_entry], "DataIngestionInfoLog")
        else:
            log_entry.update({
                "error_type": "BatchError",
                "error_message": message,
                "traceback": ''.join(traceback.format_exception(None, error, error.__traceback__)) if error else None
            })
            await asyto_mongo([log_entry], "DataIngestionErrorLog")
    
    try:
        processed_messages = []
        await add_batch_log("info", "initialization", "Starting Adobe retry batch processing")
        
        # Document validation and message preparation
        for msg in batch:
            try:
                msg_type, msg_data = msg
                doc_id, request_id = msg_data
                
                # Verify document exists
                doc_result = await get_document_async({"ID": doc_id}, MONGO_DOCUMENT_COLLECTION)
                if not doc_result:
                    await add_batch_log("error", "validation", f"Document {doc_id} not found in database")
                    continue
                
                processed_msg = (msg_type, (doc_result, request_id, app_insight_logger))
                processed_messages.append(processed_msg)
            except Exception as e:
                await add_batch_log("error", "message_preparation", "Error preparing message for processing", e)
                continue
        
        await add_batch_log("info", "preparation", f"Prepared {len(processed_messages)} messages for processing")
        
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            tasks = [loop.run_in_executor(executor, process_adobe_retry_item, msg) 
                     for msg in processed_messages]
            
            for future in asyncio.as_completed(tasks):
                try:
                    doc_id, status, metadata, original_message, logs_data = await future
                    # Store processing logs
                    await store_logs(doc_id, logs_data)
                    
                    if not doc_id or not metadata:
                        await add_batch_log("error", "validation", "Invalid document result - missing doc_id or metadata")
                        continue
                    
                    # Handle document failure
                    if metadata["status"]["state"] == "Failed":
                        try:
                            doc = await get_document_async({"ID": doc_id}, MONGO_DOCUMENT_COLLECTION)
                            retry_count = doc.get('RetryCount', 0) if doc else 0
                            
                            if retry_count < 3:
                                new_retry_count = retry_count + 1
                                update_data = {
                                    "$set": {
                                        "Status": "RETRYING_ADOBE",
                                        "RetryCount": new_retry_count,
                                        "FailureReason": f"Adobe retry attempt {new_retry_count}: {metadata['status']['message']}",
                                        "FailureStatusCode": [metadata['status']['code']],
                                        "DocumentPath": metadata["paths"].get("pdf_path"),
                                        "AdobeExtractPath": metadata["paths"].get("adobe_extract_path")
                                    }
                                }
                                
                                await update_status_mongo_async({"ID": doc_id}, update_data, MONGO_DOCUMENT_COLLECTION)
                                await adobe_retry_queue.put((msg_type, (doc_id, metadata.get("request_id"))))
                                await add_batch_log("info", "retry_handling", 
                                    f"Requeued document {doc_id} for Adobe retry attempt {new_retry_count}",
                                    extra_metadata={"retry_count": new_retry_count})
                            else:
                                update_data = {
                                    "$set": {
                                        "Status": "Failed",
                                        "FailureReason": f"All Adobe retry attempts exhausted. Final error: {metadata['status']['message']}",
                                        "FailureStatusCode": [metadata['status']['code']],
                                        "DocumentPath": metadata["paths"].get("pdf_path"),
                                        "AdobeExtractPath": metadata["paths"].get("adobe_extract_path")
                                    }
                                }
                                
                                # For non-frontend merged documents, create a new document entry
                                if metadata["document_info"]["is_non_frontend_merged"]:
                                    try:
                                        doc_result = create_document(
                                            doc_id=doc_id,
                                            doc_name=metadata["document_info"]["blob_name"],
                                            carrier_name=metadata["document_info"]["carrier_name"],
                                            doc_type=metadata["document_info"]["doc_type"]
                                        )
                                        doc_obj = doc_result.unwrap()
                                        mongo_result = await asyto_mongo([doc_obj], MONGO_DOCUMENT_COLLECTION)
                                        mongo_result.unwrap()
                                    except Exception as e:
                                        await add_batch_log("error", "document_creation", 
                                            f"Failed to create document entry for non-frontend merged doc {doc_id}: {str(e)}", e)
                                
                                await update_status_mongo_async({"ID": doc_id}, update_data, MONGO_DOCUMENT_COLLECTION)
                                await add_batch_log("info", "final_failure", f"Document {doc_id} failed after all Adobe retry attempts")
                        except Exception as e:
                            await add_batch_log("error", "retry_handling", f"Error handling document failure for {doc_id}", e)
                        continue
                    
                    # Handle successful processing
                    try:
                        doc_update = {
                            "$set": {
                                "Status": "Adobe_Retry_Complete",
                                "DocumentPath": metadata["paths"].get("pdf_path"),
                                "AdobeExtractPath": metadata["paths"].get("adobe_extract_path"),
                                "UploadedAt": metadata["processing_info"]["uploaded_at"]
                            },
                            "$unset": {
                                "RetryCount": "",
                                "FailureReason": "",
                                "FailureStatusCode": ""
                            }
                        }
                        
                        await update_status_mongo_async({"ID": doc_id}, doc_update, MONGO_DOCUMENT_COLLECTION)
                        
                        # Queue the original message for further processing
                        if original_message:
                            doc_type = metadata["document_info"]["doc_type"].lower()
                            
                            if "declaration" in doc_type:
                                await declaration_queue.put(original_message)
                            elif "merged" in doc_type:
                                await merged_queue.put(original_message)
                            else:
                                await base_endo_queue.put(original_message)
                            
                            await add_batch_log("info", "success_processing", 
                                f"Queued document {doc_id} for {doc_type} processing")
                    except Exception as e:
                        await add_batch_log("error", "success_processing", 
                            f"Error processing successful document {doc_id}", e)
                        continue
                    
                except Exception as e:
                    await add_batch_log("error", "future_task", "Future task failed", e)
                    continue
                    
    except Exception as e:
        err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
        await add_batch_log("error", "critical_error", "Critical error in Adobe retry batch processing", e, {"error_details": err_msg})
        raise


async def process_extraction_retry_batch(batch):
    """
    Process a batch of extraction retry requests with comprehensive logging.
    Handles DB operations and complex document status management.
    """
    app_insight_logger.info(f"Processing extraction retry batch: {batch}", extra=properties)
    loop = asyncio.get_running_loop()

    async def store_logs(doc_id, logs_data):
        """Store info and error logs in their respective collections"""
        try:
            if logs_data.get("info"):
                await asyto_mongo(logs_data["info"], "DataIngestionInfoLog")
                app_insight_logger.info(f"Stored {len(logs_data['info'])} info logs for document {doc_id}", extra=properties)
                
            if logs_data.get("error"):
                await asyto_mongo(logs_data["error"], "DataIngestionErrorLog")
                app_insight_logger.info(f"Stored {len(logs_data['error'])} error logs for document {doc_id}", extra=properties)
                
        except Exception as e:
            app_insight_logger.error(f"Failed to store logs for document {doc_id}: {str(e)}", extra=properties)

    # Batch-level log function
    async def add_batch_log(log_type, step, message, error=None, metadata=None):
        """Add a log entry for batch-level operations"""
        log_entry = {
            "document_id": "batch_operation",
            "processor_type": "extraction_retry",
            "timestamp": datetime.now(),
            "step": step,
            "metadata": metadata or {}
        }
        
        if log_type == "info":
            log_entry["message"] = message
            await asyto_mongo([log_entry], "DataIngestionInfoLog")
        else:  # error
            log_entry.update({
                "error_type": "BatchError",
                "error_message": message,
                "traceback": ''.join(traceback.format_exception(None, error, error.__traceback__)) if error else None
            })
            await asyto_mongo([log_entry], "DataIngestionErrorLog")

    try:
        # Initialize processed documents tracking
        processed_messages = []
        await add_batch_log("info", "initialization", "Starting batch processing")
        
        # Document validation and message preparation
        for msg in batch:
            try:
                msg_type, msg_data = msg
                doc_id, config, request_id = msg_data
                
                # Verify document exists
                doc_result = await get_document_async({"ID": doc_id}, MONGO_DOCUMENT_COLLECTION)
                if not doc_result:
                    await add_batch_log("error", "validation", 
                        f"Document {doc_id} not found in database")
                    continue
                
                # Create message for processing
                processed_msg = (msg_type, (doc_id, config, request_id, app_insight_logger))
                processed_messages.append(processed_msg)
                
            except Exception as e:
                await add_batch_log("error", "message_preparation", 
                    "Error preparing message for processing", e)
                continue
        
        await add_batch_log("info", "preparation", 
            f"Prepared {len(processed_messages)} messages for processing")

        # Process documents in parallel
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            tasks = [loop.run_in_executor(executor, process_extraction_retry_item, msg) 
                    for msg in processed_messages]
            
            for future in asyncio.as_completed(tasks):
                try:
                    doc_id, status, metadata, original_message, logs_data = await future
                    
                    # Store processing logs
                    await store_logs(doc_id, logs_data)

                    if not doc_id or not metadata:
                        await add_batch_log("error", "validation", 
                            "Invalid document result - missing doc_id or metadata")
                        continue

                    try:
                        # Get document info
                        doc_type = metadata["document_info"]["doc_type"]
                        is_frontend = metadata["document_info"]["is_frontend"]

                        # Handle failures
                        if status == 'Failed':
                            try:
                                # Get current retry count
                                doc = await get_document_async({"ID": doc_id}, MONGO_DOCUMENT_COLLECTION)
                                retry_count = doc.get('RetryCount', 0) if doc else 0

                                if retry_count < 3:
                                    # Document can be retried
                                    new_retry_count = retry_count + 1
                                    update_data = {
                                        "$set": {
                                            "RetryCount": new_retry_count,
                                            "FailureReason": f"Form number extraction attempt {new_retry_count}",
                                            "DocumentPath": metadata["paths"].get("pdf_path"),
                                            "AdobeExtractPath": metadata["paths"].get("adobe_extract_path")
                                        }
                                    }

                                    await update_status_mongo_async(
                                        {"ID": doc_id},
                                        update_data,
                                        MONGO_DOCUMENT_COLLECTION
                                    )

                                    await extraction_retry_queue.put(original_message)
                                    
                                    await add_batch_log("info", "retry_handling", 
                                        f"Requeued document {doc_id} for retry attempt {new_retry_count}", 
                                        metadata={"retry_count": new_retry_count})
                                else:
                                    # Final failure
                                    update_data = {
                                        "$set": {
                                            "FailureReason": f"Form number extraction failed after all attempts",
                                            "DocumentPath": metadata["paths"].get("pdf_path"),
                                            "AdobeExtractPath": metadata["paths"].get("adobe_extract_path")
                                        }
                                    }

                                    await update_status_mongo_async(
                                        {"ID": doc_id},
                                        update_data,
                                        MONGO_DOCUMENT_COLLECTION
                                    )
                                    
                                    await add_batch_log("info", "final_failure", 
                                        f"Document {doc_id} failed after all retry attempts")
                            except Exception as e:
                                await add_batch_log("error", "retry_handling", 
                                    f"Error handling document failure for {doc_id}", e)
                            continue

                        # Handle successful processing
                        try:
                            # Check for duplicate form numbers for Base/Endorsement documents
                            if doc_type != "Declaration" and metadata["extraction_info"]["form_number"]:
                                exists, error = await check_form_number_exists_async(
                                    metadata["extraction_info"]["normalized_form_number"],
                                    MONGO_DOCUMENT_COLLECTION,
                                    metadata["document_info"]["carrier_name"],
                                    doc_id
                                )
                                
                                if error:
                                    await add_batch_log("error", "form_number_check", 
                                        f"Error checking form number for {doc_id}: {error}")
                                    
                                    await update_status_mongo_async(
                                        {"ID": doc_id},
                                        {
                                            "$set": {
                                                "FailureReason": f"Form number check failed: {error}",
                                                "DocumentPath": metadata["paths"].get("pdf_path"),
                                                "AdobeExtractPath": metadata["paths"].get("adobe_extract_path")
                                            }
                                        },
                                        MONGO_DOCUMENT_COLLECTION
                                    )
                                    continue
                                    
                                if exists:
                                    # Handle base policy cleanup
                                    if "base" in doc_type.lower():
                                        cleanup_success, cleanup_error = await cleanup_document_entries_async(
                                            doc_id,
                                            'base',
                                            MONGO_POLICIES_COLLECTION
                                        )
                                        
                                        if cleanup_error:
                                            await add_batch_log("error", "cleanup", 
                                                f"Error during cleanup for {doc_id}: {cleanup_error}")
                                    
                                    # Set appropriate failure info
                                    status_code = 111 if "base" in doc_type.lower() else 112
                                    await update_status_mongo_async(
                                        {"ID": doc_id},
                                        {
                                            "$set": {
                                                "FailureReason": f"Duplicate form number: {metadata['extraction_info']['normalized_form_number']}",
                                                "FailureStatusCode": [status_code],
                                                "DocumentPath": metadata["paths"].get("pdf_path"),
                                                "AdobeExtractPath": metadata["paths"].get("adobe_extract_path"),
                                                "FormNumber": metadata["extraction_info"]["form_number"],
                                                "NormalizedFormNumber": metadata["extraction_info"]["normalized_form_number"]
                                            }
                                        },
                                        MONGO_DOCUMENT_COLLECTION
                                    )
                                    
                                    await add_batch_log("info", "duplicate_handling", 
                                        f"Found duplicate form number for {doc_id}", 
                                        metadata={"status_code": status_code})
                                    continue

                            # Process form numbers mapping for Declarations
                            if doc_type == "Declaration" and not is_frontend:
                                try:
                                    if metadata["extraction_info"]["final_forms"]:
                                        try:
                                            await add_batch_log("info", "declaration_mapping", 
                                                f"Mapping form numbers for declaration {doc_id}")
                                                
                                            mapping_success = await map_form_numbers_to_declaration_async(
                                                doc_id,
                                                metadata["extraction_info"]["final_forms"],
                                                MONGO_POLICIES_COLLECTION,
                                                MONGO_DECLARATION_COLLECTION,
                                                MONGO_DOCUMENT_COLLECTION,
                                                metadata["document_info"]["carrier_name"]
                                            )
                                        except Exception as e:
                                            error_msg = str(e)
                                            status_code = 104 if any(msg in error_msg.lower() for msg in ["multiple base policy", "no matching base policy"]) else 105
                                            
                                            await update_status_mongo_async(
                                                {"ID": doc_id},
                                                {
                                                    "$set": {
                                                        "FailureReason": f"Mapping failed: {error_msg}",
                                                        "FailureStatusCode": [status_code],
                                                        "DocumentPath": metadata["paths"].get("pdf_path"),
                                                        "AdobeExtractPath": metadata["paths"].get("adobe_extract_path")
                                                    }
                                                },
                                                MONGO_DOCUMENT_COLLECTION
                                            )
                                            
                                            await add_batch_log("error", "declaration_mapping", 
                                                f"Form number mapping failed for {doc_id}", e, 
                                                {"status_code": status_code})
                                            continue
                                except Exception as e:
                                    await add_batch_log("error", "declaration_processing", 
                                        f"Error processing declaration mapping for {doc_id}", e)
                                    continue

                            # Update document with extraction results
                            doc_update = {
                                "$set": {
                                    "FormNumber": metadata["extraction_info"]["form_number"],
                                    "NormalizedFormNumber": metadata["extraction_info"]["normalized_form_number"],
                                    "DocumentPath": metadata["paths"].get("pdf_path"),
                                    "AdobeExtractPath": metadata["paths"].get("adobe_extract_path")
                                },
                                "$unset": {
                                    "RetryCount": "",
                                    "FailureReason": "",
                                    "FailureStatusCode": ""
                                }
                            }

                            await update_status_mongo_async(
                                {"ID": doc_id},
                                doc_update,
                                MONGO_DOCUMENT_COLLECTION
                            )

                            # Additional updates for Base Policy
                            if doc_type == "BasePolicy":
                                try:
                                    policy_update = {
                                        "$set": {
                                            "FormNumber": metadata["extraction_info"]["form_number"],
                                            "NormalizedFormNumber": metadata["extraction_info"]["normalized_form_number"]
                                        }
                                    }

                                    if not is_frontend:
                                        policy_update["$set"]["PolicyName"] = metadata["extraction_info"]["normalized_form_number"]

                                    await update_status_mongo_async(
                                        {"BasePolicyDocumentID": doc_id},
                                        policy_update,
                                        MONGO_POLICIES_COLLECTION
                                    )

                                    if not is_frontend:
                                        await update_status_mongo_async(
                                            {"BasePolicy": metadata["extraction_info"]["form_number"]},
                                            {
                                                "$set": {
                                                    "PolicyName": metadata["extraction_info"]["normalized_form_number"]
                                                }
                                            },
                                            MONGO_DECLARATION_COLLECTION
                                        )
                                        
                                    await add_batch_log("info", "base_policy_update", 
                                        f"Updated base policy information for {doc_id}")
                                except Exception as e:
                                    await add_batch_log("error", "base_policy_update", 
                                        f"Error updating collections for base policy {doc_id}", e)

                            await add_batch_log("info", "success_processing", 
                                f"Successfully processed extraction retry for document {doc_id}")

                        except Exception as e:
                            await add_batch_log("error", "success_processing", 
                                f"Error processing successful document {doc_id}", e)
                            continue

                    except Exception as e:
                        await add_batch_log("error", "document_processing", 
                            f"Error processing document result for {doc_id}", e)
                        continue

                except Exception as e:
                    await add_batch_log("error", "future_task", 
                        "Future task failed", e)
                    continue

    except Exception as e:
        err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
        await add_batch_log("error", "critical_error", 
            "Critical error in batch processing", e, 
            {"error_details": err_msg})
        raise

async def check_merged_document_status(parent_id):
    """
    Check if all parts of a merged document are processed and update parent status accordingly.
    """
    if parent_id not in merged_documents_tracking:
        return

    tracking_info = merged_documents_tracking[parent_id]
    total_parts = tracking_info['total_parts']
    processed_parts = len(tracking_info['processed_parts'])
    failed_parts = len(tracking_info['failed_parts'])
    
    # All parts are accounted for
    if processed_parts + failed_parts == total_parts:
        update_data = {
            "$set": {
                "ProcessedAt": datetime.now(),
            }
        }
        
        # If any parts failed
        if failed_parts > 0:
            update_data["$set"].update({
                "Status": "PartiallyProcessed",
                "FailureReason": f"{failed_parts} out of {total_parts} parts failed processing"
            })
        else:
            # All parts processed successfully
            update_data["$set"].update({
                "Status": "Processed"
            })
            update_data["$unset"] = {
                "FailureReason": "",
                "FailureStatusCode": "",
                "RetryCount": ""
            }
        
        # Update parent document status
        await update_status_mongo_async(
            {"ID": parent_id},
            update_data,
            MONGO_DOCUMENT_COLLECTION
        )
        
        # Log the completion
        completion_log = {
            "document_id": parent_id,
            "processor_type": "merged_document",
            "timestamp": datetime.now(),
            "step": "processing_complete",
            "message": f"All parts processed: {processed_parts} successful, {failed_parts} failed",
            "metadata": {
                "total_parts": total_parts,
                "processed_parts": processed_parts,
                "failed_parts": failed_parts
            }
        }
        await asyto_mongo([completion_log], "DataIngestionInfoLog")
        
        # Cleanup tracking data
        del merged_documents_tracking[parent_id]


async def process_manual_split_batch(batch):
    """
    Process a batch of manual split requests, handling all DB operations.
    """
    app_insight_logger.info(f"Processing manual split batch: {batch}", extra=properties)
    loop = asyncio.get_running_loop()

    async def store_logs(doc_id, logs_data):
        """Store info and error logs in their respective collections."""
        try:
            if logs_data.get("info"):
                await asyto_mongo(logs_data["info"], "DataIngestionInfoLog")
                app_insight_logger.info(f"Stored {len(logs_data['info'])} info logs for document {doc_id}", extra=properties)
            if logs_data.get("error"):
                await asyto_mongo(logs_data["error"], "DataIngestionErrorLog")
                app_insight_logger.info(f"Stored {len(logs_data['error'])} error logs for document {doc_id}", extra=properties)
        except Exception as e:
            app_insight_logger.error(f"Failed to store logs for document {doc_id}: {str(e)}", extra=properties)

    async def add_batch_log(log_type, step, message, error=None, extra_metadata=None):
        """Add a log entry for batch-level operations."""
        log_entry = {
            "document_id": "batch_operation",
            "processor_type": "manual_split",
            "timestamp": datetime.now(),
            "step": step,
            "metadata": extra_metadata or {}
        }
        if log_type == "info":
            log_entry["message"] = message
            await asyto_mongo([log_entry], "DataIngestionInfoLog")
        else:
            log_entry.update({
                "error_type": "BatchError",
                "error_message": message,
                "traceback": ''.join(traceback.format_exception(None, error, error.__traceback__)) if error else None
            })
            await asyto_mongo([log_entry], "DataIngestionErrorLog")

    try:
        processed_messages = []
        await add_batch_log("info", "initialization", "Starting manual split batch processing")
        # Validate each message and prepare processing messages.
        for msg in batch:
            try:
                msg_type, msg_data = msg
                doc_id, split_data, request_id = msg_data
                # Retrieve document info from DB.
                doc_result = await get_document_async({"ID": doc_id}, MONGO_DOCUMENT_COLLECTION)
                if not doc_result:
                    await add_batch_log("error", "validation", f"Document {doc_id} not found in database")
                    continue
                # Prepare the message; note that we pass along the logger.
                processed_msg = (msg_type, (doc_result, split_data, request_id, app_insight_logger))
                processed_messages.append(processed_msg)
            except Exception as e:
                await add_batch_log("error", "message_preparation", "Error preparing message for processing", e)
                continue

        await add_batch_log("info", "preparation", f"Prepared {len(processed_messages)} messages for processing")
        if not processed_messages:
            app_insight_logger.error("No valid messages to process in manual split batch.", extra=properties)
            return

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            tasks = [loop.run_in_executor(executor, process_manual_split_item, msg)
                     for msg in processed_messages]

            for future in asyncio.as_completed(tasks):
                try:
                    doc_id, status, metadata, split_messages = await future
                    # (Optionally, if you collected logs in the item processor, store them here.)
                    # For this example, we assume logs were recorded internally.
                except Exception as e:
                    await add_batch_log("error", "future_task", "Future task failed", e)
                    continue

                try:
                    if not doc_id or not metadata:
                        await add_batch_log("error", "validation", "Invalid document result - missing doc_id or metadata")
                        continue

                    # Handle failures
                    if metadata["status"]["state"] == "Failed":
                        await update_status_mongo_async(
                            {"ID": doc_id},
                            {
                                "$set": {
                                    "Status": "Failed",
                                    "FailureReason": f"Manual split failed: {metadata['status']['message']}"
                                }
                            },
                            MONGO_DOCUMENT_COLLECTION
                        )
                        app_insight_logger.error(f"Manual split failed for document {doc_id}: {metadata['status']['message']}", extra=properties)
                        continue

                    # For successful processing, update tracking info if needed
                    if not metadata["document_info"]["is_non_frontend"]:
                        merged_documents_tracking[doc_id] = {
                            'total_parts': metadata["split_info"]["total_parts"],
                            'processed_parts': set(),
                            'failed_parts': {},
                            'status': 'processing',
                            'DocumentName': metadata["document_info"]["blob_name"],
                            'CarrierName': metadata["document_info"]["carrier_name"],
                            'DocumentPath': metadata["paths"]["pdf_path"],
                            'AdobeExtractPath': metadata["paths"]["adobe_extract_path"],
                            'ProcessedAt': metadata["processing_info"]["uploaded_at"]
                        }
                        add_batch_log("info", "tracking", f"Updated tracking info for document {doc_id}")

                    # Process each new split message and queue for further processing.
                    if split_messages:
                        for msg_data in split_messages:
                            (blobname, split_doc_id, filename_with_doc_id, container_name,
                             carrier_name, folder_name, appinslogger, request_id, split_indicesdict) = msg_data
                            policy_no = str(uuid.uuid4())
                            if folder_name == "Base":
                                # Create policy entry if applicable
                                if split_indicesdict.get('frontendmergedupload', 'N') != 'Y':
                                    policy_result = create_policy(
                                        policy_number=policy_no,
                                        carriername=carrier_name,
                                        base_doc_id=split_doc_id
                                    )
                                    policy_obj = policy_result.unwrap()
                                    await asyto_mongo([policy_obj], MONGO_POLICIES_COLLECTION)
                                # Create document entry for Base Policy
                                doc_result = create_document(
                                    doc_id=split_doc_id,
                                    doc_name=blobname,
                                    carrier_name=carrier_name,
                                    doc_type="BasePolicy"
                                )
                                doc_obj = doc_result.unwrap()
                                await asyto_mongo([doc_obj], MONGO_DOCUMENT_COLLECTION)
                                await base_endo_queue.put(("base_endo_doc", msg_data))
                            elif folder_name == "Declaration":
                                # Create declaration entry
                                dec_result = create_declaration(
                                    dec_doc_id=split_doc_id,
                                    carriername=carrier_name,
                                    sample_declaration=False
                                )
                                dec_obj = dec_result.unwrap()
                                await asyto_mongo([dec_obj], MONGO_DECLARATION_COLLECTION)
                                # Create document entry for Declaration
                                doc_result = create_document(
                                    doc_id=split_doc_id,
                                    doc_name=blobname,
                                    carrier_name=carrier_name,
                                    doc_type="Declaration"
                                )
                                doc_obj = doc_result.unwrap()
                                await asyto_mongo([doc_obj], MONGO_DOCUMENT_COLLECTION)
                                await declaration_queue.put(("dec_doc", msg_data))
                            elif folder_name == "Endorsement":
                                # Create document entry for Endorsement
                                doc_result = create_document(
                                    doc_id=split_doc_id,
                                    doc_name=blobname,
                                    carrier_name=carrier_name,
                                    doc_type="Endorsement"
                                )
                                doc_obj = doc_result.unwrap()
                                await asyto_mongo([doc_obj], MONGO_DOCUMENT_COLLECTION)
                                await base_endo_queue.put(("base_endo_doc", msg_data))
                        app_insight_logger.info(f"Created {len(split_messages)} split messages for document {doc_id}", extra=properties)
                except Exception as e:
                    await add_batch_log("error", "processing_result", f"Error processing document result for {doc_id}", e)
                    continue

    except Exception as e:
        err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
        await add_batch_log("error", "critical_error", "Critical error in manual split batch processing", e, {"error_details": err_msg})
        raise            


def process_base_endo_item(msg):
    """
    Process a base policy or endorsement document message with comprehensive logging.
    Returns: (doc_id, status, metadata, original_message, logs_data)
    """
    # Initialize logging collections
    info_logs = []
    error_logs = []
    current_step = "initialization"
    
    def add_info_log(step, message, extra_metadata=None):
        """Helper function to add info logs"""
        log_metadata = {
            "carrier_name": metadata["document_info"]["carrier_name"] if metadata.get("document_info") else None,
            "document_name": blobname if 'blobname' in locals() else None,
            "document_type": "Base Policy" if metadata.get("document_info", {}).get("is_base_policy") else "Endorsement",
            "processing_path": metadata["paths"].get("pdf_path") if metadata.get("paths") else None,
            "is_split_document": metadata["document_info"].get("is_split") if metadata.get("document_info") else None
        }
        if extra_metadata:
            log_metadata.update(extra_metadata)
            
        info_logs.append({
            "document_id": metadata.get("doc_id"),
            "processor_type": "base_endorsement",
            "request_id": request_id if 'request_id' in locals() else None,
            "timestamp": datetime.now(),
            "step": step,
            "message": message,
            "metadata": log_metadata
        })

    def add_error_log(step, error, error_type, extra_metadata=None):
        """Helper function to add error logs"""
        log_metadata = {
            "carrier_name": metadata["document_info"]["carrier_name"] if metadata.get("document_info") else None,
            "document_name": blobname if 'blobname' in locals() else None,
            "document_type": "Base Policy" if metadata.get("document_info", {}).get("is_base_policy") else "Endorsement",
            "processing_path": metadata["paths"].get("pdf_path") if metadata.get("paths") else None,
            "status_code": metadata["status"].get("code") if metadata.get("status") else None,
            "is_split_document": metadata["document_info"].get("is_split") if metadata.get("document_info") else None
        }
        if extra_metadata:
            log_metadata.update(extra_metadata)
            
        error_logs.append({
            "document_id": metadata.get("doc_id"),
            "processor_type": "base_endorsement",
            "request_id": request_id if 'request_id' in locals() else None,
            "timestamp": datetime.now(),
            "error_type": error_type,
            "error_message": str(error),
            "traceback": ''.join(traceback.format_exception(None, error, error.__traceback__)),
            "step": step,
            "metadata": log_metadata
        })

    # Initialize metadata structure
    metadata = {
        "doc_id": None,
        "paths": {
            "pdf_path": None,
            "adobe_extract_path": None,
            "input_dir": None,
            "pdf_file_path": None,
            "adobe_output_dir": None,
            "adobe_blob_name": None
        },
        "status": {
            "code": None,
            "message": None,
            "state": None
        },
        "document_info": {
            "form_number": None,
            "normalized_form_number": None,
            "uploaded_at": datetime.now(),
            "carrier_name": None,
            "folder_name": None,
            "is_split": False,
            "base_doc_id": None,
            "is_base_policy": False
        },
        "processing_info": {
            "split_indices": {},
            "frontend_merged": False,
            "frontend_upload": False
        }
    }

    try:
        # Message unpacking step
        current_step = "message_unpacking"
        add_info_log(current_step, "Starting to unpack message data")
        
        msg_type, msg_data = msg
        app_insight_logger = msg_data[6]
        app_insight_logger.info(f"Processing {msg_type}: {msg_data}", extra=properties)
        
        # Extract doc_id early for error tracking
        if len(msg_data) >= 2:
            doc_id = metadata["doc_id"] = msg_data[1]
            add_info_log(current_step, "Successfully extracted document ID", {
                "doc_id": doc_id
            })
        else:
            error_msg = "Invalid message data: missing doc_id"
            add_error_log(current_step, error_msg, "MessageValidationError")
            metadata["status"].update({
                "code": None,
                "message": error_msg,
                "state": "Failed"
            })
            return None, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}

        # Unpack remaining message data
        if len(msg_data) == 9:
            (blobname, _, filename_with_doc_id, container_name, carrier_name, 
             folder_name, appisnlogger, request_id, split_indicesdict) = msg_data
        else:
            (blobname, _, filename_with_doc_id, container_name, carrier_name, 
             folder_name, appisnlogger, request_id) = msg_data
            split_indicesdict = {}
            
        add_info_log(current_step, "Successfully unpacked message data", {
            "carrier_name": carrier_name,
            "folder_name": folder_name
        })

        # Document info update step
        current_step = "document_info_setup"
        add_info_log(current_step, "Updating document information")
        
        metadata["document_info"].update({
            "carrier_name": carrier_name,
            "folder_name": folder_name,
            "is_base_policy": "base" in folder_name.lower()
        })
        metadata["processing_info"]["split_indices"] = split_indicesdict
        metadata["processing_info"]["frontend_merged"] = split_indicesdict.get('frontendmergedupload') == 'Y'
        metadata["processing_info"]["frontend_upload"] = split_indicesdict.get('frontendflag') == 'Y'

        # Document ID processing
        doc_id_parts = doc_id.split('_')
        is_split_document = len(doc_id_parts) == 2
        base_doc_id = doc_id_parts[-2] if is_split_document else doc_id
        metadata["document_info"].update({
            "is_split": is_split_document,
            "base_doc_id": base_doc_id
        })
        
        add_info_log(current_step, "Document info setup completed", {
            "is_split_document": is_split_document,
            "base_doc_id": base_doc_id,
            "is_base_policy": metadata["document_info"]["is_base_policy"]
        })

        # Carrier config step
        current_step = "carrier_config"
        add_info_log(current_step, f"Getting carrier configuration for {carrier_name}")
        
        carrier_document = pull_carrier_config(carrier_name, MONGO_CLIENT_COLLECTION, MONGO_REGEX_COLLECTION)
        if not carrier_document:
            error_msg = f"Carrier configuration not found for: {carrier_name}"
            add_error_log(current_step, error_msg, "CarrierConfigError")
            metadata["status"].update({
                "code": None,
                "message": error_msg,
                "state": "Failed"
            })
            return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}

        add_info_log(current_step, "Successfully retrieved carrier configuration")

        # Form number pattern compilation
        formnumberpatternfromconfig = carrier_document["FormNumberPattern"]
        compiledformnumberpatternfromconfig = re.compile(formnumberpatternfromconfig) if isinstance(formnumberpatternfromconfig, str) else [re.compile(pattrn) for pattrn in formnumberpatternfromconfig]

        # Path setup step
        current_step = "path_setup"
        add_info_log(current_step, "Setting up processing paths")
        
        input_dir, pdf_file_path, adobe_output_dir = get_processing_paths(
            doc_id=doc_id,
            base_doc_id=base_doc_id,
            blobname=blobname,
            is_split=is_split_document,
            split_indices_dict=split_indicesdict,
            base_dir=BASE_ENDORSEMENT_DIR  # or DECLARATION_DIR for dec processor
        )

        metadata["paths"].update({
            "input_dir": input_dir,
            "pdf_file_path": pdf_file_path,
            "adobe_output_dir": adobe_output_dir
        })
        
        add_info_log(current_step, "Path setup completed", {
            "input_dir": input_dir,
            "pdf_file_path": pdf_file_path
        })

        # PDF processing step
        current_step = "pdf_processing"
        if not is_split_document and not os.path.isfile(pdf_file_path):
            add_info_log(current_step, f"Downloading PDF from blob storage: {blobname}")
            try:
                app_insight_logger.info(f"BLOOB: {container_name}\n{carrier_name}\n{folder_name}\n{blobname}")
                download_result = download_blob(
                    blobname,  # Use original blobname for download
                    input_dir=input_dir,
                    container_name=f"{container_name}/{carrier_name}/{folder_name}"
                )
                download_result.unwrap()
                add_info_log(current_step, "PDF download completed successfully")
            except Exception as e:
                add_error_log(current_step, e, "BlobDownloadError")
                metadata["status"].update({
                    "code": None,
                    "message": f"Failed to download blob: {str(e)}",
                    "state": "Failed"
                })
                return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}

        # PDF encoding check
        if not is_split_document:
            add_info_log(current_step, "Checking PDF encoding")
            try:
                check_encoding_result = checkPDFencodingerrors(pdf_file_path)
                pdf_processing_path = check_encoding_result.unwrap()
                add_info_log(current_step, "PDF encoding check completed")
            except Exception as e:
                add_error_log(current_step, e, "PDFEncodingError")
                return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}
        else:
            pdf_processing_path = pdf_file_path
            add_info_log(current_step, "Using original path for split document")

        # PDF blob setup - handle both frontend cases
        is_frontend_merged = split_indicesdict.get('frontendmergedupload') == 'Y'
        is_frontend_upload = split_indicesdict.get('frontendflag') == 'Y'
        
        if is_frontend_merged or is_frontend_upload:
            transformed_blobname = transform_frontend_blob_name(blobname)
            pdf_blob_name = f"{transformed_blobname.split('.')[0]}_{base_doc_id}.pdf"
        else:
            pdf_blob_name = f"{blobname.split('.')[0]}_{base_doc_id}.pdf"
            
        pdf_blob_name = remove_spaces(pdf_blob_name)
        pdf_path = f'https://{BLOB_ACCOUNT_NAME}.blob.core.windows.net/{OCR_CONTAINER_NAME}/input/{pdf_blob_name}'
        metadata["paths"]["pdf_path"] = pdf_path

        if not is_split_document:
            add_info_log(current_step, "Uploading PDF to OCR container")
            try:
                upload_result = upload_blob_with_transform(
                    pdf_blob_name,
                    filepath=pdf_file_path,
                    container_name=f"{OCR_CONTAINER_NAME}/input",
                    content_type="application/pdf"
                )
                transformed_pdf_blob_name = upload_result
                add_info_log(current_step, "PDF upload completed successfully", {
                    "original_name": blobname,
                    "transformed_name": transformed_pdf_blob_name
                })
            except Exception as e:
                add_error_log(current_step, e, "BlobUploadError")
                return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}

        # Adobe API processing step
        current_step = "adobe_processing"
        if is_frontend_merged or is_frontend_upload:
            adobe_base_name = transform_frontend_blob_name(blobname).split('.')[0]
        else:
            adobe_base_name = blobname.split('.')[0]
            
        adobe_api_filename = f"adobe-api-{adobe_base_name}-{base_doc_id}.zip"
        adobe_api_zip_path = os.path.join(adobe_output_dir, adobe_api_filename)

        try:
            if not os.path.isfile(adobe_api_zip_path):
                add_info_log(current_step, "Starting Adobe API extraction")
                extract_result = run_extract_pdf(
                    filename=pdf_processing_path,
                    adobe_dir=adobe_api_zip_path,
                    logger_name=appisnlogger,
                    request_id=request_id
                )
                extract_result.unwrap()
                
                add_info_log(current_step, "Extracting Adobe API results")
                extract_zip_result = extractZipNew(zip_file_path=adobe_api_zip_path)
                extract_zip_result.unwrap()

                if not is_split_document:
                    add_info_log(current_step, "Uploading Adobe extraction results")
                    upload_result = upload_blob_with_transform(
                        adobe_api_filename,
                        filepath=adobe_api_zip_path,
                        container_name=f"{OCR_CONTAINER_NAME}/output/adobe",
                        content_type="application/zip"
                    )
                    transformed_adobe_filename = upload_result
                    add_info_log(current_step, "Adobe upload completed successfully", {
                        "original_name": adobe_api_filename,
                        "transformed_name": transformed_adobe_filename
                    })

            adobe_api_path = f'https://{BLOB_ACCOUNT_NAME}.blob.core.windows.net/{OCR_CONTAINER_NAME}/output/adobe/{adobe_api_filename}'
            metadata["paths"].update({
                "adobe_extract_path": adobe_api_path,
                "adobe_blob_name": adobe_api_filename
            })
            add_info_log(current_step, "Adobe processing completed successfully")

        except Exception as e:
            add_error_log(current_step, e, "AdobeAPIError")
            metadata["status"].update({
                "code": 109,
                "message": f"Adobe API processing failed: {str(e)}",
                "state": "Failed"
            })
            return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}

        # Form number extraction step
        current_step = "form_number_extraction"
        if split_indicesdict.get("FormNumberExist", "Y") == "Y":
            add_info_log(current_step, "Starting form number extraction")
            try:
                if is_split_document:
                    try:
                        doc_num_result = extractDocumentNumber(
                            split_indicesdict['text'],
                            compiledformnumberpatternfromconfig,
                            carrier_document,
                            pdf_file_path,
                            client, AZURE_OPENAI_CHATGPT_DEPLOYMENT

                        )
                        documentNumber, normalizeDocumentNumber = doc_num_result.unwrap()
                        add_info_log(current_step, "Successfully extracted form number from split document", {
                            "form_number": documentNumber,
                            "normalized_form_number": normalizeDocumentNumber
                        })
                    except Exception as e:
                        status_code = 101 if metadata["document_info"]["is_base_policy"] else 102
                        add_error_log(current_step, e, "FormNumberExtractionError", {
                            "status_code": status_code,
                            "is_split": True
                        })
                        metadata["status"].update({
                            "code": status_code,
                            "message": f"Form number extraction failed: {str(e)}",
                            "state": "Ingested"  # Still mark as Ingested for form number failures
                        })
                        # Continue processing with the status code set
                else:
                    try:
                        json_data_path = os.path.join(adobe_output_dir, "extractedAdobeRes", 'structuredData.json')
                        add_info_log(current_step, "Processing document for form number extraction")
                        
                        split_processor = DocumentSplitProcessor(doc_id, pdf_file_path, json_data_path, images_folder=None)
                        split_result, successfailuremessage = split_processor.run()

                        if successfailuremessage != "Success":
                            status_code = 101 if metadata["document_info"]["is_base_policy"] else 102
                            error_msg = f"Form number extraction failed: {successfailuremessage}"
                            add_error_log(current_step, error_msg, "SplitProcessingError", {
                                "status_code": status_code,
                                "is_split": False
                            })
                            metadata["status"].update({
                                "code": status_code,
                                "message": error_msg,
                                "state": "Failed"
                            })
                            return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}

                        doc_num_result = extractDocumentNumber(
                            split_result,
                            compiledformnumberpatternfromconfig,
                            carrier_document,
                            pdf_file_path,
                            client, AZURE_OPENAI_CHATGPT_DEPLOYMENT
                        )
                        documentNumber, normalizeDocumentNumber = doc_num_result.unwrap()
                        add_info_log(current_step, "Successfully extracted form number from document", {
                            "form_number": documentNumber,
                            "normalized_form_number": normalizeDocumentNumber
                        })
                    except Exception as e:
                        status_code = 101 if metadata["document_info"]["is_base_policy"] else 102
                        add_error_log(current_step, e, "FormNumberExtractionError", {
                            "status_code": status_code,
                            "is_split": False
                        })
                        metadata["status"].update({
                            "code": status_code,
                            "message": f"Form number extraction failed: {str(e)}",
                            "state": "Failed"
                        })
                        return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}

                metadata["document_info"].update({
                    "form_number": documentNumber,
                    "normalized_form_number": normalizeDocumentNumber
                })

            except Exception as e:
                status_code = 101 if metadata["document_info"]["is_base_policy"] else 102
                add_error_log(current_step, e, "FormNumberProcessingError", {
                    "status_code": status_code
                })
                metadata["status"].update({
                    "code": status_code,
                    "message": f"Error extracting document information: {str(e)}",
                    "state": "Failed"
                })
                return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}
        else:
            add_info_log(current_step, f"Skipping form number extraction due to FormNumberExist flag", {
                "FormNumberExist": "N"
            })

        # Final success status
        current_step = "completion"
        metadata["status"].update({
            "code": None,
            "message": "Successfully processed document",
            "state": "Ingested"
        })
        add_info_log(current_step, "Document processing completed successfully", {
            "final_status": "Ingested",
            "form_number": metadata["document_info"].get("form_number"),
            "normalized_form_number": metadata["document_info"].get("normalized_form_number")
        })
        return doc_id, 'Ingested', metadata, msg, {"info": info_logs, "error": error_logs}

    except Exception as e:
        add_error_log(current_step, e, "UnhandledError")
        metadata["status"].update({
            "code": None,
            "message": f"Unhandled exception in process_base_endo_item: {str(e)}",
            "state": "Failed"
        })
        return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}
    
def process_dec_item(msg):
    """
    Process a declaration document with comprehensive logging.
    Returns: (doc_id, status, metadata, original_message, logs_data)
    """
    # Initialize logging collections
    info_logs = []
    error_logs = []
    current_step = "initialization"
    
    def add_info_log(step, message, extra_metadata=None):
        """Helper function to add info logs"""
        log_metadata = {
            "carrier_name": metadata["document_info"]["carrier_name"] if metadata.get("document_info") else None,
            "document_name": blobname if 'blobname' in locals() else None,
            "document_type": "Declaration",
            "processing_path": metadata["paths"].get("pdf_path") if metadata.get("paths") else None,
            "is_split_document": metadata["document_info"].get("is_split") if metadata.get("document_info") else None,
            "manual_update_required": metadata["status"].get("manual_update_required", False) if metadata.get("status") else False
        }
        if extra_metadata:
            log_metadata.update(extra_metadata)
            
        info_logs.append({
            "document_id": metadata.get("doc_id"),
            "processor_type": "declaration",
            "request_id": request_id if 'request_id' in locals() else None,
            "timestamp": datetime.now(),
            "step": step,
            "message": message,
            "metadata": log_metadata
        })

    def add_error_log(step, error, error_type, extra_metadata=None):
        """Helper function to add error logs"""
        log_metadata = {
            "carrier_name": metadata["document_info"]["carrier_name"] if metadata.get("document_info") else None,
            "document_name": blobname if 'blobname' in locals() else None,
            "document_type": "Declaration",
            "processing_path": metadata["paths"].get("pdf_path") if metadata.get("paths") else None,
            "status_code": metadata["status"].get("code") if metadata.get("status") else None,
            "is_split_document": metadata["document_info"].get("is_split") if metadata.get("document_info") else None,
            "manual_update_required": metadata["status"].get("manual_update_required", False) if metadata.get("status") else False
        }
        if extra_metadata:
            log_metadata.update(extra_metadata)
            
        error_logs.append({
            "document_id": metadata.get("doc_id"),
            "processor_type": "declaration",
            "request_id": request_id if 'request_id' in locals() else None,
            "timestamp": datetime.now(),
            "error_type": error_type,
            "error_message": str(error),
            "traceback": ''.join(traceback.format_exception(None, error, error.__traceback__)),
            "step": step,
            "metadata": log_metadata
        })

    # Initialize metadata structure
    metadata = {
        "doc_id": None,
        "paths": {
            "pdf_path": None,
            "adobe_extract_path": None,
            "adobe_blob_name": None,
            "input_dir": None,
            "pdf_file_path": None,
            "adobe_output_dir": None
        },
        "status": {
            "code": None,
            "message": None,
            "state": None,
            "manual_update_required": False
        },
        "document_info": {
            "form_number": None,
            "normalized_form_number": None,
            "carrier_name": None,
            "folder_name": None,
            "is_split": False,
            "base_doc_id": None
        },
        "form_numbers": {
            "original": [],
            "normalized": [],
            "excluded": [],
            "final": []
        },
        "declaration_info": {
            "policy_number": None,
            "holder_name": None,
            "start_date": None,
            "end_date": None,
            "next_version": None
        },
        "processing_info": {
            "uploaded_at": datetime.now(),
            "split_indices": {}
        }
    }

    try:
        # Message unpacking step
        current_step = "message_unpacking"
        add_info_log(current_step, "Starting to unpack message data")
        
        msg_type, msg_data = msg
        app_insight_logger = msg_data[6]
        app_insight_logger.info(f"Processing {msg_type}: {msg_data}")

        # Extract doc_id early
        if len(msg_data) >= 2:
            doc_id = metadata["doc_id"] = msg_data[1]
            add_info_log(current_step, "Successfully extracted document ID", {
                "doc_id": doc_id
            })
        else:
            error_msg = "Invalid message data: missing doc_id"
            add_error_log(current_step, error_msg, "MessageValidationError")
            metadata["status"].update({
                "state": "Failed",
                "message": error_msg
            })
            return None, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}

        # Unpack remaining message data
        if len(msg_data) == 9:
            (blobname, _, filename_with_doc_id, container_name, carrier_name,
             folder_name, appisnlogger, request_id, split_indicesdict) = msg_data
        else:
            (blobname, _, filename_with_doc_id, container_name, carrier_name,
             folder_name, appisnlogger, request_id) = msg_data
            split_indicesdict = {}
            
        add_info_log(current_step, "Successfully unpacked message data", {
            "carrier_name": carrier_name,
            "folder_name": folder_name
        })

        # Document info update
        current_step = "document_info_setup"
        add_info_log(current_step, "Updating document information")
        
        metadata["document_info"].update({
            "carrier_name": carrier_name,
            "folder_name": folder_name
        })
        metadata["processing_info"]["split_indices"] = split_indicesdict

        # Handle doc_id splitting
        doc_id_parts = doc_id.split('_')
        is_split_document = len(doc_id_parts) == 2
        base_doc_id = doc_id_parts[-2] if is_split_document else doc_id
        metadata["document_info"].update({
            "is_split": is_split_document,
            "base_doc_id": base_doc_id
        })
        
        add_info_log(current_step, "Document info setup completed", {
            "is_split_document": is_split_document,
            "base_doc_id": base_doc_id
        })

        # Carrier config step
        current_step = "carrier_config"
        add_info_log(current_step, f"Getting carrier configuration for {carrier_name}")
        
        carrier_document = pull_carrier_config(carrier_name, MONGO_CLIENT_COLLECTION, MONGO_REGEX_COLLECTION)
        if not carrier_document:
            error_msg = f"Carrier configuration not found for: {carrier_name}"
            add_error_log(current_step, error_msg, "CarrierConfigError")
            metadata["status"].update({
                "state": "Failed",
                "message": error_msg
            })
            return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}

        add_info_log(current_step, "Successfully retrieved carrier configuration")

        # Form number pattern compilation
        formnumberpatternfromconfig = carrier_document["FormNumberPattern"]
        compiledformnumberpatternfromconfig = re.compile(formnumberpatternfromconfig) if isinstance(formnumberpatternfromconfig, str) else [re.compile(pattrn) for pattrn in formnumberpatternfromconfig]

        # Path setup step
        current_step = "path_setup"
        add_info_log(current_step, "Setting up processing paths")
        
        input_dir, pdf_file_path, adobe_output_dir = get_processing_paths(
            doc_id=doc_id,
            base_doc_id=base_doc_id,
            blobname=blobname,
            is_split=is_split_document,
            split_indices_dict=split_indicesdict,
            base_dir=BASE_ENDORSEMENT_DIR  # or DECLARATION_DIR for dec processor
        )

        metadata["paths"].update({
            "input_dir": input_dir,
            "pdf_file_path": pdf_file_path,
            "adobe_output_dir": adobe_output_dir
        })
        
        add_info_log(current_step, "Path setup completed", {
            "input_dir": input_dir,
            "pdf_file_path": pdf_file_path
        })

        # PDF processing step
        current_step = "pdf_processing"
        if not is_split_document and not os.path.isfile(pdf_file_path):
            add_info_log(current_step, f"Downloading PDF from blob storage: {blobname}")
            try:
                download_result = download_blob(
                    blobname,  # Use original blobname for download
                    input_dir=input_dir,
                    container_name=f"{container_name}/{carrier_name}/{folder_name}"
                )
                download_result.unwrap()
                add_info_log(current_step, "PDF download completed successfully")
            except Exception as e:
                add_error_log(current_step, e, "BlobDownloadError")
                metadata["status"].update({
                    "code": None,
                    "message": f"Failed to download blob: {str(e)}",
                    "state": "Failed"
                })
                return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}

        # PDF encoding check
        if not is_split_document:
            add_info_log(current_step, "Checking PDF encoding")
            try:
                check_encoding_result = checkPDFencodingerrors(pdf_file_path)
                pdf_processing_path = check_encoding_result.unwrap()
                add_info_log(current_step, "PDF encoding check completed")
            except Exception as e:
                add_error_log(current_step, e, "PDFEncodingError")
                return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}
        else:
            pdf_processing_path = pdf_file_path
            add_info_log(current_step, "Using original path for split document")

        # PDF blob setup - handle both frontend cases
        is_frontend_merged = split_indicesdict.get('frontendmergedupload') == 'Y'
        is_frontend_upload = split_indicesdict.get('frontendflag') == 'Y'
        
        if is_frontend_merged or is_frontend_upload:
            transformed_blobname = transform_frontend_blob_name(blobname)
            pdf_blob_name = f"{transformed_blobname.split('.')[0]}_{base_doc_id}.pdf"
        else:
            pdf_blob_name = f"{blobname.split('.')[0]}_{base_doc_id}.pdf"
            
        pdf_blob_name = remove_spaces(pdf_blob_name)
        pdf_path = f'https://{BLOB_ACCOUNT_NAME}.blob.core.windows.net/{OCR_CONTAINER_NAME}/input/{pdf_blob_name}'
        metadata["paths"]["pdf_path"] = pdf_path

        if not is_split_document:
            add_info_log(current_step, "Uploading PDF to OCR container")
            try:
                upload_result = upload_blob_with_transform(
                    pdf_blob_name,
                    filepath=pdf_file_path,
                    container_name=f"{OCR_CONTAINER_NAME}/input",
                    content_type="application/pdf"
                )
                transformed_pdf_blob_name = upload_result
                add_info_log(current_step, "PDF upload completed successfully", {
                    "original_name": blobname,
                    "transformed_name": transformed_pdf_blob_name
                })
            except Exception as e:
                add_error_log(current_step, e, "BlobUploadError")
                return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}

        # Adobe API processing step
        current_step = "adobe_processing"
        if is_frontend_merged or is_frontend_upload:
            adobe_base_name = transform_frontend_blob_name(blobname).split('.')[0]
        else:
            adobe_base_name = blobname.split('.')[0]
            
        adobe_api_filename = f"adobe-api-{adobe_base_name}-{base_doc_id}.zip"
        adobe_api_zip_path = os.path.join(adobe_output_dir, adobe_api_filename)

        try:
            if not os.path.isfile(adobe_api_zip_path):
                add_info_log(current_step, "Starting Adobe API extraction")
                extract_result = run_extract_pdf(
                    filename=pdf_processing_path,
                    adobe_dir=adobe_api_zip_path,
                    logger_name=appisnlogger,
                    request_id=request_id
                )
                extract_result.unwrap()
                
                add_info_log(current_step, "Extracting Adobe API results")
                extract_zip_result = extractZipNew(zip_file_path=adobe_api_zip_path)
                extract_zip_result.unwrap()

                if not is_split_document:
                    add_info_log(current_step, "Uploading Adobe extraction results")
                    upload_result = upload_blob_with_transform(
                        adobe_api_filename,
                        filepath=adobe_api_zip_path,
                        container_name=f"{OCR_CONTAINER_NAME}/output/adobe",
                        content_type="application/zip"
                    )
                    transformed_adobe_filename = upload_result
                    add_info_log(current_step, "Adobe upload completed successfully", {
                        "original_name": adobe_api_filename,
                        "transformed_name": transformed_adobe_filename
                    })

            adobe_api_path = f'https://{BLOB_ACCOUNT_NAME}.blob.core.windows.net/{OCR_CONTAINER_NAME}/output/adobe/{adobe_api_filename}'
            metadata["paths"].update({
                "adobe_extract_path": adobe_api_path,
                "adobe_blob_name": adobe_api_filename
            })
            add_info_log(current_step, "Adobe processing completed successfully")

        except Exception as e:
            add_error_log(current_step, e, "AdobeAPIError")
            metadata["status"].update({
                "code": 109,
                "message": f"Adobe API processing failed: {str(e)}",
                "state": "Failed"
            })
            return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}

        # Form number processing step
        current_step = "form_number_processing"



        pages = split_indicesdict['page_num'] if is_split_document else None
        extractPolicyDetailsres = visionpolicydetails(
            pdf_file_path, pages, client, AZURE_OPENAI_CHATGPT_DEPLOYMENT
        )

        metadata["declaration_info"].update({
            "policy_number": extractPolicyDetailsres.get('declaration_number'),
            "holder_name": extractPolicyDetailsres.get('policy_holder_name'),
            "start_date": convert_date_format(extractPolicyDetailsres.get('policy_start_date')),
            "end_date": convert_date_format(extractPolicyDetailsres.get('policy_end_date'))
        })



        if split_indicesdict.get("FormNumberExist", "Y") == "Y":
            add_info_log(current_step, "Starting form number processing")
            json_data_path = os.path.join(adobe_output_dir, "extractedAdobeRes", 'structuredData.json')

            try:
                with open(json_data_path, 'r') as json_file:
                    json_data = json.load(json_file)
                add_info_log(current_step, "Successfully loaded JSON data")

                # Extract document number
                try:
                    if is_split_document:
                        add_info_log(current_step, "Extracting form number from split document")
                        doc_num_result = extractDocumentNumber(
                            split_indicesdict['text'],
                            compiledformnumberpatternfromconfig,
                            carrier_document,
                            pdf_file_path,
                            client, AZURE_OPENAI_CHATGPT_DEPLOYMENT
                        )
                        app_insight_logger.info(f"SPLIT INDICES DICT: {doc_num_result.unwrap()}")
                    else:
                        add_info_log(current_step, "Processing non-split document for form number extraction")
                        split_processor = DocumentSplitProcessor(
                            doc_id, pdf_file_path, json_data_path, images_folder=None)
                        split_result, successfailuremessage = split_processor.run()
                        splitoutput = split_result

                        doc_num_result = extractDocumentNumber(
                            splitoutput,
                            compiledformnumberpatternfromconfig,
                            carrier_document,
                            pdf_file_path,
                            client, AZURE_OPENAI_CHATGPT_DEPLOYMENT
                        )
                        app_insight_logger.info(f"SPLIT OUT DICT: {doc_num_result.unwrap()}")
                    
                    documentNumber, normalizeDocumentNumber = doc_num_result.unwrap()
                    metadata["document_info"].update({
                        "form_number": documentNumber,
                        "normalized_form_number": normalizeDocumentNumber
                    })
                    add_info_log(current_step, "Successfully extracted document number", {
                        "form_number": documentNumber,
                        "normalized_form_number": normalizeDocumentNumber
                    })
                except Exception as e:
                    add_error_log(current_step, e, "FormNumberExtractionError")
                    app_insight_logger.info(f"Document number extraction failed: {str(e)}", extra=properties)
                    # Non-critical for declarations

                # Form ordering processing
                try:
                    add_info_log(current_step, "Processing form ordering")
                    form_number, successerror_message = process_ordering(
                        adobe_api_zip_path,
                        config=carrier_document
                    )
                    
                    if successerror_message == "Success":
                        normalized_form_number = [normalize_pattern(i) for i in form_number]
                        metadata["form_numbers"]["original"] = form_number.copy()
                        metadata["form_numbers"]["normalized"] = normalized_form_number.copy()

                        add_info_log(current_step, "Initial form number collection completed", {
                            "form_count": len(form_number),
                            "original_forms": form_number,
                            "normalized_forms": normalized_form_number
                        })

                        # Remove document's own form number if it exists
                        if metadata["document_info"]["form_number"]:
                            matched_indexes = match_document_numbers(metadata["document_info"]["form_number"], form_number)
                            print("Matched Indexes: ",matched_indexes)
                            # Sort indexes in reverse order to avoid shifting issues
                            matched_indexes.sort(reverse=True)
                            for index in matched_indexes:
                                del form_number[index]
                                del normalized_form_number[index]
                            
                            add_info_log(current_step, "Removed document's own form number", {
                                "removed_indexes": matched_indexes,
                                "remaining_count": len(form_number),
                                "matched_form_number": metadata["document_info"]["normalized_form_number"]
                            })

                        # Handle excluded forms
                        if form_number and 'ExcludedFormNumbers' in carrier_document:
                            excluded_forms = [normalize_pattern(form) for form in carrier_document['ExcludedFormNumbers']]
                            final_forms = []
                            final_normalized = []
                            excluded_count = 0
                            
                            for idx, (form_num, norm_form) in enumerate(zip(form_number, normalized_form_number)):
                                if any(excl_form in norm_form for excl_form in excluded_forms):
                                    metadata["form_numbers"]["excluded"].append(form_num)
                                    excluded_count += 1
                                else:
                                    final_forms.append(form_num)
                                    final_normalized.append(norm_form)
                            
                            add_info_log(current_step, "Processed excluded forms", {
                                "excluded_count": excluded_count,
                                "remaining_count": len(final_forms)
                            })
                            
                            metadata["form_numbers"]["final"] = [
                                {"FormNumber": fn, "NormalizedFormNumber": nfn}
                                for fn, nfn in zip(final_forms, final_normalized)
                            ]
                        else:
                            # Use remaining form numbers after own form number removal
                            metadata["form_numbers"]["final"] = [
                                {"FormNumber": fn, "NormalizedFormNumber": nfn}
                                for fn, nfn in zip(form_number, normalized_form_number)
                            ]
                            
                        add_info_log(current_step, "Completed form number processing", {
                            "final_form_count": len(metadata["form_numbers"]["final"]),
                            "excluded_count": len(metadata["form_numbers"]["excluded"])
                        })
                    else:
                        print(f"FORM NUMBER EXTRACTION FAILURE Previous {successerror_message}")
                        add_error_log(current_step, successerror_message, "FormOrderingError")
                        app_insight_logger.info(f"FORM NUMBER EXTRACTION FAILURE Previous {successerror_message}", extra=properties)
                        metadata["status"].update({
                            "code": 103,
                            "message": f"Form number extraction failed: {successerror_message}",
                            "state": "Ingested"  # Form extraction failure is non-critical
                        })

                except Exception as e:
                    add_error_log(current_step, e, "FormNumberProcessingError")
                    err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
                    print(f"FORM NUMBER EXTRACTION FAILURE {err_msg}")
                    app_insight_logger.info(f"FORM NUMBER EXTRACTION FAILURE {err_msg}", extra=properties)
                    metadata["status"].update({
                        "code": 103,
                        "message": f"Form number extraction failed: {err_msg}",
                        "state": "Ingested"  # Form extraction failure is non-critical
                    })

                # Policy details extraction
                current_step = "policy_details_extraction"
                try:
                    add_info_log(current_step, "Starting policy details extraction")
                    # pages = split_indicesdict['page_num'] if is_split_document else None
                    # extractPolicyDetailsres = visionpolicydetails(
                    #     pdf_file_path, pages, client, AZURE_OPENAI_CHATGPT_DEPLOYMENT
                    # )

                    # metadata["declaration_info"].update({
                    #     "policy_number": extractPolicyDetailsres.get('declaration_number'),
                    #     "holder_name": extractPolicyDetailsres.get('policy_holder_name'),
                    #     "start_date": convert_date_format(extractPolicyDetailsres.get('policy_start_date')),
                    #     "end_date": convert_date_format(extractPolicyDetailsres.get('policy_end_date'))
                    # })
                    
                    add_info_log(current_step, "Successfully extracted policy details", {
                        "policy_number": metadata["declaration_info"]["policy_number"],
                        "holder_name": metadata["declaration_info"]["holder_name"],
                        "start_date": metadata["declaration_info"]["start_date"],
                        "end_date": metadata["declaration_info"]["end_date"]
                    })

                except Exception as e:
                    add_error_log(current_step, e, "PolicyDetailsExtractionError")
                    app_insight_logger.warning(f"Policy details extraction failed: {e}", extra=properties)
                    metadata["status"]["manual_update_required"] = True
                    add_info_log(current_step, "Policy details extraction failed, marking for manual update")

            except Exception as e:
                add_error_log(current_step, e, "DocumentProcessingError")
                metadata["status"].update({
                    "state": "Failed",
                    "message": f"Error processing document: {str(e)}"
                })
                return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}

        else:
            metadata["declaration_info"].update({
                        "policy_number": extractPolicyDetailsres.get('declaration_number'),
                        "holder_name": extractPolicyDetailsres.get('policy_holder_name'),
                        "start_date": convert_date_format(extractPolicyDetailsres.get('policy_start_date')),
                        "end_date": convert_date_format(extractPolicyDetailsres.get('policy_end_date'))
                    })
            add_info_log(current_step, "Extracting metadata for no form number", {
                        "policy_number": metadata["declaration_info"]["policy_number"],
                        "holder_name": metadata["declaration_info"]["holder_name"],
                        "start_date": metadata["declaration_info"]["start_date"],
                        "end_date": metadata["declaration_info"]["end_date"]
                    })
            add_info_log(current_step, "Skipping form number processing due to FormNumberExist flag but extracting metadata", {
                "FormNumberExist": "N"
            })

        # Set final status
        current_step = "completion"
        if not metadata["status"]["state"]:
            final_status = "Ingested but Manual update required" if metadata["status"]["manual_update_required"] else "Ingested"
            metadata["status"]["state"] = final_status
            
        add_info_log(current_step, "Document processing completed", {
            "final_status": metadata["status"]["state"],
            "manual_update_required": metadata["status"]["manual_update_required"],
            "form_count": len(metadata["form_numbers"]["final"]) if metadata["form_numbers"]["final"] else 0
        })

        return doc_id, metadata["status"]["state"], metadata, msg, {"info": info_logs, "error": error_logs}

    except Exception as e:
        add_error_log(current_step, e, "UnhandledError")
        err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
        metadata["status"].update({
            "state": "Failed",
            "message": f"Unhandled exception in process_dec_item: {err_msg}"
        })
        return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}
        
def process_merged_item(msg):
    """
    Process an individual merged document message with comprehensive logging.
    Returns: (doc_id, status, metadata, original_message, logs_data)
    """
    # Initialize logging collections
    info_logs = []
    error_logs = []
    current_step = "initialization"
    
    def add_info_log(step, message, extra_metadata=None):
        """Helper function to add info logs"""
        log_metadata = {
            "carrier_name": metadata["document_info"]["carrier_name"] if metadata.get("document_info") else None,
            "document_name": blobname if 'blobname' in locals() else None,
            "document_type": "Merged Document",
            "processing_path": metadata["paths"].get("pdf_path") if metadata.get("paths") else None
        }
        if extra_metadata:
            log_metadata.update(extra_metadata)
            
        info_logs.append({
            "document_id": metadata.get("doc_id"),
            "processor_type": "merged",
            "request_id": request_id if 'request_id' in locals() else None,
            "timestamp": datetime.now(),
            "step": step,
            "message": message,
            "metadata": log_metadata
        })

    def add_error_log(step, error, error_type, extra_metadata=None):
        """Helper function to add error logs"""
        log_metadata = {
            "carrier_name": metadata["document_info"]["carrier_name"] if metadata.get("document_info") else None,
            "document_name": blobname if 'blobname' in locals() else None,
            "document_type": "Merged Document",
            "processing_path": metadata["paths"].get("pdf_path") if metadata.get("paths") else None,
            "status_code": metadata["status"].get("code") if metadata.get("status") else None
        }
        if extra_metadata:
            log_metadata.update(extra_metadata)
            
        error_logs.append({
            "document_id": metadata.get("doc_id"),
            "processor_type": "merged",
            "request_id": request_id if 'request_id' in locals() else None,
            "timestamp": datetime.now(),
            "error_type": error_type,
            "error_message": str(error),
            "traceback": ''.join(traceback.format_exception(None, error, error.__traceback__)),
            "step": step,
            "metadata": log_metadata
        })

    # Initialize metadata structure
    metadata = {
        "doc_id": None,
        "paths": {
            "pdf_path": None,
            "adobe_extract_path": None,
            "adobe_blob_name": None,
            "input_dir": None,
            "pdf_file_path": None
        },
        "status": {
            "code": None,
            "message": None,
            "state": None
        },
        "document_info": {
            "carrier_name": None,
            "folder_name": None,
            "is_split": True,
            "base_doc_id": None
        },
        "processing_info": {
            "uploaded_at": datetime.now(),
            "split_indices": {},
            "ordering_info": {
                "base_index": None,
                "declaration_index": None
            },
            "split_result": None,
            "order_result": None,
            "total_parts": 0,
            "processed_parts": []
        }
    }

    try:
        # Message unpacking step
        current_step = "message_unpacking"
        add_info_log(current_step, "Starting to unpack message data")
        
        msg_type, msg_data = msg
        app_insight_logger = msg_data[6]
        app_insight_logger.info(f"Processing {msg_type}: {msg_data}", extra=properties)
        
        if len(msg_data) == 9:
            (blobname, doc_id, filename_with_doc_id, container_name, carrier_name, 
             folder_name, appisnlogger, request_id, split_indicesdict) = msg_data
        else:
            (blobname, doc_id, filename_with_doc_id, container_name, carrier_name, 
             folder_name, appisnlogger, request_id) = msg_data
            split_indicesdict = {}
            
        add_info_log(current_step, "Successfully unpacked message data", {
            "msg_type": msg_type,
            "doc_id": doc_id,
            "carrier_name": carrier_name
        })

        # Metadata update step
        current_step = "metadata_setup"
        add_info_log(current_step, "Updating metadata with basic document information")
        
        metadata["doc_id"] = doc_id
        metadata["document_info"].update({
            "carrier_name": carrier_name,
            "folder_name": folder_name
        })
        metadata["processing_info"]["split_indices"] = split_indicesdict

        # Handle doc_id splitting
        doc_id_parts = doc_id.split('_')
        base_doc_id = doc_id_parts[-2] if len(doc_id_parts) == 2 else doc_id
        metadata["document_info"]["base_doc_id"] = base_doc_id
        
        add_info_log(current_step, "Metadata setup completed", {
            "base_doc_id": base_doc_id
        })

        # Directory setup step
        current_step = "directory_setup"
        add_info_log(current_step, "Setting up processing directories")
        
        input_dir = os.path.join(MERGED_DIR, base_doc_id)
        os.makedirs(input_dir, exist_ok=True)
        pdf_file_path = os.path.join(input_dir, blobname)
        
        # PDF blob setup with transformation for frontend uploads
        is_frontend_upload = split_indicesdict.get('frontendflag') == 'Y'
        
        if is_frontend_upload:
            transformed_blobname = transform_frontend_blob_name(blobname)
            pdf_blob_name = f"{transformed_blobname.split('.')[0]}_{doc_id}.pdf"
        else:
            pdf_blob_name = f"{blobname.split('.')[0]}_{doc_id}.pdf"
            
        pdf_blob_name = remove_spaces(pdf_blob_name)
        pdf_path = f'https://{BLOB_ACCOUNT_NAME}.blob.core.windows.net/{OCR_CONTAINER_NAME}/input/{pdf_blob_name}'
        
        metadata["paths"].update({
            "input_dir": input_dir,
            "pdf_file_path": pdf_file_path,
            "pdf_path": pdf_path
        })
        
        add_info_log(current_step, "Directory setup completed", {
            "input_dir": input_dir,
            "pdf_path": pdf_path,
            "is_frontend": is_frontend_upload
        })

        # PDF download step - uses original blobname
        current_step = "pdf_download"
        if not os.path.isfile(pdf_file_path):
            add_info_log(current_step, f"Downloading PDF from blob storage: {blobname}, The container name is {container_name}/{carrier_name}/{folder_name} ")
            try:
                download_result = download_blob(
                    blobname,  # Use original blobname for download
                    input_dir=input_dir, 
                    container_name=f"{container_name}/{carrier_name}/{folder_name}"
                )
                download_result.unwrap()
                add_info_log(current_step, "PDF download completed successfully")
            except Exception as e:
                add_error_log(current_step, e, "BlobDownloadError")
                metadata["status"].update({
                    "code": None,
                    "message": f"Failed to download blob: {str(e)}",
                    "state": "Failed"
                })
                return doc_id, "Failed", metadata, msg, {"info": info_logs, "error": error_logs}
        else:
            add_info_log(current_step, "PDF file already exists, skipping download")

        # PDF upload step - uses transformed name for frontend uploads
        current_step = "pdf_upload"
        add_info_log(current_step, "Uploading PDF to OCR container")
        try:
            upload_result = upload_blob_with_transform(
                pdf_blob_name,
                filepath=pdf_file_path,
                container_name=f"{OCR_CONTAINER_NAME}/input",
                content_type="application/pdf"
            )
            transformed_pdf_blob_name = upload_result
            add_info_log(current_step, "PDF upload completed successfully", {
                "original_name": blobname,
                "transformed_name": transformed_pdf_blob_name
            })
        except Exception as e:
            add_error_log(current_step, e, "BlobUploadError")
            metadata["status"].update({
                "code": None,
                "message": f"Failed to upload PDF: {str(e)}",
                "state": "Failed"
            })
            return doc_id, "Failed", metadata, msg, {"info": info_logs, "error": error_logs}

        # PDF encoding check step
        current_step = "pdf_encoding_check"
        add_info_log(current_step, "Checking PDF encoding")
        try:
            check_encoding_result = checkPDFencodingerrors(pdf_file_path)
            pdf_processing_path = check_encoding_result.unwrap()
            add_info_log(current_step, "PDF encoding check passed", {
                "processing_path": pdf_processing_path
            })
        except Exception as e:
            add_error_log(current_step, e, "PDFEncodingError")
            metadata["status"].update({
                "code": None,
                "message": f"PDF encoding check failed: {str(e)}",
                "state": "Failed"
            })
            return doc_id, "Failed", metadata, msg, {"info": info_logs, "error": error_logs}

        # Adobe API processing step - Modified for frontend uploads
        current_step = "adobe_processing"
        if is_frontend_upload:
            adobe_base_name = transform_frontend_blob_name(blobname).split('.')[0]
        else:
            adobe_base_name = blobname.split('.')[0]
            
        adobe_output_dir = os.path.join(input_dir, f"adobe-api-{adobe_base_name}-{doc_id}")
        os.makedirs(adobe_output_dir, exist_ok=True)
        adobe_api_filename = f"adobe-api-{adobe_base_name}-{base_doc_id}.zip"
        adobe_api_zip_path = os.path.join(adobe_output_dir, adobe_api_filename)
        
        try:
            if not os.path.isfile(adobe_api_zip_path):
                add_info_log(current_step, "Starting Adobe API extraction")
                extract_result = run_extract_pdf(
                    filename=pdf_processing_path,
                    adobe_dir=adobe_api_zip_path, 
                    logger_name=appisnlogger, 
                    request_id=request_id
                )
                extract_result.unwrap()
                
                add_info_log(current_step, "Extracting Adobe API results")
                extract_zip_result = extractZipNew(zip_file_path=adobe_api_zip_path)
                extract_zip_result.unwrap()
            else:
                add_info_log(current_step, "Adobe API zip file already exists, skipping extraction")

            adobe_api_path = f'https://{BLOB_ACCOUNT_NAME}.blob.core.windows.net/{OCR_CONTAINER_NAME}/output/adobe/{adobe_api_filename}'
            metadata["paths"].update({
                "adobe_extract_path": adobe_api_path,
                "adobe_blob_name": adobe_api_filename
            })

            add_info_log(current_step, "Uploading Adobe API results")
            upload_result = upload_blob_with_transform(
                adobe_api_filename,
                filepath=adobe_api_zip_path,
                container_name=f"{OCR_CONTAINER_NAME}/output/adobe",
                content_type="application/zip"
            )
            transformed_adobe_filename = upload_result
            add_info_log(current_step, "Adobe processing completed successfully", {
                "original_name": adobe_api_filename,
                "transformed_name": transformed_adobe_filename
            })

        except Exception as e:
            add_error_log(current_step, e, "AdobeAPIError")
            metadata["status"].update({
                "code": 108,
                "message": f"Adobe API processing failed: {str(e)}",
                "state": "Failed"
            })
            return doc_id, "Failed", metadata, msg, {"info": info_logs, "error": error_logs}

        # Carrier config step
        current_step = "carrier_config"
        add_info_log(current_step, f"Getting carrier configuration for {carrier_name}")
        carrier_document = pull_carrier_config(carrier_name, MONGO_CLIENT_COLLECTION, MONGO_REGEX_COLLECTION)
        if not carrier_document:
            error_msg = f"Carrier configuration not found for: {carrier_name}"
            add_error_log(current_step, error_msg, "CarrierConfigError")
            metadata["status"].update({
                "code": None,
                "message": error_msg,
                "state": "Failed"
            })
            return doc_id, "Failed", metadata, msg, {"info": info_logs, "error": error_logs}
            
        add_info_log(current_step, "Carrier configuration retrieved successfully")

        # Form number pattern compilation
        current_step = "form_pattern_compile"
        add_info_log(current_step, "Compiling form number patterns")
        formnumerpatternfromconfig = carrier_document["FormNumberPattern"]
        compiledformnumber = re.compile(formnumerpatternfromconfig) if isinstance(formnumerpatternfromconfig, str) else [re.compile(pattrn) for pattrn in formnumerpatternfromconfig]
        add_info_log(current_step, "Form number patterns compiled successfully")

        # Document splitting step
        current_step = "document_splitting"
        try:
            add_info_log(current_step, "Starting document splitting process")
            json_data_path = os.path.join(adobe_output_dir, "extractedAdobeRes", 'structuredData.json')
            
            split_processor = DocumentSplitProcessor(doc_id, pdf_file_path, json_data_path, compiledformnumber, images_folder=None)
            split_result, split_message = split_processor.run()

            if split_message != "Success":
                raise Exception(f"Splitting failed: {split_message}")

            metadata["processing_info"]["split_result"] = split_result
            add_info_log(current_step, "Document splitting completed successfully", {
                "split_count": len(split_result) if split_result else 0
            })

            # Document ordering step
            current_step = "document_ordering"
            add_info_log(current_step, "Starting document ordering process")
            order_result, order_message = process_ordering(
                adobe_api_zip_path,
                config=carrier_document,
                split_dict=split_result
            )

            if order_message != "Success":
                raise Exception(f"Ordering failed: {order_message}")

            metadata["processing_info"]["order_result"] = order_result
            metadata["processing_info"]["ordering_info"].update({
                "base_index": order_result["OrderingKey"]["Base"],
                "declaration_index": order_result["OrderingKey"]["Declaration"][0]
            })
            
            add_info_log(current_step, "Document ordering completed successfully")

            # Final processing step
            current_step = "final_processing"
            add_info_log(current_step, "Merging declarations and preparing final split dictionary")
            
            merged_split_dict = merge_declarations(split_result, order_result['OrderingKey'])
            metadata["processing_info"]["total_parts"] = len(merged_split_dict)

            # Prepare split documents
            add_info_log(current_step, f"Processing {len(merged_split_dict)} split documents")
            new_messages = []
            for key, split_dict in merged_split_dict.items():
                split_doc_id = key
                index = int(key.split('_')[-1])
                
                # Determine folder name based on ordering
                if order_result["OrderingKey"]["Base"] == index:
                    folder_name = "Base"
                elif order_result["OrderingKey"]["Declaration"][0] == index:
                    folder_name = "Declaration"
                else:
                    folder_name = "Endorsement"

                # Update frontend flags if needed
                if isinstance(split_indicesdict, dict):
                    if split_indicesdict.get('frontendflag') == 'Y':
                        split_dict['frontendmergedupload'] = "Y"
                    split_dict['frontendflag'] = "N"
                
                new_message_data = (
                    blobname, split_doc_id, filename_with_doc_id,
                    container_name, carrier_name, folder_name,
                    appisnlogger, request_id, split_dict
                )
                new_messages.append(new_message_data)
                metadata["processing_info"]["processed_parts"].append({
                    "doc_id": split_doc_id,
                    "folder_name": folder_name
                })
                
                add_info_log(current_step, f"Prepared split document", {
                    "split_doc_id": split_doc_id,
                    "folder_name": folder_name,
                    "index": index
                })

            # Update final status
            metadata["status"].update({
                "code": None,
                "message": "Successfully processed merged document",
                "state": "Processed"
            })
            
            add_info_log(current_step, "Completed processing merged document", {
                "total_parts": len(merged_split_dict),
                "base_index": metadata["processing_info"]["ordering_info"]["base_index"],
                "declaration_index": metadata["processing_info"]["ordering_info"]["declaration_index"]
            })
            
            return doc_id, "Processed", metadata, msg, {"info": info_logs, "error": error_logs}

        except Exception as e:
            add_error_log(current_step, e, "DocumentProcessingError")
            metadata["status"].update({
                "code": 107,
                "message": str(e),
                "state": "Failed"
            })
            return doc_id, "Failed", metadata, msg, {"info": info_logs, "error": error_logs}

    except Exception as e:
        err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
        app_insight_logger.error(f"Error in process_merged_item: {err_msg}", extra=properties)
        add_error_log(current_step, e, "UnhandledError", {
            "error_details": err_msg
        })
        metadata["status"].update({
            "code": None,
            "message": f"Unhandled exception: {err_msg}",
            "state": "Failed"
        })
        return doc_id, "Failed", metadata, msg, {"info": info_logs, "error": error_logs}

def process_adobe_retry_item(msg):
    """
    Process a document that needs Adobe API retry with comprehensive logging.
    Returns: (doc_id, status, metadata, original_message, logs_data)
    """
    # Initialize logging collections
    info_logs = []
    error_logs = []
    current_step = "initialization"
    
    def add_info_log(step, message, extra_metadata=None):
        """Helper function to add info logs."""
        log_metadata = {
            "carrier_name": metadata["document_info"]["carrier_name"] if metadata.get("document_info") else None,
            "document_name": metadata["document_info"].get("blob_name"),
            "document_type": metadata["document_info"].get("doc_type"),
            "processing_path": metadata["paths"].get("pdf_path"),
            "is_split_document": metadata["document_info"].get("is_split"),
            "is_non_frontend_merged": metadata["document_info"].get("is_non_frontend_merged", False),
            "retry_completed": metadata["processing_info"].get("retry_completed", False)
        }
        if extra_metadata:
            log_metadata.update(extra_metadata)
            
        info_logs.append({
            "document_id": metadata.get("doc_id"),
            "processor_type": "adobe_retry",
            "request_id": request_id if 'request_id' in locals() else None,
            "timestamp": datetime.now(),
            "step": step,
            "message": message,
            "metadata": log_metadata
        })
    
    def add_error_log(step, error, error_type, extra_metadata=None):
        """Helper function to add error logs."""
        log_metadata = {
            "carrier_name": metadata["document_info"]["carrier_name"] if metadata.get("document_info") else None,
            "document_name": metadata["document_info"].get("blob_name"),
            "document_type": metadata["document_info"].get("doc_type"),
            "processing_path": metadata["paths"].get("pdf_path"),
            "is_split_document": metadata["document_info"].get("is_split"),
            "is_non_frontend_merged": metadata["document_info"].get("is_non_frontend_merged", False),
            "retry_completed": metadata["processing_info"].get("retry_completed", False)
        }
        if extra_metadata:
            log_metadata.update(extra_metadata)
            
        error_logs.append({
            "document_id": metadata.get("doc_id"),
            "processor_type": "adobe_retry",
            "request_id": request_id if 'request_id' in locals() else None,
            "timestamp": datetime.now(),
            "error_type": error_type,
            "error_message": str(error),
            "traceback": ''.join(traceback.format_exception(None, error, error.__traceback__)),
            "step": step,
            "metadata": log_metadata
        })
    
    # Initialize metadata structure
    metadata = {
        "doc_id": None,
        "paths": {
            "pdf_path": None,
            "adobe_extract_path": None,
            "adobe_blob_name": None,
            "input_dir": None,
            "pdf_file_path": None,
            "adobe_output_dir": None
        },
        "status": {
            "code": None,
            "message": None,
            "state": None
        },
        "document_info": {
            "carrier_name": None,
            "doc_type": None,
            "blob_name": None,
            "is_split": False,
            "base_doc_id": None,
            "is_non_frontend_merged": False
        },
        "processing_info": {
            "uploaded_at": datetime.now(),
            "retry_completed": False,
            "split_indices": {}
        }
    }
    
    try:
        # Message unpacking step
        current_step = "message_unpacking"
        msg_type, msg_data = msg
        doc_info, request_id, app_insight_logger = msg_data
        add_info_log(current_step, "Processing Adobe retry request")
        
        # Document ID validation
        if doc_info and 'ID' in doc_info:
            doc_id = metadata["doc_id"] = doc_info['ID']
            add_info_log(current_step, "Successfully extracted document ID", {"doc_id": doc_id})
        else:
            error_msg = "Invalid document info: missing ID"
            add_error_log(current_step, error_msg, "DocumentValidationError")
            metadata["status"].update({
                "state": "Failed",
                "message": error_msg
            })
            return None, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}
        
        # Document info extraction step
        current_step = "document_info"
        try:
            blobname = doc_info['DocumentName']
            carrier_name = doc_info['CarrierName']
            doc_type = doc_info['DocumentType']
            document_path = doc_info['DocumentPath']
            
            metadata["document_info"].update({
                "carrier_name": carrier_name,
                "doc_type": doc_type,
                "blob_name": blobname
            })
            
            add_info_log(current_step, "Extracted basic document information", {
                "document_path": document_path,
                "doc_type": doc_type
            })
        except KeyError as e:
            add_error_log(current_step, e, "DocumentInfoError", {"missing_field": str(e)})
            metadata["status"].update({
                "state": "Failed",
                "message": f"Missing required document information: {str(e)}"
            })
            return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}
        
       # Document type analysis step
        current_step = "document_analysis"
        is_split_document = bool(re.search(r'_\d+$', doc_id))
        base_doc_id = doc_id.split('_')[0] if is_split_document else doc_id
        
        # New check for frontend merged upload
        is_non_frontend_merged = (
            doc_info['DocumentType'] == "Merged Document" and 
            doc_info.get('UserID') is not None
        )
        
        metadata["document_info"].update({
            "is_split": is_split_document,
            "base_doc_id": base_doc_id,
            "is_non_frontend_merged": is_non_frontend_merged
        })
        
        add_info_log(current_step, "Analyzed document structure", {
            "is_split": is_split_document,
            "base_doc_id": base_doc_id,
            "is_non_frontend_merged": is_non_frontend_merged,
            "user_id": doc_info.get('UserID')  # Added for better logging
        })
        
        # Path setup step
        current_step = "path_setup"
        try:
            if is_split_document:
                input_dir = os.path.join(MERGED_DIR, base_doc_id)
                pdf_file_path = os.path.join(input_dir, blobname)
                adobe_output_dir = os.path.join(input_dir, f"adobe-api-{blobname.split('.')[0]}-{base_doc_id}")
            else:
                if doc_type == "Merged Document":
                    input_dir = os.path.join(MERGED_DIR, doc_id)
                elif doc_type == "Declaration":
                    input_dir = os.path.join(DECLARATION_DIR, doc_id)
                else:  # Base Policy or Endorsement
                    input_dir = os.path.join(BASE_ENDORSEMENT_DIR, doc_id)
                
                os.makedirs(input_dir, exist_ok=True)
                pdf_file_path = os.path.join(input_dir, blobname)
                adobe_output_dir = os.path.join(input_dir, f"adobe-api-{blobname.split('.')[0]}-{doc_id}")
                os.makedirs(adobe_output_dir, exist_ok=True)
            
            metadata["paths"].update({
                "input_dir": input_dir,
                "pdf_file_path": pdf_file_path,
                "adobe_output_dir": adobe_output_dir
            })
            
            add_info_log(current_step, "Set up processing paths", {
                "input_dir": input_dir,
                "pdf_file_path": pdf_file_path
            })
        except Exception as e:
            add_error_log(current_step, e, "PathSetupError")
            metadata["status"].update({
                "state": "Failed",
                "message": f"Failed to set up processing paths: {str(e)}"
            })
            return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}
        
        # PDF download step
        current_step = "pdf_download"
        if not os.path.isfile(pdf_file_path):
            add_info_log(current_step, "Downloading PDF file")
            try:
                container_path = f"{OCR_CONTAINER_NAME}/input"
                processed_blob_name = document_path.split('/input/')[-1]
                
                download_result = download_blob(
                    processed_blob_name,
                    input_dir=input_dir,
                    container_name=container_path,
                    save_as=blobname
                )
                download_result.unwrap()
                add_info_log(current_step, "Successfully downloaded PDF file")
            except Exception as e:
                error_msg = f"Failed to download blob {processed_blob_name}: {str(e)}"
                add_error_log(current_step, e, "PDFDownloadError")
                metadata["status"].update({
                    "state": "Failed",
                    "code": 108 if is_non_frontend_merged else 109,
                    "message": error_msg
                })
                return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}
        
        # PDF encoding check step
        current_step = "pdf_encoding"
        try:
            add_info_log(current_step, "Checking PDF encoding")
            check_encoding_result = checkPDFencodingerrors(pdf_file_path)
            pdf_processing_path = check_encoding_result.unwrap()
            add_info_log(current_step, "PDF encoding check completed")
        except Exception as e:
            add_error_log(current_step, e, "PDFEncodingError")
            metadata["status"].update({
                "state": "Failed",
                "code": 108 if is_non_frontend_merged else 109,
                "message": f"PDF encoding check failed: {str(e)}"
            })
            return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}
        
        # Adobe API processing step
        current_step = "adobe_processing"
        adobe_api_filename = f"adobe-api-{blobname.split('.')[0]}-{base_doc_id}.zip"
        adobe_api_zip_path = os.path.join(adobe_output_dir, adobe_api_filename)
        adobe_api_path = f'https://{BLOB_ACCOUNT_NAME}.blob.core.windows.net/{OCR_CONTAINER_NAME}/output/adobe/{adobe_api_filename}'
        try:
            add_info_log(current_step, "Starting Adobe API extraction")
            extract_result = run_extract_pdf(
                filename=pdf_processing_path,
                adobe_dir=adobe_api_zip_path,
                logger_name=app_insight_logger,
                request_id=request_id
            )
            extract_result.unwrap()
            
            add_info_log(current_step, "Extracting Adobe API results")
            extract_zip_result = extractZipNew(zip_file_path=adobe_api_zip_path)
            extract_zip_result.unwrap()
            
            # Update paths and status
            metadata["paths"].update({
                "adobe_extract_path": adobe_api_path,
                "adobe_blob_name": adobe_api_filename,
                "pdf_path": document_path
            })
            metadata["processing_info"]["retry_completed"] = True
            metadata["status"].update({
                "state": "Success",
                "message": "Adobe API retry completed successfully"
            })
            
            add_info_log(current_step, "Adobe processing completed successfully", {
                "adobe_path": adobe_api_path,
                "pdf_path": document_path
            })
        except Exception as e:
            error_message = f"Adobe API processing failed: {str(e)}"
            add_error_log(current_step, e, "AdobeProcessingError")
            metadata["status"].update({
                "state": "Failed",
                "code": 108 if is_non_frontend_merged else 109,
                "message": error_message
            })
            return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}
        
        # Message preparation step
        current_step = "message_preparation"
        try:
            add_info_log(current_step, "Preparing queue message")
            split_indices_dict = {'frontendflag': 'Y' if is_non_frontend_merged else 'N'}
            if doc_info.get('FormNumberExist'):
                split_indices_dict["FormNumberExist"] = "Y"
            if doc_info.get('JsonData'):
                split_indices_dict['split'] = doc_info['JsonData']
            
            container_name = f"{OCR_CONTAINER_NAME}/input"
            folder_name = {
                "Declaration": "Declaration",
                "BasePolicy": "Base",
                "Endorsement": "Endorsement",
                "Merged Document": "Merged"
            }.get(doc_type, "Merged")
            
            message_data = (
                blobname,
                doc_id,
                None,  # filename_with_doc_id not needed for Adobe retry
                container_name,
                carrier_name,
                folder_name,
                app_insight_logger,
                request_id,
                split_indices_dict
            )
            
            add_info_log(current_step, "Successfully prepared queue message", {
                "folder_name": folder_name,
                "queue_type": doc_type.lower().replace(" ", "_")
            })
            
            return doc_id, 'Success', metadata, (doc_type.lower().replace(" ", "_"), message_data), {"info": info_logs, "error": error_logs}
        except Exception as e:
            add_error_log(current_step, e, "MessagePreparationError")
            metadata["status"].update({
                "state": "Failed",
                "message": f"Queue message preparation failed: {str(e)}"
            })
            return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}
    
    except Exception as e:
        error_message = f"Unhandled exception in process_adobe_retry_item: {str(e)}"
        add_error_log(current_step, e, "UnhandledError")
        metadata["status"].update({
            "state": "Failed",
            "message": error_message
        })
        return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}


def process_extraction_retry_item(msg):
    """
    Process a document for extraction retry with comprehensive logging.
    Returns: (doc_id, status, metadata, original_message, logs_data)
    """
    # Initialize logging collections
    info_logs = []
    error_logs = []
    current_step = "initialization"
    
    def add_info_log(step, message, extra_metadata=None):
        """Helper function to add info logs"""
        log_metadata = {
            "carrier_name": metadata["document_info"]["carrier_name"] if metadata.get("document_info") else None,
            "document_name": metadata["document_info"].get("blob_name"),
            "document_type": metadata["document_info"].get("doc_type"),
            "processing_path": metadata["paths"].get("pdf_path"),
            "is_split_document": metadata["document_info"].get("is_split"),
            "is_frontend": metadata["document_info"].get("is_frontend", False)
        }
        if extra_metadata:
            log_metadata.update(extra_metadata)
            
        info_logs.append({
            "document_id": metadata.get("doc_id"),
            "processor_type": "extraction_retry",
            "request_id": request_id if 'request_id' in locals() else None,
            "timestamp": datetime.now(),
            "step": step,
            "message": message,
            "metadata": log_metadata
        })

    def add_error_log(step, error, error_type, extra_metadata=None):
        """Helper function to add error logs"""
        log_metadata = {
            "carrier_name": metadata["document_info"]["carrier_name"] if metadata.get("document_info") else None,
            "document_name": metadata["document_info"].get("blob_name"),
            "document_type": metadata["document_info"].get("doc_type"),
            "processing_path": metadata["paths"].get("pdf_path"),
            "is_split_document": metadata["document_info"].get("is_split"),
            "is_frontend": metadata["document_info"].get("is_frontend", False)
        }
        if extra_metadata:
            log_metadata.update(extra_metadata)
            
        error_logs.append({
            "document_id": metadata.get("doc_id"),
            "processor_type": "extraction_retry",
            "request_id": request_id if 'request_id' in locals() else None,
            "timestamp": datetime.now(),
            "error_type": error_type,
            "error_message": str(error),
            "traceback": ''.join(traceback.format_exception(None, error, error.__traceback__)),
            "step": step,
            "metadata": log_metadata
        })

    # Initialize metadata structure
    metadata = {
        "doc_id": None,
        "paths": {
            "pdf_path": None,
            "adobe_extract_path": None,
            "adobe_blob_name": None,
            "input_dir": None,
            "pdf_file_path": None,
            "adobe_output_dir": None
        },
        "document_info": {
            "doc_type": None,
            "carrier_name": None,
            "is_frontend": False,
            "is_split": False,
            "base_doc_id": None,
            "blob_name": None
        },
        "extraction_info": {
            "form_number": None,
            "normalized_form_number": None,
            "declaration_forms": {
                "original": [],
                "normalized": []
            },
            "excluded_forms": [],
            "final_forms": []
        },
        "processing_info": {
            "uploaded_at": datetime.now()
        }
    }

    try:
        # Message unpacking step
        current_step = "message_unpacking"
        msg_type, msg_data = msg
        doc_id, config, request_id, app_insight_logger = msg_data
        metadata["doc_id"] = doc_id
        
        add_info_log(current_step, f"Processing extraction retry request", {
            "request_id": request_id
        })

        # Document type checking step
        current_step = "document_validation"
        is_split_document = bool(re.search(r'_\d+$', doc_id))
        base_doc_id = doc_id.split('_')[0] if is_split_document else doc_id
        split_index = int(doc_id.split('_')[1]) if is_split_document else None
        
        add_info_log(current_step, "Checking document type", {
            "is_split": is_split_document,
            "base_doc_id": base_doc_id,
            "split_index": split_index
        })

        # Document info retrieval
        current_step = "document_info"
        doc_info = get_document({"ID": doc_id}, MONGO_DOCUMENT_COLLECTION).unwrap()
        if not doc_info:
            error_msg = f"Document {doc_id} not found in database"
            add_error_log(current_step, error_msg, "DocumentNotFoundError")
            return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}

        # Update document info
        blobname = doc_info['DocumentName']
        metadata["document_info"].update({
            "blob_name": blobname,
            "carrier_name": doc_info['CarrierName'],
            "doc_type": doc_info['DocumentType'],
            "is_split": is_split_document,
            "base_doc_id": base_doc_id,
            "is_frontend": doc_info.get('UserID') is not None
        })
        
        add_info_log(current_step, "Retrieved document information", {
            "doc_type": doc_info['DocumentType'],
            "carrier_name": doc_info['CarrierName']
        })

        # Form number pattern compilation
        current_step = "pattern_compilation"
        try:
            formnumberpatternfromconfig = config['FormNumberPattern']
            compiledformnumberpatternfromconfig = (
                re.compile(formnumberpatternfromconfig) 
                if isinstance(formnumberpatternfromconfig, str) 
                else [re.compile(pattrn) for pattrn in formnumberpatternfromconfig]
            )
            add_info_log(current_step, "Successfully compiled form number patterns")
        except (re.error, TypeError) as e:
            add_error_log(current_step, e, "PatternCompilationError")
            return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}

        # Path setup step
        current_step = "path_setup"
        if is_split_document:
            input_dir = os.path.join(MERGED_DIR, base_doc_id)
            pdf_file_path = os.path.join(input_dir, blobname)
            adobe_output_dir = os.path.join(input_dir, f"adobe-api-{blobname.split('.')[0]}-{base_doc_id}")
        else:
            input_dir = os.path.join(DECLARATION_DIR if metadata["document_info"]["doc_type"] == "Declaration" else BASE_ENDORSEMENT_DIR, doc_id)
            pdf_file_path = os.path.join(input_dir, blobname)
            adobe_output_dir = os.path.join(input_dir, f"adobe-api-{blobname.split('.')[0]}-{doc_id}")

        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(adobe_output_dir, exist_ok=True)

        metadata["paths"].update({
            "input_dir": input_dir,
            "pdf_file_path": pdf_file_path,
            "adobe_output_dir": adobe_output_dir
        })
        
        add_info_log(current_step, "Set up processing paths", {
            "input_dir": input_dir,
            "pdf_file_path": pdf_file_path
        })

        # PDF download step
        current_step = "pdf_download"
        if not os.path.isfile(pdf_file_path):
            add_info_log(current_step, "Downloading PDF file")
            try:
                container_path = f"{OCR_CONTAINER_NAME}/input"
                processed_blob_name = doc_info['DocumentPath'].split('/input/')[-1]
                
                download_result = download_blob(
                    processed_blob_name,
                    input_dir=input_dir,
                    container_name=container_path,
                    save_as=blobname
                )
                download_result.unwrap()
                add_info_log(current_step, "Successfully downloaded PDF file")
            except Exception as e:
                add_error_log(current_step, e, "PDFDownloadError")
                return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}

        # Adobe processing step
        current_step = "adobe_processing"
        adobe_api_filename = f"adobe-api-{blobname.split('.')[0]}-{base_doc_id}.zip"
        adobe_api_zip_path = os.path.join(adobe_output_dir, adobe_api_filename)
        json_data_path = os.path.join(adobe_output_dir, "extractedAdobeRes", 'structuredData.json')

        metadata["paths"].update({
            "adobe_blob_name": adobe_api_filename,
            "adobe_extract_path": f'https://{BLOB_ACCOUNT_NAME}.blob.core.windows.net/{OCR_CONTAINER_NAME}/output/adobe/{adobe_api_filename}',
            "pdf_path": doc_info['DocumentPath']
        })

        if not os.path.isfile(json_data_path):
            if not os.path.isfile(adobe_api_zip_path):
                try:
                    add_info_log(current_step, "Attempting to download existing Adobe extract")
                    if doc_info.get('AdobeExtractPath'):
                        adobe_blob_name = doc_info['AdobeExtractPath'].split('/adobe/')[-1]
                        download_result = download_blob(
                            adobe_blob_name,
                            input_dir=adobe_output_dir,
                            container_name=f"{OCR_CONTAINER_NAME}/output/adobe",
                            save_as=adobe_api_filename
                        )
                        download_result.unwrap()
                        add_info_log(current_step, "Successfully downloaded existing Adobe extract")
                except Exception as e:
                    add_error_log(current_step, e, "AdobeDownloadError")
                    # Generate new Adobe extract if download fails
                    try:
                        add_info_log(current_step, "Generating new Adobe extract")
                        check_encoding_result = checkPDFencodingerrors(pdf_file_path)
                        pdf_processing_path = check_encoding_result.unwrap()
                        
                        extract_result = run_extract_pdf(
                            filename=pdf_processing_path,
                            adobe_dir=adobe_api_zip_path,
                            logger_name=app_insight_logger,
                            request_id=request_id
                        )
                        extract_result.unwrap()
                        add_info_log(current_step, "Successfully generated new Adobe extract")
                    except Exception as e:
                        add_error_log(current_step, e, "AdobeGenerationError")
                        return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}

            # Extract ZIP file
            try:
                add_info_log(current_step, "Extracting Adobe ZIP file")
                extract_zip_result = extractZipNew(zip_file_path=adobe_api_zip_path)
                extract_zip_result.unwrap()
                add_info_log(current_step, "Successfully extracted Adobe ZIP file")
            except Exception as e:
                add_error_log(current_step, e, "ZipExtractionError")
                return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}

        # Document extraction step
        current_step = "form_extraction"
        try:
            add_info_log(current_step, "Starting form number extraction")
            # Extract form number for split document
            if is_split_document:
                split_processor = DocumentSplitProcessor(
                    base_doc_id, 
                    pdf_file_path, 
                    json_data_path, 
                    compiledformnumberpatternfromconfig
                )
                split_result, successfailuremessage = split_processor.run()

                if successfailuremessage != "Success":
                    if "declaration" not in metadata["document_info"]["doc_type"].lower():
                        add_error_log(current_step, successfailuremessage, "SplitProcessingError")
                        return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}

                split_text = split_result.get(doc_id, {}).get('text')
                if not split_text:
                    if "declaration" not in metadata["document_info"]["doc_type"].lower():
                        error_msg = f"No split text found for document {doc_id}"
                        add_error_log(current_step, error_msg, "NoSplitTextError")
                        return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}
                    else:
                        add_info_log(current_step, "No split text found for declaration document, continuing processing")
                        split_text = ""

                try:
                    doc_num_result = extractDocumentNumber(split_text, compiledformnumberpatternfromconfig,config,pdf_file_path,client, AZURE_OPENAI_CHATGPT_DEPLOYMENT)
                    document_number, normalized_document_number = doc_num_result.unwrap()
                    metadata["extraction_info"].update({
                        "form_number": document_number,
                        "normalized_form_number": normalized_document_number
                    })
                    add_info_log(current_step, "Successfully extracted form number", {
                        "form_number": document_number,
                        "normalized_form_number": normalized_document_number
                    })
                except Exception as e:
                    if "declaration" not in metadata["document_info"]["doc_type"].lower():
                        add_error_log(current_step, e, "FormNumberExtractionError")
                        return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}
                    add_info_log(current_step, "Form number extraction failed for declaration, continuing processing")

            else:
                # Process non-split document
                    split_processor = DocumentSplitProcessor(
                        doc_id, 
                        pdf_file_path, 
                        json_data_path, 
                        None
                    )
                    split_result, successfailuremessage = split_processor.run()

                    if successfailuremessage != "Success":
                        if "declaration" not in metadata["document_info"]["doc_type"].lower():
                            add_error_log(current_step, successfailuremessage, "SplitProcessingError")
                            return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}
                        add_info_log(current_step, "Split processing failed for declaration, continuing processing")

                    try:
                        doc_num_result = extractDocumentNumber(split_result, compiledformnumberpatternfromconfig,config,pdf_file_path,client, AZURE_OPENAI_CHATGPT_DEPLOYMENT)
                        document_number, normalized_document_number = doc_num_result.unwrap()
                        metadata["extraction_info"].update({
                            "form_number": document_number,
                            "normalized_form_number": normalized_document_number
                        })
                        add_info_log(current_step, "Successfully extracted form number", {
                            "form_number": document_number,
                            "normalized_form_number": normalized_document_number
                        })
                    except Exception as e:
                        if "declaration" not in metadata["document_info"]["doc_type"].lower():
                            add_error_log(current_step, e, "FormNumberExtractionError")
                            return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}
                        add_info_log(current_step, "Form number extraction failed for declaration, continuing processing")

            # Additional declaration processing
            if metadata["document_info"]["doc_type"] == "Declaration":
                current_step = "declaration_processing"
                try:
                    add_info_log(current_step, "Processing declaration form numbers")
                    form_number, successerror_message = process_ordering(
                        adobe_api_zip_path,
                        config=config
                    )
                    
                    if successerror_message == "Success":
                        normalize_form_number = [normalize_pattern(i) for i in form_number]
                        metadata["extraction_info"]["declaration_forms"].update({
                            "original": form_number.copy(),
                            "normalized": normalize_form_number.copy()
                        })

                        add_info_log(current_step, "Initial form number collection completed", {
                            "form_count": len(form_number),
                            "original_forms": form_number
                        })

                        # Remove document's own form number if it exists
                        if metadata["extraction_info"]["form_number"]:
                            matched_indexes = match_document_numbers(metadata["extraction_info"]["form_number"], form_number)
                            matched_indexes.sort(reverse=True)
                            for index in matched_indexes:
                                del form_number[index]
                                del normalize_form_number[index]
                            
                            add_info_log(current_step, "Removed document's own form number", {
                                "removed_indexes": matched_indexes,
                                "remaining_count": len(form_number)
                            })

                        # Handle excluded forms
                        if form_number and 'ExcludedFormNumbers' in config:
                            excluded_forms = [normalize_pattern(form) for form in config['ExcludedFormNumbers']]
                            final_forms = []
                            final_normalized = []
                            excluded = []
                            excluded_count = 0

                            for idx, (form_num, norm_form) in enumerate(zip(form_number, normalize_form_number)):
                                if any(excl_form in norm_form for excl_form in excluded_forms):
                                    excluded.append({"FormNumber": form_num, "NormalizedFormNumber": norm_form})
                                    excluded_count += 1
                                else:
                                    final_forms.append(form_num)
                                    final_normalized.append(norm_form)
                            
                            metadata["extraction_info"]["excluded_forms"] = excluded
                            metadata["extraction_info"]["final_forms"] = [
                                {"FormNumber": fn, "NormalizedFormNumber": nfn}
                                for fn, nfn in zip(final_forms, final_normalized)
                            ]
                            
                            add_info_log(current_step, "Processed excluded forms", {
                                "excluded_count": excluded_count,
                                "final_form_count": len(final_forms)
                            })
                        else:
                            # Use remaining form numbers after own form number removal
                            metadata["extraction_info"]["final_forms"] = [
                                {"FormNumber": fn, "NormalizedFormNumber": nfn}
                                for fn, nfn in zip(form_number, normalize_form_number)
                            ]
                            
                            add_info_log(current_step, "Completed form number processing without exclusions", {
                                "final_form_count": len(metadata["extraction_info"]["final_forms"])
                            })
                    else:
                        metadata["extraction_info"]["final_forms"] = []
                        add_info_log(current_step, f"Form number processing failed: {successerror_message}")

                except Exception as e:
                    metadata["extraction_info"]["final_forms"] = []
                    add_error_log(current_step, e, "DeclarationProcessingError", {
                        "continues_processing": True
                    })
                    app_insight_logger.warning(f"Error processing declaration form numbers for {doc_id}: {str(e)}", extra=properties)

            add_info_log("completion", "Document processing completed successfully", {
                "has_form_number": bool(metadata["extraction_info"]["form_number"]),
                "has_final_forms": bool(metadata["extraction_info"]["final_forms"])
            })
            return doc_id, 'Success', metadata, msg, {"info": info_logs, "error": error_logs}

        except Exception as e:
            add_error_log(current_step, e, "DocumentProcessingError")
            return doc_id, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}

    except Exception as e:
        error_message = f"Unhandled exception in process_extraction_retry_item: {str(e)}"
        if not current_step:
            current_step = "initialization"
        add_error_log(current_step, e, "UnhandledError")
        return doc_id if 'doc_id' in locals() else None, 'Failed', metadata, msg, {"info": info_logs, "error": error_logs}
 

def process_manual_split_item(msg):
    """
    Process a manual split request for a merged document.
    Returns:
        tuple: (doc_id, status, metadata, next_messages)
    """
    # Initialize logging collections and current step
    info_logs = []
    error_logs = []
    current_step = "initialization"

    def add_info_log(step, message, extra_metadata=None):
        """Helper to record info logs."""
        log_metadata = {
            "carrier_name": metadata["document_info"].get("carrier_name"),
            "document_name": metadata["document_info"].get("blob_name"),
            "document_type": metadata["document_info"].get("doc_type"),
            "processing_path": metadata["paths"].get("pdf_path"),
            "is_non_frontend": metadata["document_info"].get("is_non_frontend")
        }
        if extra_metadata:
            log_metadata.update(extra_metadata)
        info_logs.append({
            "document_id": metadata.get("doc_id"),
            "processor_type": "manual_split",
            "request_id": request_id if 'request_id' in locals() else None,
            "timestamp": datetime.now(),
            "step": step,
            "message": message,
            "metadata": log_metadata
        })

    def add_error_log(step, error, error_type, extra_metadata=None):
        """Helper to record error logs."""
        log_metadata = {
            "carrier_name": metadata["document_info"].get("carrier_name"),
            "document_name": metadata["document_info"].get("blob_name"),
            "document_type": metadata["document_info"].get("doc_type"),
            "processing_path": metadata["paths"].get("pdf_path"),
            "is_non_frontend": metadata["document_info"].get("is_non_frontend")
        }
        if extra_metadata:
            log_metadata.update(extra_metadata)
        error_logs.append({
            "document_id": metadata.get("doc_id"),
            "processor_type": "manual_split",
            "request_id": request_id if 'request_id' in locals() else None,
            "timestamp": datetime.now(),
            "error_type": error_type,
            "error_message": str(error),
            "traceback": ''.join(traceback.format_exception(None, error, error.__traceback__)),
            "step": step,
            "metadata": log_metadata
        })

    # Initialize metadata structure
    metadata = {
        "doc_id": None,
        "paths": {
            "pdf_path": None,
            "adobe_extract_path": None,
            "adobe_blob_name": None,
            "input_dir": None,
            "pdf_file_path": None,
            "adobe_output_dir": None
        },
        "status": {
            "code": None,
            "message": None,
            "state": None
        },
        "document_info": {
            "carrier_name": None,
            "doc_type": "Merged Document",
            "blob_name": None,
            "is_non_frontend": False
        },
        "split_info": {
            "total_parts": 0,
            "splits": {},
            "ordering": None
        },
        "processing_info": {
            "uploaded_at": datetime.now(),
            "split_completed": False
        }
    }

    try:
        # Message unpacking
        current_step = "message_unpacking"
        msg_type, msg_data = msg
        doc_info, split_data, request_id, app_insight_logger = msg_data
        add_info_log(current_step, f"Starting manual split processing for document.")

        # Extract basic document info
        try:
            doc_id = metadata["doc_id"] = doc_info['ID']
            blobname = metadata["document_info"]["blob_name"] = doc_info['DocumentName']
            carrier_name = metadata["document_info"]["carrier_name"] = doc_info['CarrierName']
            # Determine if this is a non–frontend merged document (no underscore in blobname)
            is_non_frontend = metadata["document_info"]["is_non_frontend"] = doc_info.get('UserID') is None
            add_info_log(current_step, "Extracted basic document info.", {
                "doc_id": doc_id,
                "blobname": blobname,
                "carrier_name": carrier_name,
                "is_non_frontend": is_non_frontend
            })
        except Exception as e:
            add_error_log(current_step, e, "DocumentInfoExtractionError")
            metadata["status"].update({
                "state": "Failed",
                "message": f"Failed to extract basic document info: {str(e)}"
            })
            return doc_id if metadata.get("doc_id") else None, 'Failed', metadata, None

        # Setup paths
        current_step = "path_setup"
        try:
            input_dir = os.path.join(MERGED_DIR, str(doc_id))
            pdf_file_path = os.path.join(input_dir, blobname)
            adobe_output_dir = os.path.join(input_dir, f"adobe-api-{blobname.split('.')[0]}-{doc_id}")
            adobe_api_filename = f"adobe-api-{blobname.split('.')[0]}-{doc_id}.zip"
            adobe_api_zip_path = os.path.join(adobe_output_dir, adobe_api_filename)
            metadata["paths"].update({
                "input_dir": input_dir,
                "pdf_file_path": pdf_file_path,
                "adobe_output_dir": adobe_output_dir,
                "adobe_blob_name": adobe_api_filename
            })
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(adobe_output_dir, exist_ok=True)
            add_info_log(current_step, "Set up processing paths.", {
                "input_dir": input_dir,
                "pdf_file_path": pdf_file_path,
                "adobe_output_dir": adobe_output_dir
            })
        except Exception as e:
            add_error_log(current_step, e, "PathSetupError")
            metadata["status"].update({
                "state": "Failed",
                "message": f"Failed to set up paths: {str(e)}"
            })
            return doc_id, 'Failed', metadata, None

        # Download PDF if needed
        current_step = "pdf_download"
        if not os.path.isfile(pdf_file_path):
            try:
                processed_blob_name = doc_info['DocumentPath'].split('/input/')[-1]
                add_info_log(current_step, "PDF not found locally. Downloading PDF.", {"processed_blob": processed_blob_name})
                download_result = download_blob(
                    processed_blob_name,
                    input_dir=input_dir,
                    container_name=f"{OCR_CONTAINER_NAME}/input",
                    save_as=blobname
                )
                download_result.unwrap()
                add_info_log(current_step, "Successfully downloaded PDF.")
            except Exception as e:
                add_error_log(current_step, e, "PDFDownloadError")
                metadata["status"].update({
                    "state": "Failed",
                    "message": f"Failed to download PDF {processed_blob_name}: {str(e)}"
                })
                return doc_id, 'Failed', metadata, None

        # Check PDF encoding
        current_step = "pdf_encoding"
        try:
            add_info_log(current_step, "Performing PDF encoding check.")
            check_encoding_result = checkPDFencodingerrors(pdf_file_path)
            pdf_processing_path = check_encoding_result.unwrap()
            add_info_log(current_step, "PDF encoding check completed.")
        except Exception as e:
            add_error_log(current_step, e, "PDFEncodingError")
            metadata["status"].update({
                "state": "Failed",
                "message": f"PDF encoding check failed: {str(e)}"
            })
            return doc_id, 'Failed', metadata, None

        # Adobe API processing (if needed)
        current_step = "adobe_processing"
        if not doc_info.get('AdobeExtractPath'):
            try:
                add_info_log(current_step, "Starting Adobe API processing for manual split.")
                extract_result = run_extract_pdf(
                    filename=pdf_processing_path,
                    adobe_dir=adobe_api_zip_path,
                    logger_name=app_insight_logger,
                    request_id=request_id
                )
                extract_result.unwrap()
                add_info_log(current_step, "Adobe API extraction completed. Extracting ZIP contents.")
                extract_zip_result = extractZipNew(zip_file_path=adobe_api_zip_path)
                extract_zip_result.unwrap()
                # Prepare the PDF blob name (remove spaces if needed)
                pdf_blob_name = f"{blobname.split('.')[0]}_{doc_id}.pdf"
                pdf_blob_name = remove_spaces(pdf_blob_name)
                metadata["paths"].update({
                    "pdf_path": f'https://{BLOB_ACCOUNT_NAME}.blob.core.windows.net/{OCR_CONTAINER_NAME}/input/{pdf_blob_name}',
                    "adobe_extract_path": f'https://{BLOB_ACCOUNT_NAME}.blob.core.windows.net/{OCR_CONTAINER_NAME}/output/adobe/{adobe_api_filename}'
                })
                add_info_log(current_step, "Updated paths after Adobe processing.", {
                    "pdf_path": metadata["paths"].get("pdf_path"),
                    "adobe_extract_path": metadata["paths"].get("adobe_extract_path")
                })
            except Exception as e:
                add_error_log(current_step, e, "AdobeProcessingError")
                metadata["status"].update({
                    "state": "Failed",
                    "message": f"Adobe API processing failed: {str(e)}"
                })
                return doc_id, 'Failed', metadata, None

        # Process split information
        current_step = "split_processing"
        json_data_path = os.path.join(adobe_output_dir, "extractedAdobeRes", "structuredData.json")
        try:
            add_info_log(current_step, "Reading Adobe JSON data.", {"json_data_path": json_data_path})
            with open(json_data_path, 'r') as json_file:
                json_data = json.load(json_file)
            elements_list = json_data.get('elements', [])
            if not elements_list:
                error_msg = "No elements found in Adobe JSON"
                add_error_log(current_step, error_msg, "EmptyAdobeJSON")
                metadata["status"].update({
                    "state": "Failed",
                    "message": error_msg
                })
                return doc_id, 'Failed', metadata, None
        except Exception as e:
            add_error_log(current_step, e, "JSONReadError")
            metadata["status"].update({
                "state": "Failed",
                "message": f"Failed to read Adobe JSON data: {str(e)}"
            })
            return doc_id, 'Failed', metadata, None

        # Process document splits using API results and the provided split data
        try:
            add_info_log(current_step, "Processing document splits.")
            split_dict, ordering_key = process_document_splitsforapi(split_data, elements_list)
            merged_split_dict = merge_declarations(split_dict, ordering_key)
            metadata["split_info"].update({
                "total_parts": len(merged_split_dict),
                "splits": merged_split_dict,
                "ordering": ordering_key
            })
            add_info_log(current_step, "Document splits processed.", {
                "total_parts": len(merged_split_dict)
            })
            # Create new messages for each split part
            new_messages = []
            for key, split_info in merged_split_dict.items():
                # Extract an index from the key (assumes key format contains an underscore and a number)
                try:
                    index = int(key.split('_')[-1])
                except Exception:
                    index = 0
                # Determine folder name based on ordering: default to Endorsement,
                # but if the split index matches the Base or Declaration ordering, set accordingly.
                folder_name = "Endorsement"
                if ordering_key.get("Base") == index:
                    folder_name = "Base"
                elif ordering_key.get("Declaration") and ordering_key["Declaration"][0] == index:
                    folder_name = "Declaration"
                # Build the split indices dictionary for the message.
                split_indices_dict = {
                    'frontendflag': 'N',
                    'text': split_info.get('text'),
                    'split': split_info.get('split')
                }
                if not is_non_frontend:
                    split_indices_dict['frontendmergedupload'] = 'Y'
                message_data = (
                    blobname,
                    key,  # This will be the split document's ID
                    f"{blobname.split('.')[0]}_{key}",
                    f"{OCR_CONTAINER_NAME}/input",
                    carrier_name,
                    folder_name,
                    app_insight_logger,
                    request_id,
                    split_indices_dict
                )
                new_messages.append(message_data)
            metadata["status"].update({
                "state": "Success",
                "message": "Manual split completed successfully"
            })
            metadata["processing_info"]["split_completed"] = True
            add_info_log("completion", "Manual split processing completed.", {"total_new_messages": len(new_messages)})
            return doc_id, 'Success', metadata, new_messages
        except Exception as e:
            add_error_log(current_step, e, "SplitProcessingError")
            metadata["status"].update({
                "state": "Failed",
                "message": f"Failed to process document splits: {str(e)}"
            })
            return doc_id, 'Failed', metadata, None

    except Exception as e:
        error_message = f"Unhandled exception in process_manual_split_item: {str(e)}"
        add_error_log(current_step, e, "UnhandledError")
        if not metadata["status"].get("state"):
            metadata["status"].update({
                "state": "Failed",
                "message": error_message
            })
        return metadata.get("doc_id"), 'Failed', metadata, None


def start_background_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


async def main():
    global receive_messages_started
    app_insight_logger.info(f"Inside main() function in process id: {os.getpid()}", extra=properties)
    
    # Start regular message receiver if not already started
    if not receive_messages_started:
        receive_task = asyncio.create_task(receive_messages())
        receive_messages_started = True
        app_insight_logger.info("Starting the receive_messages().", extra=properties)
    else:
        app_insight_logger.info("receive_messages() was already started.", extra=properties)
    
    # Start MSA Lease message receiver
    receive_msa_lease_task = asyncio.create_task(receiveMSALeaseMessages())
    app_insight_logger.info("Starting the receiveMSALeaseMessages().", extra=properties)
    
    process_task = asyncio.create_task(process_batches())
    await asyncio.gather(receive_task, receive_msa_lease_task, process_task)


def run_main():
    global background_loop
    app.logger.info(f"Inside run_main() function in process id: {os.getpid()}")
    background_loop = asyncio.new_event_loop()

    # Start the background loop in a separate thread
    t = Thread(target=start_background_loop, args=(background_loop,), daemon=True)
    t.start()

    # Schedule the main coroutine to run in the background
    if os.getpid() == main_pid:
        main_future = asyncio.run_coroutine_threadsafe(main(), background_loop)
    else:
        app.logger.info("Not scheduling main() since this is not the main process")

    try:
        if ENV_NAME in ENV_NAMES:
            app.run(
                debug=False,
                use_reloader=False,
                threaded=False,
                processes=1,
                port=WEBSITE_PORT,
                host="0.0.0.0",
            )
        else:
            app.run(
                debug=True,
                use_reloader=False,
                threaded=False,
                processes=1,
                port=WEBSITE_PORT,
            )
    except KeyboardInterrupt:
        app_insight_logger.info("Received KeyboardInterrupt", extra=properties)
    finally:
        app.logger.info("Shutting down")
        stop_event.set()
        if os.getpid() == main_pid:
            main_future.result()
        background_loop.call_soon_threadsafe(background_loop.stop)
        t.join()
        app.logger.info("Background tasks have been shut down")




if __name__ == "__main__":
    app_insight_logger.info("Starting the application...", extra=properties)
    os.makedirs("./" + BASE_ENDORSEMENT_DIR + "/", exist_ok=True)
    os.makedirs("./" + DECLARATION_DIR + "/", exist_ok=True)
    os.makedirs("./" + MERGED_DIR + "/", exist_ok=True)
    os.makedirs("./" + "MSALease" + "/", exist_ok=True)

    run_main()