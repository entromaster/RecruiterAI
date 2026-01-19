import json
import os
import uuid
from datetime import datetime
from urllib.parse import urlparse
import asyncio
from asyncio import Queue
from agentic_rag.processor.batch_processor import (
    process_base_endo_batch
)
from agentic_rag import (
    get_clientID_async,
    get_document_async,
    to_mongo_async,
    update_status_mongo_async,
    get_merge_doc_obj,
    get_document_obj,
    get_traceback,
    contains_uuid4
)
from agentic_rag.src import (
    APP_NAME,
    STATUS_IN_PROGRESS,
    WORKER_AND_BATCH_SIZE,
    MONGO_MERGED_COLLECTION,
    MONGO_DOCUMENT_COLLECTION,
)

base_endo_queue = Queue()
declaration_queue = Queue()
merged_queue = Queue()
properties = {'custom_dimensions': {'ApplicationName': APP_NAME}}
app_insight_logger = None

async def process_message(message_data):
    folder_name = message_data["folder_name"]
    if "declaration" in folder_name:
        await declaration_queue.put(message_data)
    elif "endorsement" in folder_name or "base" in folder_name or "msa" in folder_name:
        await base_endo_queue.put(message_data)
    elif "merged" in folder_name:
        await merged_queue.put(message_data)

async def receive_messages(message):
    pass

async def receive_mssage(message):
    try:
        print("\n\n",f"AGENTIC RAG MESSAGE RECEIVED.")
        for body in message.body:
            try:
                j_body = json.loads(body)
                document_path = j_body["data"]["url"]
                parsed_url = urlparse(document_path)
                path_parts = parsed_url.path.strip('/').split('/')
                if len(path_parts) < 4:
                    app_insight_logger.error(f"Invalid document path structure (less than 4 parts): {document_path}", extra=properties)
                blobname = path_parts[-1]
                folder_name = path_parts[-2].lower() # Immediate parent folder
                client_name = path_parts[-3]
                blob_download_prefix = "/".join(path_parts[:-1])    
                doc_id = None
                user_id = None
                use_case = None
                policy_name = None
                transformed_name = blobname
                msa_split = blobname.split("_")
                if "==" in blobname:
                    filename_parts = blobname.split("==")
                    if len(filename_parts) == 5:
                        original_name, doc_id, user_id, policy_name, use_case_part = filename_parts
                        use_case, doc_ext = os.path.splitext(use_case_part)
                        transformed_name = original_name + doc_ext # Assumes exists
                        app_insight_logger.info(f"Frontend upload detected. UserID: {user_id}, PolicyName: {policy_name}, UseCase: {use_case}", extra=properties)
                elif len(msa_split) == 2 and contains_uuid4(msa_split[1]):
                        doc_id, ext = os.path.splitext(msa_split[1])
                        transformed_name = msa_split[0] + ext
                else:
                    doc_id = str(uuid.uuid4())
                    app_insight_logger.info(f"Agentic generated new doc_id for B2B upload: {doc_id}", extra=properties)
                document_exist = await get_document_async({"documentId": doc_id}, MONGO_DOCUMENT_COLLECTION)
                client_name = document_exist.get("clientName") if document_exist else client_name
                policy_name = document_exist.get("documentSubTag") if document_exist else policy_name
                client_id = await get_clientID_async(client_name) if client_name else None
                doc_obj = get_document_obj(
                    doc_id=doc_id, 
                    doc_name=transformed_name,
                    client_name=client_name,
                    client_id=client_id,
                    user_id=user_id,
                    doc_type=None, # Pass finalized type
                    file_path=document_path, 
                    policy_name=policy_name
                )
                if "base" in folder_name:
                    doc_obj["documentTag"] = "BasePolicy"
                elif "endo" in folder_name:
                    doc_obj["documentTag"] = "Endorsement"
                elif "declaration" in folder_name:
                    doc_obj["documentTag"] = "Declaration"
                    doc_obj["metaData"] = {"version": 1}
                elif "merged" in folder_name:
                    doc_obj["documentTag"] = "MergeNoForm"
                elif "msa" in folder_name:
                    doc_obj["documentTag"] = document_exist.get("documentTag")
                    policy_name = document_exist.get("documentSubTag")
                app_insight_logger.info(f"DOC ID {doc_id}\nDOCUMENT EXIST in {MONGO_DOCUMENT_COLLECTION}:\n{document_exist}", extra=properties)
                declaration_version = 1
                if not document_exist:
                    await to_mongo_async([doc_obj], MONGO_DOCUMENT_COLLECTION)
                    app_insight_logger.info(f"Inserted documents with the following IDs to: {MONGO_DOCUMENT_COLLECTION}", extra=properties)
                else:
                    if folder_name in ["merged-document", "declaration-document", "msaleaseagreementinput"]:
                        declaration_version = document_exist.get("metaData").get("version", 0) + 1 if document_exist.get("metaData") else 1
                        update_doc = {"processStatus": STATUS_IN_PROGRESS,
                                      "uploadDate": datetime.now(),
                                      "documentName": transformed_name}
                        await update_status_mongo_async({"documentId": doc_id}, update_doc)
                message_data = {
                    "blobname": blobname,
                    "doc_id": doc_id,
                    "transformed_name": transformed_name,
                    "doc_type": doc_obj["documentTag"],
                    "file_path": document_path,
                    "client_name": client_name,
                    "policy_name": policy_name,
                    "folder_name": folder_name,
                    "blob_download_prefix": blob_download_prefix,
                    "declaration_version": declaration_version,
                    "app_insight_logger": app_insight_logger
                }
                await process_message(message_data)
                await asyncio.sleep(0.1)
            except asyncio.CancelledError as e:
                err_msg = get_traceback(e, "receive_messages task cancelled.")
                app_insight_logger.info(err_msg, extra=properties)
                break
            except Exception as e:
                err_msg = get_traceback(e, "Failed receive_messages task cancelled.")
                app_insight_logger.info(err_msg, extra=properties)
    except asyncio.CancelledError as e:
        err_msg = get_traceback(e, "Failed receive_messages task cancelled.")
        app_insight_logger.info(err_msg, extra=properties)
    except Exception as e:
        err_msg = get_traceback(e, "Failed while receiving the messages.")
        app_insight_logger.info(err_msg, extra=properties)

    
def form_batches(queue, batch_size=WORKER_AND_BATCH_SIZE):
    batches = []
    current_batch = []
    while not queue.empty():
        # try:
        item = queue.get_nowait()
        current_batch.append(item)
        if len(current_batch) == batch_size:
            batches.append(current_batch)
            current_batch = []
    if current_batch:
        batches.append(current_batch)
    return batches

async def process_batches(stop_event):
    iteration = 0
    while not stop_event.is_set():
        iteration += 1
        await asyncio.sleep(2)
        tasks = []
        # Process base_endo_queue
        while not base_endo_queue.empty():
            base_endo_batches = form_batches(base_endo_queue)
            for batch in base_endo_batches:
                task = asyncio.create_task(safe_process_batch(process_base_endo_batch(batch, app_insight_logger)))
                tasks.append(task)
        # Process declaration_queue
        while not declaration_queue.empty():
            declaration_batches = form_batches(declaration_queue)
            for batch in declaration_batches:
                task = asyncio.create_task(safe_process_batch(process_base_endo_batch(batch, app_insight_logger)))
                tasks.append(task)
        # Process merged_queue
        while not merged_queue.empty():
            merged_batches = form_batches(merged_queue)
            for batch in merged_batches:
                task = asyncio.create_task(safe_process_batch(process_base_endo_batch(batch, app_insight_logger)))
                tasks.append(task)

async def safe_process_batch(coro):
    try:
        result = await coro
        print(f"Successfully processed batch")
        return result
    except Exception as e:
        print(f"Error processing batch: {e}")
        import traceback
        traceback.print_exc()
        return None
 
def agentic_rag_main(stop_event, insight_logger):
    global app_insight_logger
    app_insight_logger = insight_logger
    tasks = []
    process_base_endo_task = asyncio.create_task(process_batches(stop_event))
    tasks.append(process_base_endo_task)
    app_insight_logger.info("Starting the Agentic RAG Messages().", extra=properties)
    return tasks