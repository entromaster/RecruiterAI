import os
import asyncio
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from agentic_rag.src import (
    WORKER_AND_BATCH_SIZE,
    STATUS_PROCESSED,
    MONGO_AGENTIC_RAG_COLLECTION,
    APP_NAME,
)
from agentic_rag import (
     clear_dir,
     to_mongo_async,
     update_status_mongo_async,
     get_traceback,
     get_mapped_child_doc_ids_async,
     get_document_async
)

from agentic_rag.processor.item_processor import item_processor

properties = {'custom_dimensions': {'ApplicationName': APP_NAME}}

async def process_declaration_batch(batch):
    pass

async def process_base_endo_batch(batch, app_insight_logger):
    print("\n\n","Inside batch processor.\n\n", batch)
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=WORKER_AND_BATCH_SIZE) as executor:
        tasks = [loop.run_in_executor(executor, item_processor, msg) for msg in batch]
        for future in asyncio.as_completed(tasks):
            try:
                metadata = await future
                app_insight_logger = metadata["app_insight_logger"]
                if metadata["status"]["state"] == STATUS_PROCESSED:
                    await to_mongo_async(
                        metadata["result"]["chunks_obj"], 
                        MONGO_AGENTIC_RAG_COLLECTION
                        )
                    if metadata["doc_type"] in ["Declaration", "MergeNoForm"]:
                        child_doc_ids, extracted_form_numbers = await get_mapped_child_doc_ids_async(metadata["doc_id"], 
                                                                                                     metadata["client_name"], 
                                                                                                     metadata["policy_name"], metadata["result"]["form_numbers"])
                        update_doc = {
                            "$set": {
                                "uniqueId": metadata["result"]["declaration_metadata"]["policy_number"],
                                "processStatus": metadata["status"]["state"],
                                "processedDate": datetime.now(),
                                "documentSubTag": metadata["policy_name"],
                                "metaData": {
                                    "version": metadata["result"]["declaration_metadata"]["version"],
                                    "holderName": metadata["result"]["declaration_metadata"]["holder_name"],
                                    "startDate": datetime.strptime(metadata["result"]["declaration_metadata"]["start_date"], "%m/%d/%Y"),
                                    "endDate": datetime.strptime(metadata["result"]["declaration_metadata"]["end_date"], "%m/%d/%Y"),
                                },
                            },
                            "$addToSet": {
                                "childDocumentIds": {"$each": child_doc_ids},   # append unique child docs
                                "extractedFormNumbers": {"$each": extracted_form_numbers}  # append unique form numbers
                            }
                        }                        
                    else:
                        child_doc_obj = {
                            "$addToSet": {
                                "childDocumentIds": {
                                    "documentId": metadata["doc_id"],
                                    "isActive": True,
                                    "documentTag": metadata["doc_type"],
                                    "filePath": metadata["file_path"],
                                    "jsonId": None
                                }
                            }
                        }
                        query = {
                            "documentTag": "Declaration",
                            "documentSubTag": metadata["policy_name"],
                            "clientName": metadata["client_name"]
                            }
                        dec_doc_exist = await get_document_async(query)
                        if dec_doc_exist:
                            await update_status_mongo_async({"documentId": dec_doc_exist.get("documentId")}, child_doc_obj)
                        update_doc = {
                            "processStatus": metadata["status"]["state"],
                            "processedDate": datetime.now(),
                            "documentSubTag": metadata["policy_name"],
                        } 
                else:
                    update_doc = {
                            "processStatus": metadata["status"]["state"],
                            "processedDate": datetime.now(),
                            "documentSubTag": metadata["policy_name"],
                            "failure.reason": metadata["status"]["message"],
                            "failure.code": metadata["status"]["code"],
                        }
                try:
                    await update_status_mongo_async(
                        {"documentId": metadata["doc_id"]},
                        update_doc
                    )
                    if metadata["paths"]["pdf_file_path"]:
                        clear_dir(os.path.dirname(metadata["paths"]["pdf_file_path"]), cur_dir=True) 
                except Exception as e:
                    err_msg = get_traceback(e, "Failed while updating status inside batch processor.")
                    app_insight_logger.info(err_msg, extra=properties)
            except Exception as e:
                clear_dir(os.path.dirname(metadata["paths"]["pdf_file_path"]), cur_dir=True)
                err_msg = get_traceback(e, "Failed while updating the status in db:")
                app_insight_logger.info(err_msg, extra=properties)