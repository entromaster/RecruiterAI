from agentic_rag import (
    get_metadata_obj,
    get_processing_paths,
    download_blob,
    get_traceback,
    extract_pypdf_text,
    get_chunks_obj,
    vision_extractor,
    check_encoding_and_convert_pdf_2_image,
    scanned_2_text,
    is_image_pdf,
    get_processed_form_numbers    
)
from agentic_rag.src import (
    INPUT_DIR,
    STATUS_FAILED,
    STATUS_PROCESSED
)

def item_processor(msg_data):
    print("\n\n","Inside item processor.", msg_data["doc_id"])
    try:
        metadata = get_metadata_obj()
        if not isinstance(msg_data, dict):
            error_msg = f"Invalid message data type: expected dict, got {type(msg_data)}"
            metadata["status"].update({
                "code": [114], 
                "message": error_msg, 
                "state": STATUS_FAILED
            })
            return metadata
        required_fields = ["doc_id", "policy_name", "folder_name", "blobname",
                          "transformed_name", "blob_download_prefix", 
                          "doc_type", "declaration_version", "client_name", 
                          "app_insight_logger", "file_path"]
        
        missing_fields = [field for field in required_fields if field not in msg_data]
        if missing_fields:
            error_msg = f"Invalid message data: missing fields {missing_fields}"
            metadata["status"].update({
                "code": [114], 
                "message": error_msg, 
                "state": STATUS_FAILED
            })
            return metadata
        
        doc_id = metadata["doc_id"] = msg_data["doc_id"]
        blobname = msg_data["blobname"]
        transformed_name = msg_data["transformed_name"]
        blob_download_prefix = msg_data["blob_download_prefix"]
        doc_type = msg_data["doc_type"]
        declaration_version = msg_data["declaration_version"]
        client_name = msg_data["client_name"]
        policy_name = msg_data["policy_name"]
        app_insight_logger = msg_data["app_insight_logger"]
        file_path = msg_data["file_path"]
        
        input_dir, pdf_file_path = get_processing_paths(INPUT_DIR, doc_id, blobname)
        metadata["original_message"] = msg_data 
        metadata["paths"]["input_dir"] = input_dir
        metadata["paths"]["pdf_file_path"] = pdf_file_path
        metadata["client_name"] = client_name
        metadata["policy_name"] = policy_name
        metadata["doc_type"] = doc_type
        metadata["app_insight_logger"] = app_insight_logger
        metadata["transformed_doc_name"] = transformed_name
        metadata["file_path"] = file_path
        try:
            download_blob(
                blobname=blobname,
                input_dir=input_dir,
                container_name=blob_download_prefix
            )
        except Exception as e:
            err_msg = get_traceback(e, f"Failed to download blob '{blobname}' from '{blob_download_prefix}':")
            metadata["status"].update({
                "code": [115],
                "message": err_msg,
                "state": STATUS_FAILED
            })
            return metadata
        # Convert scanned PDF to selectable text
        # try:
        #     pdf_file_path, has_unicode_issues = check_encoding_and_convert_pdf_2_image(pdf_file_path)
        #     is_image_type = is_image_pdf(pdf_file_path)
        #     if has_unicode_issues or is_image_type:
        #         pdf_file_path, error = scanned_2_text(pdf_file_path)
        #         if error:
        #             metadata["status"].update({
        #             "code": 115,
        #             "message": error,
        #             "state": STATUS_FAILED
        #         })
        #         return metadata

        # except:
        #     err_msg = get_traceback(e)
        #     metadata["status"].update({
        #         "code": 115,
        #         "message": f"Failed to check PDF encoding and convert to image: {err_msg}",
        #         "state": STATUS_FAILED
        #     })
        #     return metadata
        try:
            full_text, total_pages, text = extract_pypdf_text(pdf_file_path, transformed_name, doc_id)
        except Exception as e:
            err_msg = get_traceback(e, "Failed to extract text using PyPDF2:")
            metadata["status"].update({
                "code": [120],
                "message": err_msg,
                "state": STATUS_FAILED
            })
            return metadata
        try:
            if doc_type in ["MergeNoForm", "Declaration"]:
                declaration_metadata, error = vision_extractor(target_text=text, extraction_type="declaration_metadata")
                # Extract the form numbers to perform the mapping using form numbers
                # csv_tables, error = vision_extractor(pdf_file_path, max_pages=3, extraction_type="tables")
                # pure_form_table_cols = ['NAME', 'FORM NUMBER', 'EDITION DATE']
                # if csv_tables:
                #     metadata["form_numbers"] = get_processed_form_numbers(csv_tables, pure_form_table_cols)
                # else:
                #     metadata["status"].update({
                #         "code": 115,
                #         "message": f"Failed while extracting documents form numbers for mapping: {error}",
                #         "state": STATUS_FAILED
                #     })
                #     return metadata
                metadata["result"]["version"] = declaration_version
                
                metadata["result"]["declaration_metadata"].update(declaration_metadata if declaration_metadata else {})
                if error:
                    metadata["status"].update({
                        "code": [118],
                        "message": f"Failed to extract declaration metadata: {error}",
                        "state": STATUS_FAILED
                    })
                    return metadata
            if not policy_name:
                print("POLICY NAME:", policy_name)
                policy_name, error = vision_extractor(pdf_path=pdf_file_path, extraction_type="policy_name")
                print("POLICY NAME:", policy_name, error)
                if policy_name:
                    metadata["policy_name"] = policy_name
                else:
                    metadata["status"].update({
                        "code": [118],
                        "message": f"Failed to extract policy name for B2B: {error}",
                        "state": STATUS_FAILED
                    })
                    return metadata
            chunks_obj = get_chunks_obj(full_text)
            metadata["result"]["chunks_obj"] = chunks_obj
            metadata["status"].update({
                "code": [],
                "message": None,
                "state": STATUS_PROCESSED
            })
            print("Item processed successfully...")
            return metadata
        except Exception as e:
            err_msg = get_traceback(e, "Failed to create the mongo objs for the chunks:")
            metadata["status"].update({
                "code": [121],
                "message": err_msg,
                "state": STATUS_FAILED
            })
            return metadata
    except Exception as e:
        err_msg  = get_traceback(e, "Failed while processing document in item processor.")
        metadata["status"].update({
                "code": [114],
                "message": err_msg,
                "state": STATUS_FAILED
            })
        return metadata
