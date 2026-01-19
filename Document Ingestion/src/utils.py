import os
import re
import shutil
import traceback
from datetime import datetime, timezone
from zipfile import ZipFile

import fitz  # PyMuPDF
from PIL import Image
import pdfplumber
from collections import defaultdict
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
from reportlab.lib.utils import ImageReader

from azure.identity import (
    DefaultAzureCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from azure.storage.blob import BlobServiceClient, ContentSettings
from motor.motor_asyncio import AsyncIOMotorClient
from openai import AzureOpenAI
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from src import (
    ACCOUNT_URL,
    ADOBE_RESULT_FILENAME,
    AZURE_OPENAI_SERVICE,
    ENV_NAME,
    MONGO_CONNECTION_STRING,
    MONGO_DB,
    ENV_NAMES,
    MERGED_DIR
)
from monads.monad_class import monad_wrapper, async_monad_wrapper

import os

MONGO_DOCUMENT_COLLECTION = os.getenv('MONGO_DOCUMENT_COLLECTION')

if ENV_NAME in ENV_NAMES:
    managed_credential = ManagedIdentityCredential()
else:
    managed_credential = DefaultAzureCredential()

mgclient = MongoClient(MONGO_CONNECTION_STRING)
amgclient = AsyncIOMotorClient(MONGO_CONNECTION_STRING)
adb = amgclient[MONGO_DB]
db = mgclient[MONGO_DB]

blob_service_client = BlobServiceClient(account_url=ACCOUNT_URL, credential=managed_credential)


@monad_wrapper
def openai_client(ENV):
    if ENV_NAME in ENV_NAMES:
        azure_credential = ManagedIdentityCredential()
    else:
        azure_credential = DefaultAzureCredential()

    token_provider = get_bearer_token_provider(azure_credential, "https://cognitiveservices.azure.com/.default")
    client = AzureOpenAI(
        azure_endpoint=f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com",
        api_version="2024-02-15-preview",
        azure_ad_token_provider=token_provider,
    )
    return client, token_provider


@monad_wrapper
def download_blob(blobname, input_dir, container_name, save_as=None):
    """
    Download a blob from Azure storage
    
    Args:
        blobname: Name of the blob to download
        input_dir: Directory to save the downloaded file
        container_name: Azure container name/path
        save_as: Optional filename to save the blob as (if different from blobname)
    """
    blob_client_instance = blob_service_client.get_blob_client(container_name, blobname, snapshot=None)
    blob_data = blob_client_instance.download_blob()
    
    # Use save_as if provided, otherwise use original blobname
    output_filename = save_as if save_as is not None else blobname
    
    with open(file=os.path.join(input_dir, output_filename), mode="wb") as sample_blob:
        sample_blob.write(blob_data.readall())


@monad_wrapper
def upload_blob(blobname, filepath, container_name, content_type="application/json"):
    blob_client_instance = blob_service_client.get_blob_client(container=container_name, blob=blobname)
    content_settings = ContentSettings(content_type=content_type)
    with open(filepath, "rb") as data:
        blob_client_instance.upload_blob(data, content_settings=content_settings, overwrite=True)


@monad_wrapper
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


@monad_wrapper
def create_document(doc_id, doc_name, carrier_name, doc_type):
    doc_obj = {
        "UserID": None,
        "ID": doc_id,
        "DocumentName": doc_name,
        "CarrierName": carrier_name,
        "DocumentVersion": None,
        "DocumentPath": None,
        "DocumentType": doc_type,
        "Status": None,
        "AdobeExtractPath": None,
        "UploadedAt": None,
        "ProcessedAt": None,
        "JsonData": None,
        "MetaData": None,
        "WaitingList": None,
        "RetryCount": 0,
        "FailureReason": None,
        # "PageNumber": None,
    }
    return doc_obj


@monad_wrapper
def create_policy(policy_number, carriername, base_doc_id=None, endo_doc_id=None, dec_doc_id=None, form_number=None, normalized_form_number=None):
    policy_obj = {
        "PolicyNumber": policy_number,
        "DeclarationNumber": None,
        "PolicyName": None,
        "CarrierName": carriername,
        #"HolderName": None,
        "PolicyType": None,
        "AppliedEndorsement": None,
        #"StartDate": None,
        #"ExpiryDate": None,
        "BasePolicyDocumentID": base_doc_id,
        "EndorsementDocumentID": [],  # Removed list from here
        "DeclarationDocumentID": dec_doc_id,
        "CreatedDate": datetime.now(),
        "IsActive": True,
    }
    return policy_obj


@monad_wrapper
def create_declaration(dec_doc_id, carriername, sample_declaration=False):
    dec_obj = {
        "PolicyName": None,
        "CarrierName": carriername,
        "DeclarationNumber": None,
        "ExtractedFormNumber": [
            {"FormNumber": None, "NormalizedFormNumber": None},
        ],
        "AppliedEndorsement": [],
        "HolderName": None,
        "StartDate": None,
        "ExpiryDate": None,
        "DeclarationDocumentID": [dec_doc_id],
        "CreatedDate": datetime.now(),
        "ProcessedFormat": "A",
        "SampleDeclaration": sample_declaration,
        "Version": 1
    }
    return dec_obj

@monad_wrapper
def create_merge(doc_id, 
                 doc_name, 
                 file_path,
                 uploaded_at,
                 carriername,
                 is_active=True):
    merge_obj = {
        "uniqueId": None,
        "documentId": doc_id,
        "userId": None,
        "carrierName": carriername,
        "documentTag": None,
        "documentName": doc_name,
        "filePath": file_path,
        "adobeExtractPath": None,
        "uploadDate": uploaded_at,
        "processedDate": None,
        "processStatus": None,
        "failureReason": None,
        "failureCode": None,
        "isActive": is_active,
        "splitDocumentList": None
    }
    return merge_obj


@monad_wrapper
def to_mongo(mongo_dict, mongo_collection):
    target_collection = db[mongo_collection]
    try:
        result = target_collection.insert_many(mongo_dict)
        inserted_ids = result.inserted_ids
        print("Inserted documents with the following IDs:")
        for inserted_id in inserted_ids:
            print(inserted_id)
    except DuplicateKeyError as e:
        print("Duplicate key error occurred:", str(e))
    except Exception as e:
        print("An error occurred while inserting documents:", str(e))
    print("Upserting of documents done!")


@async_monad_wrapper
async def asyto_mongo(mongo_dict, mongo_collection):
    """
    Asynchronously insert multiple documents into MongoDB using Motor.

    Args:
        mongo_dict: List of documents to insert
        mongo_collection: Name of the collection to insert into

    Returns:
        None
    """
    target_collection = adb[mongo_collection]
    try:
        result = await target_collection.insert_many(mongo_dict)
        inserted_ids = result.inserted_ids
        print("Inserted documents with the following IDs:")
        for inserted_id in inserted_ids:
            print(inserted_id)
    except Exception as e:
        print("An error occurred while inserting documents:", str(e))
    print("Upserting of documents done!")


@async_monad_wrapper
async def validate_doc_id(doc_id,carriername ,collection):
    """
    Validates if a document ID exists in the specified collection
    Returns (exists: bool, is_valid: bool)
    """
    if not doc_id:
        return False, False
    try:
        doc = await adb[collection].find_one({"ID": doc_id, "CarrierName" :carriername})
        if not doc:
            return False, False
        return True, True
    except Exception:
        return False, False

async def check_form_number_existskey(doc_id, collection):
    """
    Checks if a document has FormNumberExist field and returns its value
    Returns the value of FormNumberExist (defaults to False if not found)
    """
    if not doc_id:
        return False
    try:
        doc = await adb[collection].find_one({"ID": doc_id})
        if not doc:
            return False
        return doc.get("FormNumberExist", True)
    except Exception:
        return False

async def update_status_mongo_async(query, update_doc, mongo_collection, arr_filter=None):
    """
    Asynchronously updates documents in a MongoDB collection using Motor.
    
    Args:
        query (dict): The query to find documents to update.
        update_doc (dict): The update operations to perform. If the keys do not start with '$',
                          the document will be wrapped in {"$set": ...}.
        mongo_collection (str): Name of the collection to update.
        arr_filter (dict, optional): Array filters for updating array elements.
    
    Returns:
        UpdateResult: Result of the update operation.
    """
    target_collection = adb[mongo_collection]
    
    # If update_doc does not contain any update operators, wrap it in "$set"
    if not any(key.startswith("$") for key in update_doc):
        update_doc = {"$set": update_doc}
    
    try:
        if arr_filter:
            result = await target_collection.update_one(
                query, 
                update_doc,
                array_filters=[arr_filter]
            )
        else:
            result = await target_collection.update_one(
                query, 
                update_doc
            )
        
        return result
    except Exception as e:
        raise        


@monad_wrapper
def update_status_mongo(query, mongo_dict, mongo_collection, Success=False, arr_filter=None):
    target_collection = db[mongo_collection]
    try:
        # First, check if the document exists and get its current state
        current_doc = target_collection.find_one(query)
        
        update_operation = {}
        set_fields = {}
        
        # Handle Success=True case
        if Success:
            set_fields["FailureStatusCode"] = []
            set_fields["FailureReason"] = None
            update_operation["$set"] = set_fields
        
        else:
            # Process non-FailureStatusCode fields
            for key, value in mongo_dict.items():
                if key != "FailureStatusCode":
                    set_fields[key] = value
            
            if set_fields:
                update_operation["$set"] = set_fields
            
            # Handle FailureStatusCode separately
            if "FailureStatusCode" in mongo_dict:
                failure_code = mongo_dict["FailureStatusCode"]
                
                # If document doesn't exist or FailureStatusCode is None/not an array
                if not current_doc or not isinstance(current_doc.get("FailureStatusCode"), list):
                    # Set the array to just this one value
                    # If you want it sorted even at this point (though it's a single element),
                    # you can directly set it; there's nothing to sort with one element anyway.
                    update_operation.setdefault("$set", {})
                    update_operation["$set"]["FailureStatusCode"] = [failure_code]
                else:
                    # If the failure_code is not already in the array, push and sort
                    if failure_code not in current_doc["FailureStatusCode"]:
                        # Use $push with $sort to ensure ascending order
                        update_operation["$push"] = {
                            "FailureStatusCode": {
                                "$each": [failure_code],
                                "$sort": 1  # 1 for ascending order
                            }
                        }

        # Execute update
        if arr_filter:
            result = target_collection.update_one(
                query, 
                update_operation,
                array_filters=[arr_filter]
            )
        else:
            result = target_collection.update_one(
                query, 
                update_operation,
                upsert=True  # Allow creation of new documents
            )
            
        print("Updated documents with the following IDs:")
        print(result)
        
    except Exception as e:
        print("An error occurred while updating documents:", str(e))


@monad_wrapper
def get_document(query, mongo_collection):
    collection = db[mongo_collection]
    document = collection.find_one(query)
    return document

async def get_document_async(query, mongo_collection: str):
    """
    Fetches a document from a MongoDB collection asynchronously using Motor.

    :param query: The query to find the document.
    :param mongo_collection: The name of the MongoDB collection.
    :return: The document matching the query or None if not found.
    """
    collection = adb[mongo_collection]
    document = await collection.find_one(query)
    return document


# @monad_wrapper
# def extract_zip(filepath="./adobe/"):
#     try:
#         with ZipFile(filepath + ADOBE_RESULT_FILENAME + ".zip", "r") as zObject:
#             zObject.extractall(path=filepath + ADOBE_RESULT_FILENAME)

#         shutil.move(filepath + ADOBE_RESULT_FILENAME + "/structuredData.json", filepath + "structuredData.json")
#     except Exception:
#         traceback.print_exc()


@monad_wrapper
def extractZipNew(zip_file_path):
    # Get the directory of the zip file
    dir_name = os.path.dirname(zip_file_path)

    # Create a new folder for extracted files
    extract_folder = os.path.join(dir_name, "extractedAdobeRes")
    os.makedirs(extract_folder, exist_ok=True)

    # Extract the zip file
    with ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extract_folder)

    print(f"Extracted files to: {extract_folder}")


# @monad_wrapper
# def check_in_mongo(query, mongo_collection):
#     """
#     Checks if documents matching the given query exist in the specified MongoDB collection.

#     Parameters:
#     - query: The MongoDB query to match documents.
#     - mongo_collection: The name of the MongoDB collection.

#     Returns:
#     - bool: True if matching documents are found, False otherwise.
#     """
#     target_collection = db[mongo_collection]  # Assumes `db` is a global variable for the MongoDB database

#     try:
#         # Check if any document matches the query
#         exists = target_collection.find_one(query) is not None
#         return exists
#     except Exception as e:
#         print("An error occurred while checking documents:", str(e))
#         return False


@monad_wrapper
def get_available_and_unavailable_form_numbers(query, collection_name, form_numbers, normalize_form_numbers):
    matching_documents = list(db[collection_name].find(query))

    found_normalize_form_numbers = set()
    for doc in matching_documents:
        base_policy = doc.get("BasePolicyDocumentID", {})
        if base_policy:
            if "DocumentNumber" in base_policy and base_policy["DocumentNumber"] in form_numbers:
                found_normalize_form_numbers.add(base_policy["DocumentNumber"])
            if "NormalizedDocumentNumber" in base_policy and base_policy["NormalizedDocumentNumber"] in normalize_form_numbers:
                found_normalize_form_numbers.add(base_policy["NormalizedDocumentNumber"])
        endorsement_docs = doc.get("EndorsementDocumentID", [])
        for endo_doc in endorsement_docs:
            if "FormNumber" in endo_doc and endo_doc["FormNumber"] in form_numbers:
                found_normalize_form_numbers.add(endo_doc["FormNumber"])
            if "NormalizedFormNumber" in endo_doc and endo_doc["NormalizedFormNumber"] in normalize_form_numbers:
                found_normalize_form_numbers.add(endo_doc["NormalizedFormNumber"])

    available_form_numbers = found_normalize_form_numbers
    unavailable_form_numbers = set(normalize_form_numbers) - found_normalize_form_numbers
    return available_form_numbers, unavailable_form_numbers


@monad_wrapper
def map_form_numbers_to_declaration(declaration_document_id, form_numbers, policy_collection_name, declaration_collection_name, document_collection_name, carrier_name):
    """
    Maps form numbers to either BasePolicyID or AppliedEndorsement and updates the declaration collection directly.
    Uses monad error wrapper for error propagation.

    Args:
        declaration_document_id: The ID of the declaration document to update.
        form_numbers: List of dictionaries containing FormNumber and NormalizedFormNumber.
        policy_collection_name: Name of the policies collection in MongoDB
        declaration_collection_name: Name of the declarations collection in MongoDB
        document_collection_name: Name of the documents collection in MongoDB
        carrier_name: Name of the carrier to filter documents and policies

    Returns:
        bool: True if mapping was successful, False if no updates were needed
    """
    print(f"Starting mapping for DeclarationDocumentID: {declaration_document_id} and CarrierName: {carrier_name}")

    # Get MongoDB collections
    policy_collection = db[policy_collection_name]
    declaration_collection = db[declaration_collection_name]
    document_collection = db[document_collection_name]

    # Initialize variables
    base_policy_id = None
    endorsement_document_ids = []
    missing_endorsements = []
    base_policy_form_found = False
    update_data = {}

    try:
        for form_entry in form_numbers:
            form_number = form_entry["FormNumber"]
            print(f"Processing FormNumber: {form_number}")

            # Query policy collection for matching form numbers with carrier filter
            policy_doc = policy_collection.find_one(
                {
                    "FormNumber": {"$regex": f".*{form_number}.*", "$options": "i"},
                    "CarrierName": carrier_name
                }
            )
            if policy_doc:
                print(f"Found policy document for FormNumber {form_number}: {policy_doc}")
            else:
                print(f"No policy document found for FormNumber {form_number}")

            # Query document collection for matching form numbers with carrier filter
            document_doc = document_collection.find_one(
                {
                    "FormNumber": {"$regex": f".*{form_number}.*", "$options": "i"},
                    "CarrierName": carrier_name
                }
            )
            if document_doc:
                print(f"Found document for FormNumber {form_number}: {document_doc}")
            else:
                print(f"No document found for FormNumber {form_number}")

            # Check if document is base policy in policy collection
            if policy_doc and policy_doc.get("BasePolicyDocumentID"):
                if base_policy_id and base_policy_id != policy_doc["BasePolicyDocumentID"]:
                    raise Exception(
                        f"Multiple base policy IDs found. Previous: {base_policy_id}, New: {policy_doc['BasePolicyDocumentID']}"
                    )
                base_policy_id = policy_doc["BasePolicyDocumentID"]
                base_policy_form_found = True

                # Set BasePolicy and PolicyName
                update_data.setdefault("$set", {})["BasePolicy"] = base_policy_id
                update_data.setdefault("$set", {})["PolicyName"] = policy_doc.get("PolicyName", "Unknown")
                update_data.setdefault("$set", {})["CarrierName"] = carrier_name
                print(f"Set BasePolicy to {base_policy_id}, PolicyName: {policy_doc.get('PolicyName', 'Unknown')}, and CarrierName: {carrier_name}")
                continue

            if not document_doc:
                missing_endorsements.append(form_number)
                print(f"Missing endorsement for FormNumber: {form_number}")
                continue
            else:
                # Check if the document is not a base policy before adding to endorsements
                if document_doc.get("DocumentType", "").lower() != "BasePolicy".lower():
                    endorsement_document_id = document_doc.get("ID")
                    endorsement_document_ids.append(endorsement_document_id)
                    print(f"Added endorsement DocumentID: {endorsement_document_id} for FormNumber: {form_number}")

        # Failure Scenarios with detailed error messages
        if not base_policy_form_found:
            raise Exception(
                f"No matching base policy found for form numbers: {[entry['FormNumber'] for entry in form_numbers]} and carrier: {carrier_name}"
            )

        if missing_endorsements:
            raise Exception(
                f"Required endorsements missing for form numbers: {missing_endorsements} and carrier: {carrier_name}"
            )

        # Add "AppliedEndorsement" with $addToSet if endorsement_ids exist
        if endorsement_document_ids:
            update_data.setdefault("$addToSet", {})["AppliedEndorsement"] = {"$each": endorsement_document_ids}
            print(f"AppliedEndorsement IDs to add: {endorsement_document_ids}")

        # Add ExtractedFormNumber to the update
        update_data.setdefault("$set", {})["ExtractedFormNumber"] = form_numbers
        print(f"Added ExtractedFormNumber to update: {form_numbers}")

        # Wrap DeclarationDocumentID in an array
        declaration_document_id_array = [declaration_document_id]
        print(f"Wrapped DeclarationDocumentID into an array: {declaration_document_id_array}")

        try:
            # Perform the MongoDB update
            update_result = declaration_collection.update_one(
                {"DeclarationDocumentID": declaration_document_id_array},
                update_data
            )

            if update_result.modified_count == 0:
                # Check if the document exists but no changes were needed
                existing_doc = declaration_collection.find_one({"DeclarationDocumentID": declaration_document_id_array})
                if existing_doc:
                    print(f"No modifications needed for DeclarationDocumentID: {declaration_document_id} - Document already up to date")
                    return False
                else:
                    raise Exception(f"Declaration document not found for ID: {declaration_document_id}")

            print(f"Mapping completed successfully for DeclarationDocumentID: {declaration_document_id}")
            return True

        except Exception as update_error:
            print(f"Error during MongoDB update: {str(update_error)}")
            # Update document status to reflect the error
            try:
                error_update = {
                    "$set": {
                        "Status": "Failed",
                        "ProcessedAt": datetime.now()
                    }
                }
                document_collection.update_one(
                    {"ID": declaration_document_id}, 
                    error_update
                )
            except Exception as status_update_error:
                print(f"Failed to update error status: {str(status_update_error)}")
            
            raise Exception(f"MongoDB Update Error: {str(update_error)}")

    except Exception as e:
        # Log the error
        print(f"Mapping Error for DeclarationDocumentID {declaration_document_id}: {str(e)}")
        
        # Update document status to reflect the error
        try:
            error_update = {
                "$set": {
                    "Status": "Failed",
                    "ProcessedAt": datetime.now()
                }
            }
            document_collection.update_one(
                {"ID": declaration_document_id}, 
                error_update
            )
        except Exception as update_error:
            print(f"Failed to update error status: {str(update_error)}")
            
        # Raise the original error to be handled by the monad
        raise Exception(f"Mapping Error: {str(e)}")
    

# @monad_wrapper
# def match_document_number(document_number, form_numbers):
#     pattern = r"^(.*?) \(.*?\)$"
#     match = re.match(pattern, document_number)
#     if match:
#         document_number_without_date = match.group(1)
#     else:
#         document_number_without_date = document_number

#     # Search for the document number in the list of form numbers
#     for i, form_number in enumerate(form_numbers):
#         if form_number == document_number_without_date:
#             return i
#     return -1

def match_single_document_number(document_number, form_numbers):
    # Normalize both sides for comparison
    normalized_doc = normalize_pattern(document_number)
    
    for i, form_number in enumerate(form_numbers):
        normalized_form = normalize_pattern(form_number)
        if normalized_doc == normalized_form:
            return i
    return -1

def match_document_numbers(document_numbers, form_numbers):
    # Split document numbers by semicolon if present
    doc_numbers = [num.strip() for num in document_numbers.split(';')] if ';' in document_numbers else [document_numbers]
    
    # Find all matching indexes
    matched_indexes = []
    for doc_num in doc_numbers:
        index = match_single_document_number(doc_num, form_numbers)
        if index != -1:
            matched_indexes.append(index)
    
    return matched_indexes

@async_monad_wrapper
async def fetch_carriers_with_base_policy(collection_name):
    """
    Fetch distinct carrier names from policy collection where base policy exists.

    Args:
        collection_name: Name of the policy collection
        query: Optional additional query parameters

    Returns:
        List of carrier names that have base policies
    """

    query = {"BasePolicyDocumentID": {"$exists": True, "$ne": None}}

    try:
        # Assuming adb is a global Motor AsyncIOMotorDatabase instance
        carriers = await adb[collection_name].distinct("CarrierName", query)
        return carriers or []
    except Exception:
        return []

@async_monad_wrapper
async def log_ingestion_status_toDB(doc_id, status, error=None):
    """
    Logs the processing status of a document to the database.
    Args:
        doc_id (str): The ID of the document.
        status (str): The processing status ('Ingested' or 'Failed').
        error (str, optional): Error message if any.
    """
    query = {"ID": doc_id}
    update_fields = {
        "Status": status,
        "ProcessedAt": datetime.now(),
        "FailureReason": error if status == "Failed" else None
    }
    
    try:
        await adb[MONGO_DOCUMENT_COLLECTION].update_one(query, {'$set': update_fields})
        print(f"Updated document {doc_id} status to {status}")
    except Exception as e:
        print(f"Failed to update document {doc_id} status: {str(e)}")

@monad_wrapper
def update_endorsement_document_id(collection_name, document_id, carrier, split_indices_dict=None):
    """
    Updates the EndorsementDocumentID array by appending a new document ID for a given policy.
    """
    collection = db[collection_name]

    # Build the base query
    query = {
        "CarrierName": carrier,
        "IsActive": True,
    }

    # Add merged upload criteria if applicable
    if split_indices_dict and split_indices_dict.get("frontendmergedupload") == "Y":
        parent_id = document_id.split("_")[0]
        query["MergedDocumentID"] = parent_id

    # Perform the update
    result = collection.update_many(
        query,
        {
            "$push": {"EndorsementDocumentID": document_id},
            "$set": {
                "UpdatedDate": datetime.now(),
            },
        },
    )

    return result.modified_count > 0





def check_incomplete_unicode(pdf_path):
    """
    Check if a PDF has incomplete Unicode mappings by analyzing font information.
    
    Args:
        pdf_path (str): Path to the PDF file
    
    Returns:
        bool: True if incomplete Unicode mappings are detected, False otherwise
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Get fonts used in the page
                fonts = defaultdict(set)
                
                # Collect all unique fonts and their characters
                for char in page.chars:
                    font_name = char.get('fontname', '')
                    fonts[font_name].add(char.get('text', ''))
                
                # Check each font
                for font_name in fonts:
                    if not font_name:  # If font name is missing
                        return True
                        
                    # Check for non-unicode fonts (typically have names containing)
                    problematic_indicators = [
                        'Symbol', 
                        'ZapfDingbats',
                        'MT Extra',
                        'Wingdings',
                        'Type3'
                    ]
                    
                    if any(indicator in font_name for indicator in problematic_indicators):
                        return True
                        
                    # Check characters in this font
                    chars = fonts[font_name]
                    
                    # If we find any characters that encode to patterns containing '\u'
                    if any('\\u' in c.encode('unicode_escape').decode('ascii') for c in chars if c):
                        return True
                            
                    # Check for characters in Unicode private use areas
                    if any(any(ord(c) in range(0xE000, 0xF8FF + 1) for c in char) 
                           for char in chars if char):
                        return True
                    
            return False
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return True  # Return True to indicate potential issues if an error occurs
    

@monad_wrapper
def checkPDFencodingerrors(pdf_path):
    """
    Converts each page of a PDF into an image and compiles them back into a PDF.
    This function processes images in memory to avoid file access conflicts when used with multiprocessing.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        str: Path to the new PDF file with images
    """
    # Get the document directory from pdf_path
    doc_dir = os.path.dirname(pdf_path)
    
    # Check for Unicode issues using the new mechanism
    has_unicode_issues = check_incomplete_unicode(pdf_path)
    
    if not has_unicode_issues:
        return pdf_path
        
    try:
        # Open PDF
        pdf_document = fitz.open(pdf_path)
        
        # Create output filename in the same directory
        output_pdf = os.path.join(doc_dir, os.path.basename(pdf_path).rsplit('.', 1)[0] + '_images.pdf')
        
        # Create PDF from images
        c = canvas.Canvas(output_pdf, pagesize=letter)
        
        # Process each page
        for page_num in range(pdf_document.page_count):
            # Get the page
            page = pdf_document[page_num]
            
            # Convert page to image
            pix = page.get_pixmap(dpi=300)  # Adjust DPI as needed
            
            # Get image data as PNG bytes
            img_data = pix.tobytes("png")
            
            # Open image with PIL from bytes
            img = Image.open(BytesIO(img_data))
            img_width, img_height = img.size
            aspect = img_height / float(img_width)
            
            # Scale to fit page width
            width = letter[0]
            height = width * aspect
            
            # Add image to PDF
            img_reader = ImageReader(BytesIO(img_data))
            c.drawImage(img_reader, 0, letter[1] - height, width=width, height=height)
            c.showPage()
                
        c.save()
        pdf_document.close()
        
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        raise
        
    return output_pdf

async def update_endorsement_document_id_async(collection_name, endorsement_messages):
    """
    Updates EndorsementDocumentID array for all policies of matching carriers in batch.
    For frontend uploads (identified by split_indices_dict), updates only specific policies.
    Handles null values in EndorsementDocumentID by initializing empty list first.
    """
    collection = adb[collection_name]
    
    # Group endorsement messages by carrier
    carrier_endorsements = {}
    frontend_updates = []
    
    for msg in endorsement_messages:
        _, msg_data = msg
        carrier = msg_data[4]
        doc_id = msg_data[1]
        split_indices_dict = msg_data[8]
        
        if split_indices_dict.get("frontendmergedupload") == "Y":
            parent_id = doc_id.split("_")[0]
            frontend_updates.append({
                'carrier': carrier,
                'doc_id': doc_id,
                'parent_id': parent_id
            })
        else:
            if carrier not in carrier_endorsements:
                carrier_endorsements[carrier] = []
            carrier_endorsements[carrier].append(doc_id)

    # Initialize empty arrays for documents with null EndorsementDocumentID
    # for carrier in carrier_endorsements.keys():
    #     try:
    #         await collection.update_many(
    #             {
    #                 "CarrierName": carrier,
    #                 "IsActive": True,
    #                 "MergedDocumentID": {"$exists": False},
    #                 "EndorsementDocumentID": None
    #             },
    #             {
    #                 "$set": {
    #                     "EndorsementDocumentID": [],
    #                     "UpdatedDate": datetime.now()
    #                 }
    #             }
    #         )
    #     except Exception as e:
    #         raise

    # Process batch updates for each carrier
    for carrier, doc_ids in carrier_endorsements.items():
        try:
            await collection.update_many(
                {
                    "CarrierName": carrier,
                    "IsActive": True,
                    "MergedDocumentID": {"$exists": False}  # Exclude documents with MergedDocumentID
                },
                {
                    "$addToSet": {
                        "EndorsementDocumentID": {
                            "$each": doc_ids
                        }
                    },
                    "$set": {
                        "UpdatedDate": datetime.now()
                    }
                }
            )
        except Exception as e:
            raise

    # Initialize empty arrays for frontend updates with null EndorsementDocumentID
    # for update in frontend_updates:
    #     try:
    #         await collection.update_many(
    #             {
    #                 "CarrierName": update['carrier'],
    #                 "IsActive": True,
    #                 "MergedDocumentID": update['parent_id'],
    #                 "EndorsementDocumentID": None
    #             },
    #             {
    #                 "$set": {
    #                     "EndorsementDocumentID": [],
    #                     "UpdatedDate": datetime.now()
    #                 }
    #             }
    #         )
    #     except Exception as e:
    #         raise

    # Process frontend updates
    for update in frontend_updates:
        try:
            await collection.update_one(
                {
                    "CarrierName": update['carrier'],
                    "IsActive": True,
                    "MergedDocumentID": update['parent_id']
                },
                {
                    "$addToSet": {"EndorsementDocumentID": update['doc_id']},
                    "$set": {"UpdatedDate": datetime.now()}
                }
            )
        except Exception as e:
            raise

    return True

def merge_declarations(data_dict, order_dict):
    """
    Merges multiple declarations into one based on the provided order dictionary.
    Ensures all splits and page spans are in list format regardless of merging status.

    Args:
        data_dict (dict): Dictionary containing the document information
        order_dict (dict): Dictionary containing the ordering information with declaration indexes

    Returns:
        dict: Updated data dictionary with merged declarations
    """
    # Create a copy of the input dictionary to avoid modifying the original
    result_dict = data_dict.copy()

    # First pass: ensure all splits and page_num are in list format
    for key in result_dict:
        if not isinstance(result_dict[key]['split'], list):
            result_dict[key]['split'] = [result_dict[key]['split']]
        if not isinstance(result_dict[key]['page_num'], list):
            result_dict[key]['page_num'] = [result_dict[key]['page_num']]

    # Extract the document ID prefix from the first key
    doc_id = next(iter(data_dict)).split('_')[0]

    # Process each key in the order dictionary
    for key, value in order_dict.items():
        if isinstance(value, list) and len(value) > 1:
            # Get the primary index (first in the list)
            primary_key = f'{doc_id}_{value[0]}'

            # Initialize merged text, splits, and page spans
            merged_text = result_dict[primary_key]['text']
            merged_splits = result_dict[primary_key]['split']
            merged_pages = result_dict[primary_key]['page_num']

            # Merge additional declarations
            for idx in value[1:]:
                secondary_key = f'{doc_id}_{idx}'
                if secondary_key in result_dict:
                    # Concatenate text with semicolon separator
                    merged_text += '; ' + result_dict[secondary_key]['text']
                    # Extend splits and page spans
                    merged_splits.extend(result_dict[secondary_key]['split'])
                    merged_pages.extend(result_dict[secondary_key]['page_num'])
                    # Remove the secondary declaration from the result
                    del result_dict[secondary_key]

            # Update the primary declaration with merged text, splits, and page spans
            result_dict[primary_key]['text'] = merged_text
            result_dict[primary_key]['split'] = merged_splits
            result_dict[primary_key]['page_num'] = merged_pages

    return result_dict

def convert_to_split_dict(input_data):
    """
    Converts the input data format to the required split dictionary format.
    
    Args:
        input_data (dict): Input data containing DocID and splitList
        
    Returns:
        dict: Split dictionary in the format {doc_id_index: {text, split, page_num}}
    """
    split_dict = {}
    doc_id = input_data["doc_id"]
    
    for idx, split_info in enumerate(input_data["splitList"]):
        key = f"{doc_id}_{idx}"
        split_dict[key] = {
            "text": split_info["formNumber"],
            "page_num": split_info["pageSpan"][1],  # Using end page as page_num
            "split": None  # Will be populated by get_element_spans function
        }
    
    return split_dict

def get_ordering_key(input_data):
    """
    Creates ordering key dictionary based on document types in the input.
    Now handles multiple declarations by storing their indices in a list.
    
    Args:
        input_data (dict): Input data containing DocID and splitList
        
    Returns:
        dict: Ordering key with Base index and list of Declaration indices
    """
    ordering_key = {"Base": None, "Declaration": []}
    
    for idx, split_info in enumerate(input_data["splitList"]):
        if split_info["documentType"] == "Base":
            ordering_key["Base"] = idx
        elif split_info["documentType"] == "Declaration":
            ordering_key["Declaration"].append(idx)
    
    return ordering_key

def get_element_spans(page_span, elements_list):
    """
    Determines the element index spans based on page numbers from Adobe JSON response.
    
    Args:
        page_span (tuple): Tuple of (start_page, end_page)
        elements_list (list): List of dictionaries from Adobe JSON response 'elements' key
        
    Returns:
        tuple: Start and end indices for the elements corresponding to the page span
    """
    start_page, end_page = page_span
    start_idx = None
    end_idx = None
    
    for idx, element in enumerate(elements_list):
        page = element.get('Page')
        if page is None:
            continue
            
        # Find first element of start page
        if page == start_page and start_idx is None:
            start_idx = idx
            
        # Keep updating end_idx as long as we're on the end page
        # This will give us the last element of the end page
        if page == end_page:
            end_idx = idx
            
    # Default to first/last element if pages not found
    return (start_idx or 0, end_idx or len(elements_list) - 1)

def process_document_splitsforapi(input_data, elements_list):
    """
    Main function to process document splits and create required data structures.
    
    Args:
        input_data (dict): Input data containing DocID and splitList
        elements_list (list): List of dictionaries from Adobe JSON response 'elements' key
        
    Returns:
        tuple: (split_dict, ordering_key)
    """
    # Create initial split dictionary
    split_dict = convert_to_split_dict(input_data)
    
    # Get ordering key with declarations in a list
    ordering_key = get_ordering_key(input_data)
    
    # Update split ranges in split_dict
    for idx, split_info in enumerate(input_data["splitList"]):
        key = f"{input_data['doc_id']}_{idx}"
        span_indices = get_element_spans(split_info["pageSpan"], elements_list)
        split_dict[key]["split"] = span_indices
    
    return split_dict, ordering_key

def convert_date_format(date_str):
    """
    Convert date from MM/DD/YYYY format to ISO format with timezone,
    setting time to UTC midnight (start of the day)
    
    Args:
        date_str (str): Date string in format "MM/DD/YYYY"
        
    Returns:
        datetime: Datetime object at UTC midnight
    """
    # Parse the input date
    local_date = datetime.strptime(date_str, "%m/%d/%Y")
    
    # Create UTC datetime at midnight (00:00:00)
    return datetime.combine(
        local_date.date(),
        datetime.min.time(),  # This gives 00:00:00
        tzinfo=timezone.utc
    )


def cleanup_error_entries(doc_id: str, collection_name: str) -> None:
    """
    Deletes error entries for the document ID if they exist.
    
    Args:
        doc_id (str): The document ID to check and potentially delete
        collection_name (str): Name of the MongoDB collection to operate on
    """
    try:
        existing_doc = db[collection_name].find_one({"ID": doc_id})
        
        if existing_doc and existing_doc.get("Status") == "Failed":
            db[collection_name].delete_one({"ID": doc_id})
    except Exception:
        pass


def fetch_document_config(
    doc_id: str,
    document_collection_name: str,
    client_collection_name: str,
    regex_collection_name: str
) -> dict:
    """
    Fetch document configuration from MongoDB based on document ID.
    
    Args:
        doc_id (str): Document ID to lookup
        document_collection_name (str): Name of the documents collection
        client_collection_name (str): Name of the client collection
        regex_collection_name (str): Name of the regex collection
        
    Returns:
        dict: Configuration object for the document
        
    Raises:
        ValueError: If document not found, client not found, or invalid config
    """
    try:
        # Get document collection
        doc_collection = db[document_collection_name]
        
        # Find document
        document = doc_collection.find_one({"ID": doc_id})
        if not document:
            raise ValueError(f"Document not found with ID: {doc_id}")
            
        # Get carrier name
        carrier_name = document.get("CarrierName")
        if not carrier_name:
            raise ValueError(f"CarrierName not found in document: {doc_id}")
            
        # Get client collection
        client_collection = db[client_collection_name]
        
        # Find client config using carrier name
        client_config = client_collection.find_one({"ClientName": carrier_name})
        if not client_config:
            raise ValueError(f"Client config not found for carrier: {carrier_name}")
            
        # Get ConfigID
        config_id = client_config.get("ConfigID")
        if not config_id:
            raise ValueError(f"ConfigID not found for client: {carrier_name}")
            
        # Get regex collection
        regex_collection = db[regex_collection_name]
        
        # Find regex config using ConfigID
        regex_config = regex_collection.find_one({"ConfigID": config_id})
        if not regex_config:
            raise ValueError(f"Regex config not found for ConfigID: {config_id}")
            
        # Get ConfigPattern
        config_pattern = regex_config.get("ConfigPattern")
        if not config_pattern:
            raise ValueError(f"ConfigPattern not found for ConfigID: {config_id}")
            
        # Validate config has required fields
        required_fields = ['Mode', 'FormNumberPattern']
        missing_fields = [field for field in required_fields if field not in config_pattern]
        if missing_fields:
            raise ValueError(f"Config pattern missing required fields: {', '.join(missing_fields)}")
            
        # Validate Mode
        if config_pattern['Mode'] not in ['Table', 'Text']:
            raise ValueError("Invalid Mode value. Must be either 'Table' or 'Text'")
            
        # Mode-specific validation
        if config_pattern['Mode'] == 'Table':
            if 'TableConfig' not in config_pattern:
                raise ValueError("TableConfig is required when Mode is 'Table'")
                
            table_config = config_pattern['TableConfig']
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
                raise ValueError(f"TableConfig missing required fields: {', '.join(missing_table_fields)}")
                
            if not isinstance(table_config['FileExtensions'], list):
                raise ValueError("FileExtensions must be an array")
                
        # Text mode no longer requires additional configuration
        
        return config_pattern
        
    except Exception as e:
        # Log the error with traceback
        err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
        raise    


async def check_document_completion(doc_id: str, collection_name: str) -> bool:
    """
    Check if a document is completely processed.
    Uses global adb for MongoDB operations.
    """
    try:
        collection = adb[collection_name]
        doc = await collection.find_one(
            {"ID": doc_id},
            {"Status": 1, "FailureStatusCode": 1, "FailureReason": 1}
        )
        
        if not doc:
            return False
            
        return (doc.get("Status") in ["Ingested", "Processed"] and 
                not doc.get("FailureStatusCode", []) and 
                doc.get("FailureReason") is None)
    except Exception as e:
        #app_insight_logger.error(f"Error in check_document_completion for {doc_id}: {str(e)}")
        return False

async def check_merged_constituents(base_doc_id: str, collection_name: str) -> bool:
    """
    Check if all constituents of a merged document are processed.
    Uses global adb for MongoDB operations.
    """
    try:
        collection = adb[collection_name]
        # Find all constituent documents (pattern: base_doc_id_number)
        constituents = collection.find({
            "ID": {"$regex": f"^{base_doc_id}_[0-9]+$"},
            # "Status": 1, 
            # "FailureStatusCode": 1,
            # "FailureReason": 1
        })
        
        constituents_list = await constituents.to_list(length=None)
        if not constituents_list:
            return False
        
        return all(
            doc.get("Status") in ["Ingested", "Processed"] and
            not doc.get("FailureStatusCode", []) and
            doc.get("FailureReason") is None
            for doc in constituents_list
        )
    except Exception as e:
        #app_insight_logger.error(f"Error in check_merged_constituents for {base_doc_id}: {str(e)}")
        return False


def check_form_number_exists(normalized_form_number: str, document_collection: str, carrier_name: str, form_id: str) -> tuple[bool, str]:
    """
    Checks if form number exists in specified collection, excluding the document with the given ID
    
    Args:
        normalized_form_number (str): The normalized form number to check
        document_collection (str): The name of the MongoDB collection
        carrier_name (str): The name of the carrier
        form_id (str): The ID of the current form to exclude from the check
    
    Returns:
        tuple[bool, str]: (exists: bool, error: str)
    """
    try:
        query = {
            "NormalizedFormNumber": normalized_form_number,
            "CarrierName": carrier_name,
            "ID": {"$ne": form_id}  # Exclude the document with the given ID
        }
        result = db[document_collection].find_one(query)
        return True if result else False, ""
    except Exception as e:
        return False, f"Error checking form number: {str(e)}"

def cleanup_document_entries(doc_id: str, folder_name: str, policy_collection: str) -> tuple[bool, str]:
    """
    Cleans up document entries from policy collection for base policy documents
    Args:
        doc_id: Document ID to clean up
        folder_name: Folder name to determine document type
        policy_collection: Policy collection name
    Returns (success: bool, error: str)
    """
    try:
        if "base" in folder_name.lower():
            db[policy_collection].delete_one({"BasePolicyDocumentID": doc_id})
        
        return True, ""
        
    except Exception as e:
        return False, f"Error in cleanup: {str(e)}"
    
def pull_carrier_config(
    carrier_name: str,
    client_collection_name: str,
    regex_collection_name: str
) -> dict:
    """
    Fetch document configuration from MongoDB based on carrier name using pymongo.
    
    Args:
        carrier_name (str): Name of the carrier
        client_collection_name (str): Name of the client collection
        regex_collection_name (str): Name of the regex collection
        
    Returns:
        dict: Configuration pattern object for the carrier
        
    Raises:
        ValueError: If client not found or invalid config
    """
    try:
        # Get client collection
        client_collection = db[client_collection_name]
        
        # Find client config using carrier name
        client_config = client_collection.find_one({"ClientName": carrier_name})
        if not client_config:
            raise ValueError(f"Client config not found for carrier: {carrier_name}")
            
        # Get ConfigID
        config_id = client_config.get("ConfigID")
        if not config_id:
            raise ValueError(f"ConfigID not found for client: {carrier_name}")
            
        # Get regex collection
        regex_collection = db[regex_collection_name]
        
        # Find regex config using ConfigID
        regex_config = regex_collection.find_one({"ConfigID": config_id})
        if not regex_config:
            raise ValueError(f"Regex config not found for ConfigID: {config_id}")
            
        # Get ConfigPattern
        config_pattern = regex_config.get("ConfigPattern")
        if not config_pattern:
            raise ValueError(f"ConfigPattern not found for ConfigID: {config_id}")
            
        # Validate config has required fields
        required_fields = ['Mode', 'FormNumberPattern']
        missing_fields = [field for field in required_fields if field not in config_pattern]
        if missing_fields:
            raise ValueError(f"Config pattern missing required fields: {', '.join(missing_fields)}")
            
        # Validate Mode
        if config_pattern['Mode'] not in ['Table', 'Text']:
            raise ValueError("Invalid Mode value. Must be either 'Table' or 'Text'")
            
        # Mode-specific validation
        if config_pattern['Mode'] == 'Table':
            if 'TableConfig' not in config_pattern:
                raise ValueError("TableConfig is required when Mode is 'Table'")
                
            table_config = config_pattern['TableConfig']
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
                raise ValueError(f"TableConfig missing required fields: {', '.join(missing_table_fields)}")
                
            if not isinstance(table_config['FileExtensions'], list):
                raise ValueError("FileExtensions must be an array")
                
        # Text mode no longer requires additional configuration
        
        return config_pattern
        
    except Exception as e:
        # Log the error with traceback
        err_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
        raise


def get_highest_declaration_version(policy_number: str, collection_name: str, carrier_name: str) -> int:
    """
    Looks up existing declarations with the same policy number and returns the highest version.
    
    Args:
        policy_number: The policy number to search for
        collection_name: Name of the collection to search in
    
    Returns:
        int: The highest version number found + 1, or 1 if no matches or no versions found
    """
    try:
        # Find all documents with matching policy number
        matches = list(db[collection_name].find({"DeclarationNumber": policy_number, "CarrierName": carrier_name})) #Added filter for carrier name
        
        if not matches:
            return 1
        
        # Collect all versions safely
        versions = []
        for doc in matches:
            try:
                version = doc.get('Version')
                if version is not None and isinstance(version, (int, float)):
                    versions.append(int(version))
                else:
                    versions.append(1)
            except Exception as e:
                versions.append(1)
        
        # Get highest version
        highest_version = max(versions) if versions else 1
        
        return highest_version + 1
        
    except Exception as e:
        # In case of any error, safely default to version 1
        return 1
    
def remove_spaces(text: str) -> str:
    """
    Remove all whitespace characters from a given string.
    
    Args:
        text (str): Input string containing whitespace
        
    Returns:
        str: String with all whitespace characters removed
        
    Examples:
        >>> remove_spaces("hello   world")
        'helloworld'
        >>> remove_spaces("multiple   spaces   and\ttabs  \n")
        'multiplespacestandtabs'
    """
    if not text:
        return text
    return ''.join(text.split())


async def check_form_number_exists_async(normalized_form_number: str, document_collection: str, carrier_name: str, form_id: str) -> tuple[bool, str]:
    """
    Checks if form number exists in specified collection, excluding the document with the given ID
    
    Args:
        normalized_form_number (str): The normalized form number to check
        document_collection (str): The name of the MongoDB collection
        carrier_name (str): The name of the carrier
        form_id (str): The ID of the current form to exclude from the check
    
    Returns:
        tuple[bool, str]: (exists: bool, error: str)
    """
    try:
        query = {
            "NormalizedFormNumber": normalized_form_number,
            "CarrierName": carrier_name,
            "ID": {"$ne": form_id}  # Exclude the document with the given ID
        }
        result = await adb[document_collection].find_one(query)
        return True if result else False, ""
    except Exception as e:
        return False, f"Error checking form number: {str(e)}"

async def cleanup_document_entries_async(doc_id: str, folder_name: str, policy_collection: str) -> tuple[bool, str]:
    """
    Cleans up document entries from policy collection for base policy documents
    Args:
        doc_id: Document ID to clean up
        folder_name: Folder name to determine document type
        policy_collection: Policy collection name
    Returns (success: bool, error: str)
    """
    try:
        if "base" in folder_name.lower():
            await adb[policy_collection].delete_one({"BasePolicyDocumentID": doc_id})
        
        return True, ""
        
    except Exception as e:
        return False, f"Error in cleanup: {str(e)}"
    
def normalize_pattern(s: str) -> str:
    """Remove spaces, parentheses, and other special characters."""
    original = s
    normalized = re.sub(r"[ ()-/]", "", s)
    print(f"DEBUG normalize_pattern: {original} -> {normalized}")
    return normalized    
    
async def map_form_numbers_to_declaration_async(declaration_document_id, form_numbers, policy_collection_name, declaration_collection_name, document_collection_name, carrier_name):
    """
    Maps form numbers to either BasePolicyID or AppliedEndorsement using normalized form numbers for matching.
    """
    print(f"Starting mapping for DeclarationDocumentID: {declaration_document_id} and CarrierName: {carrier_name}")

    base_policy_id = None
    endorsement_document_ids = []
    missing_endorsements = []
    base_policy_form_found = False
    update_data = {}

    try:
        for form_entry in form_numbers:
            normalized_form = normalize_pattern(form_entry["FormNumber"])  # Normalize the form number
            print(f"Processing Normalized FormNumber: {normalized_form}")

            # Query policy collection for normalized form number with regex match
            policy_doc = await adb[policy_collection_name].find_one(
                {
                    "NormalizedFormNumber": {"$regex": f".*{normalized_form}.*", "$options": "i"},
                    "CarrierName": carrier_name
                }
            )

            # Query document collection for normalized form number with regex match
            document_doc = await adb[document_collection_name].find_one(
                {
                    "NormalizedFormNumber": {"$regex": f".*{normalized_form}.*", "$options": "i"},
                    "CarrierName": carrier_name
                }
            )

            # Rest of the logic remains the same...
            if policy_doc and policy_doc.get("BasePolicyDocumentID"):
                if base_policy_id and base_policy_id != policy_doc["BasePolicyDocumentID"]:
                    raise Exception(
                        f"Multiple base policy IDs found. Previous: {base_policy_id}, New: {policy_doc['BasePolicyDocumentID']}"
                    )
                base_policy_id = policy_doc["BasePolicyDocumentID"]
                base_policy_form_found = True

                update_data.setdefault("$set", {})["BasePolicy"] = base_policy_id
                update_data.setdefault("$set", {})["PolicyName"] = policy_doc.get("PolicyName", "Unknown")
                update_data.setdefault("$set", {})["CarrierName"] = carrier_name
                continue

            if not document_doc:
                missing_endorsements.append(form_entry["FormNumber"])  # Keep original form number for error message
                continue
            else:
                if document_doc.get("DocumentType", "").lower() != "BasePolicy".lower():
                    endorsement_document_id = document_doc.get("ID")
                    endorsement_document_ids.append(endorsement_document_id)

        if not base_policy_form_found:
            raise Exception(
                f"No matching base policy found for form numbers: {[entry['FormNumber'] for entry in form_numbers]} and carrier: {carrier_name}"
            )

        if missing_endorsements:
            raise Exception(
                f"Required endorsements missing for form numbers: {missing_endorsements} and carrier: {carrier_name}"
            )

        if endorsement_document_ids:
            update_data.setdefault("$addToSet", {})["AppliedEndorsement"] = {"$each": endorsement_document_ids}

        # Add ExtractedFormNumber to the update
        update_data.setdefault("$set", {})["ExtractedFormNumber"] = form_numbers

        # Perform the MongoDB update
        declaration_document_id_array = [declaration_document_id]
        update_result = await adb[declaration_collection_name].update_one(
            {"DeclarationDocumentID": declaration_document_id_array},
            update_data
        )

        if update_result.modified_count == 0:
            existing_doc = await adb[declaration_collection_name].find_one(
                {"DeclarationDocumentID": declaration_document_id_array}
            )
            if existing_doc:
                return False
            raise Exception(f"Declaration document not found for ID: {declaration_document_id}")

        return True

    except Exception as e:
        raise Exception(f"Mapping Error: {str(e)}")

async def get_highest_declaration_version_async(policy_number: str, collection_name: str, carrier_name: str) -> int:
    """
    Async version of getting highest declaration version.
    """
    try:
        # Find all documents with matching policy number
        cursor = adb[collection_name].find({"DeclarationNumber": policy_number, "CarrierName": carrier_name})
        matches = await cursor.to_list(length=None)
        
        if not matches:
            return 1
        
        # Collect all versions safely
        versions = []
        for doc in matches:
            try:
                version = doc.get('Version')
                if version is not None and isinstance(version, (int, float)):
                    versions.append(int(version))
                else:
                    versions.append(1)
            except Exception:
                versions.append(1)
        
        # Get highest version
        highest_version = max(versions) if versions else 1
        
        return highest_version + 1
        
    except Exception as e:
        # Safely default to version 1
        return 1
    

def transform_frontend_blob_name(original_blob_name):
    """
    Transforms frontend upload blob names by removing the ==userId==PolicyName part and the last UUID.
    
    Example:
    Input: 'UA-538_03_12_SpecialProvisions11_UUID==user@email.com==PolicyName.pdf'
    Output: 'UA-538_03_12_SpecialProvisions11.pdf'
    """
    # Check if this is a frontend upload
    if "==" not in original_blob_name:
        return original_blob_name
        
    # Get the extension
    extension = original_blob_name.split('.')[-1]
    
    # Remove everything after and including the first ==
    base_name = original_blob_name.split('==')[0]
    
    # Find the last underscore and remove everything after it
    if '_' in base_name:
        base_name = base_name.rsplit('_', 1)[0]
    
    # Add back the extension
    return f"{base_name}.{extension}"

def upload_blob_with_transform(blobname, filepath, container_name, content_type="application/json", metadata=None):
    """
    Uploads a blob with name transformation for frontend uploads.
    Returns the transformed blob name used for upload.
    """
    transformed_name = transform_frontend_blob_name(blobname)
    
    blob_client_instance = blob_service_client.get_blob_client(
        container=container_name, 
        blob=transformed_name
    )
    
    content_settings = ContentSettings(content_type=content_type)
    with open(filepath, "rb") as data:
        blob_client_instance.upload_blob(data, content_settings=content_settings, overwrite=True)
    
    return transformed_name


def get_processing_paths(doc_id, base_doc_id, blobname, is_split, split_indices_dict, base_dir):
    """
    Helper function to get the correct paths for processing, handling both frontend upload cases.
    Returns: (input_dir, pdf_file_path, adobe_output_dir)
    """
    # Check for both frontend cases
    is_frontend_merged = split_indices_dict.get('frontendmergedupload') == 'Y'
    is_frontend_upload = split_indices_dict.get('frontendflag') == 'Y'
    
    # Transform name if either frontend case is true
    if is_frontend_merged or is_frontend_upload:
        transformed_blobname = transform_frontend_blob_name(blobname)
        
        if is_split:
            input_dir = os.path.join(MERGED_DIR, base_doc_id)
            pdf_file_path = os.path.join(input_dir, blobname)  # Use original name for reading
            adobe_output_dir = os.path.join(input_dir, f"adobe-api-{transformed_blobname.split('.')[0]}-{base_doc_id}")
        else:
            input_dir = os.path.join(base_dir, base_doc_id)
            os.makedirs(input_dir, exist_ok=True)
            pdf_file_path = os.path.join(input_dir, blobname)  # Use original name for reading
            adobe_output_dir = os.path.join(input_dir, f"adobe-api-{transformed_blobname.split('.')[0]}-{doc_id}")
            os.makedirs(adobe_output_dir, exist_ok=True)
    else:
        # Original path handling for non-frontend cases
        if is_split:
            input_dir = os.path.join(MERGED_DIR, base_doc_id)
            pdf_file_path = os.path.join(input_dir, blobname)
            adobe_output_dir = os.path.join(input_dir, f"adobe-api-{blobname.split('.')[0]}-{base_doc_id}")
        else:
            input_dir = os.path.join(base_dir, base_doc_id)
            os.makedirs(input_dir, exist_ok=True)
            pdf_file_path = os.path.join(input_dir, blobname)
            adobe_output_dir = os.path.join(input_dir, f"adobe-api-{blobname.split('.')[0]}-{doc_id}")
            os.makedirs(adobe_output_dir, exist_ok=True)
            
    return input_dir, pdf_file_path, adobe_output_dir
