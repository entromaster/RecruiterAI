from azure.identity import (
    DefaultAzureCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from agentic_rag.src import (
    ENV_NAME,
    ENV_NAMES,
    MONGO_DB,
    MONGO_CONNECTION_STRING,
    MONGO_CONFIGURATION_COLLECTION,
    MONGO_DOCUMENT_COLLECTION,
    MONGO_CLIENT_COLLECTION
)

if ENV_NAME in ENV_NAMES:
    azure_credential = ManagedIdentityCredential()
else:
    azure_credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(azure_credential, "https://cognitiveservices.azure.com/.default")
mgclient = MongoClient(MONGO_CONNECTION_STRING)
amgclient = AsyncIOMotorClient(MONGO_CONNECTION_STRING)
adb = amgclient[MONGO_DB]
db = mgclient[MONGO_DB]

async def get_process_config(client_name, config_type="approach"):
    client_config = await adb[MONGO_CLIENT_COLLECTION].find_one({"ClientName": client_name})
    if not client_config:
        raise ValueError(f"Client config not found for carrier: {client_name}")
    config_id = client_config.get("ClientID")
    if not config_id:
        raise ValueError(f"ClientID not found for client: {client_name}")
    regex_config = await adb[MONGO_CONFIGURATION_COLLECTION].find_one({"ClientID": config_id})
    if not regex_config:
        raise ValueError(f"Regex config not found for ClientID: {config_id}")
    if config_type == "approach":
        return regex_config.get("Approach")

# async def get_process_config(doc_id, config_type="approach"):
#     # Get the process approach from the database
#     document = await adb[MONGO_DOCUMENT_COLLECTION].find_one({"documentId": doc_id})
#     if not document:
#         raise ValueError(f"Document not found with ID: {doc_id}")
#     client_name = document.get("clientName")
#     if not client_name:
#         raise ValueError(f"CarrierName not found in document: {doc_id}")
#     client_config = await adb[MONGO_CLIENT_COLLECTION].find_one({"ClientName": client_name})
#     if not client_config:
#         raise ValueError(f"Client config not found for carrier: {client_name}")
#     config_id = client_config.get("ClientID")
#     if not config_id:
#         raise ValueError(f"ClientID not found for client: {client_name}")
#     regex_config = await adb[MONGO_CONFIGURATION_COLLECTION].find_one({"ClientID": config_id})
#     if not regex_config:
#         raise ValueError(f"Regex config not found for ClientID: {config_id}")
#     if config_type == "approach":
#         return regex_config.get("Approach")
#     else:
#         config_pattern = regex_config.get("DocumentProcessingConfiguration")
#         if not config_pattern:
#             raise ValueError(f"ConfigPattern not found for ClientID: {config_id}")
#         config_pattern = regex_config.get("DocumentProcessingConfiguration")
#         if not config_pattern:
#             raise ValueError(f"ConfigPattern not found for ClientID: {config_id}")
#         if config_pattern.get("GroupedPolicy") == "Y" or config_pattern.get("CCRFP") == "Y":
#             return config_pattern
#         else:
#             required_fields = ['Mode', 'FormNumberPattern']
#             missing_fields = [field for field in required_fields if field not in config_pattern]
#             if missing_fields:
#                 raise ValueError(f"Config pattern missing required fields: {', '.join(missing_fields)}")
#             if config_pattern['Mode'] not in ['Table', 'Text']:
#                 raise ValueError("Invalid Mode value. Must be either 'Table' or 'Text'")
#             if config_pattern['Mode'] == 'Table':
#                 if 'TableConfig' not in config_pattern:
#                     raise ValueError("TableConfig is required when Mode is 'Table'")
#                 table_config = config_pattern['TableConfig']
#                 required_table_fields = [
#                     'TablesFolder',
#                     'FileExtensions',
#                     'Encoding',
#                     'ColumnRequirements',
#                     'BasePolicyCondition',
#                     'DeclarationCondition'
#                 ]
#                 missing_table_fields = [field for field in required_table_fields if field not in table_config]
#                 if missing_table_fields:
#                     raise ValueError(f"TableConfig missing required fields: {', '.join(missing_table_fields)}")
#                 if not isinstance(table_config['FileExtensions'], list):
#                     raise ValueError("FileExtensions must be an array")
#             return config_pattern

    
async def get_clientID_async(client_name):
    print("INSIDE get_clientID_async, COLLECTION NAME: ", MONGO_CLIENT_COLLECTION, "\nCLIENT_NAME: ", client_name)
    collection = adb[MONGO_CLIENT_COLLECTION]
    query = {
        "ClientName": client_name
    }
    result = await collection.find_one(query)
    print("QUERY:",query, "\nRESULT:", result)
    return result.get("ClientID")

async def get_document_async(query, mongo_collection: str=MONGO_DOCUMENT_COLLECTION):
    collection = adb[mongo_collection]
    document = await collection.find_one(query)
    return document

async def to_mongo_async(mongo_dict, collection):
    result = await adb[collection].insert_many(mongo_dict)
    inserted_ids = result.inserted_ids

async def update_status_mongo_async(query, update_doc, collection=MONGO_DOCUMENT_COLLECTION, arr_filter=None):
    if not any(key.startswith("$") for key in update_doc):
        update_doc = {"$set": update_doc}
    if arr_filter:
        result = await adb[collection].update_one(
            query, 
            update_doc,
            array_filters=[arr_filter]
        )
    else:
        result = await adb[collection].update_one(
            query, 
            update_doc
        )
    return result 

async def get_mapped_child_doc_ids_async(doc_id, client_name, policy_name, form_numbers=None):
    child_document_ids = []
    extracted_form_numbers = []
    
    if form_numbers:
        for normalized_form_number, form_number in zip(form_numbers["normalized_form_numbers"], form_numbers["form_numbers"]):
            query = {
                "processStatus": {"$ne": "Failed"}, 
                "clientName": client_name, 
                "documentSubTag": policy_name, 
                "formNumber.normalizedFormNumber": normalized_form_number
            }
            doc = await adb[MONGO_DOCUMENT_COLLECTION].find_one(query)
            dec_normalized_form_number = doc["formNumber"]["normalizedFormNumber"]
            if dec_normalized_form_number != normalized_form_number:
                child_document_ids.append({
                    "documentId": doc["documentId"],
                    "isActive": doc["isActive"],
                    "documentTag": doc["documentTag"],
                    "filePath": doc["filePath"],
                    "jsonId": doc["jsonId"]
                })
                extracted_form_numbers.append({
                    "formNumber": form_number,
                    "normalizedFormNumber": normalized_form_number
                })
        return child_document_ids, extracted_form_numbers
    else:
        query = {
            "processStatus": {"$ne": "Failed"},
            "clientName": client_name,
            "documentSubTag": policy_name,  # Match by policy name stored in documentSubTag
            "documentTag": {"$in": ["BasePolicy", "Endorsement"]},
            "documentId": {"$ne": doc_id}  # Exclude the declaration itself
        }
        matching_documents = await adb[MONGO_DOCUMENT_COLLECTION].find(query).to_list(length=100)
        if not matching_documents:
            return child_document_ids, extracted_form_numbers
        for doc in matching_documents:
            child_obj = {
                "documentId": doc.get("documentId"),
                "isActive": doc["isActive"],
                "documentTag": doc.get("documentTag"),
                "filePath": doc.get("filePath"),
                "jsonId": doc.get("jsonID")
            }
            child_document_ids.append(child_obj)
        return child_document_ids, extracted_form_numbers
    