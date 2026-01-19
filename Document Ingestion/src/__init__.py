import os
from dotenv import load_dotenv
import logging

#Manually setting these for now, might remove later


load_dotenv("config/envs/app_settings.env")
logging.info("Loaded crawford app_settings.env")
# if "dev" in ENV_NAME:
#     load_dotenv("config/envs/dev.env")
#     logging.info("Loaded crawford dev.env")
# elif "test" in ENV_NAME:
#     load_dotenv("config/envs/test.env")
#     logging.info("Loaded crawford test.env")
# elif "uat" in ENV_NAME:
#     load_dotenv("config/envs/uat.env")
#     logging.info("Loaded crawford uat.env")
# elif "preprod" in ENV_NAME:
#     load_dotenv("config/envs/preprod.env")
#     logging.info("Loaded crawford preprod.env")
# elif "prod" in ENV_NAME:
#     load_dotenv("config/envs/prod.env")
#     logging.info("Loaded crawford prod.env")
# else:
#     load_dotenv("config/envs/dev_p.env")
#     logging.info("Loaded p dev_p.env")

  


QUEUE_PROCESS_BATCH_SIZE = int(os.getenv("QUEUE_PROCESS_BATCH_SIZE"))
BASE_ENDORSEMENT_DIR = os.getenv("BASE_ENDORSEMENT_DIR")
DECLARATION_DIR = os.getenv("DECLARATION_DIR")
MERGED_DIR = os.getenv("MERGED_DIR")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_INFO_LOG_COLLECTION = os.getenv("MONGO_INFO_LOG_COLLECTION")
MONGO_ERROR_LOG_COLLECTION = os.getenv("MONGO_ERROR_LOG_COLLECTION")
MONGO_DOCUMENT_COLLECTION = os.getenv("MONGO_DOCUMENT_COLLECTION")
MONGO_POLICIES_COLLECTION = os.getenv("MONGO_POLICIES_COLLECTION")
MONGO_DECLARATION_COLLECTION = os.getenv("MONGO_DECLARATION_COLLECTION")
MONGO_CLIENT_COLLECTION = os.getenv("MONGO_CLIENT_COLLECTION")
MONGO_REGEX_COLLECTION = os.getenv("MONGO_REGEX_COLLECTION")
MONGO_MERGED_COLLECTION = os.getenv("MONGO_MERGED_COLLECTION")
MAX_WORKERS = int(os.getenv("MAX_WORKERS"))
SWAGGER_STATIC_PATH = os.getenv("SWAGGER_STATIC_PATH")
SWAGGER_URL = os.getenv("SWAGGER_URL")
API_URL = os.getenv("API_URL")
APP_NAME = os.getenv("APP_NAME")
LOGGER_NAME = os.getenv("LOGGER_NAME")
ADOBE_RESULT_FILENAME = os.getenv("ADOBE_RESULT_FILENAME")
ADOBE_DIR = os.getenv("ADOBE_DIR")
QUEUE_NAME = os.getenv("QUEUE_NAME")
ENV_NAMES =  os.getenv("ENV_NAMES").split(",")



ENV_NAME = os.environ.get("ENVIRONMENT").lower()
WEBSITE_PORT = int(os.environ.get("WEBSITES_PORT"))
INSTRUMENTATION_KEY = os.environ.get("INSTRUMENTATION_KEY")
BLOB_ACCOUNT_NAME = os.environ.get("BLOB_ACCOUNT_NAME")
MONGO_CONNECTION_STRING = os.environ.get("MONGO_CONNECTION_STRING")
AZURE_OPENAI_SERVICE = os.environ.get("AZURE_OPENAI_SERVICE")
AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
SERVICE_BUS_NAME = os.environ.get("SERVICE_BUS_NAME")
OCR_SERVICE_BUS_NAME = os.environ.get("OCR_SERVICE_BUS_NAME")
BASE_ENDO_QUEUE_NAME = os.environ.get("BASE_ENDO_QUEUE_NAME")
OCR_CONTAINER_NAME = os.environ.get("OCR_CONTAINER_NAME")
ACCOUNT_URL = 'https://' + BLOB_ACCOUNT_NAME + '.blob.core.windows.net'
FULLY_QUALIFIED_NAME = "https://" + SERVICE_BUS_NAME + ".servicebus.windows.net"
OCRFULLY_QUALIFIED_NAME = "https://" + OCR_SERVICE_BUS_NAME + ".servicebus.windows.net"
ADOBE_CLIENT_ID = os.environ.get("ADOBE_CLIENT_ID")
ADOBE_CLIENT_SECRET = os.environ.get("ADOBE_CLIENT_SECRET")
MSALREC = os.environ.get("MSALRECIEVE")
MSALNotify = os.environ.get("MSALNOTIFY")