import logging
from pymongo import MongoClient
import json
from src import (MONGO_CONNECTION_STRING, MONGO_DB, MONGO_ERROR_LOG_COLLECTION, MONGO_INFO_LOG_COLLECTION, INSTRUMENTATION_KEY)
import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler
from opencensus.ext.azure.trace_exporter import AzureExporter
from opencensus.ext.flask.flask_middleware import FlaskMiddleware
from opencensus.trace.samplers import ProbabilitySampler


def azure_log(app):
    exporter = AzureExporter(connection_string=INSTRUMENTATION_KEY)
    FlaskMiddleware(
        app,
        exporter=exporter,
        sampler=ProbabilitySampler(1.0)
    )
    logger = logging.getLogger(__name__)
    logger.addHandler(AzureLogHandler(connection_string=INSTRUMENTATION_KEY))
    logger.setLevel(logging.INFO)    
    return logger


class JsonFormatter(logging.Formatter):
    def __init__(self, fmt_dict: dict = None, time_format: str = "%Y-%m-%dT%H:%M:%S", msec_format: str = "%s.%03dZ"):
        self.fmt_dict = fmt_dict if fmt_dict is not None else {"message": "message"}
        self.default_time_format = time_format
        self.default_msec_format = msec_format
        self.datefmt = None

    def usesTime(self) -> bool:
        return "asctime" in self.fmt_dict.values()

    def formatMessage(self, record) -> dict:
        return {fmt_key: record.__dict__[fmt_val] for fmt_key, fmt_val in self.fmt_dict.items()}

    def format(self, record) -> str:
        record.message = record.getMessage()
        
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)

        message_dict = self.formatMessage(record)

        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)

        if record.exc_text:
            message_dict["exc_info"] = record.exc_text

        if record.stack_info:
            message_dict["stack_info"] = self.formatStack(record.stack_info)
        
        # Ensure requestId is included in the message_dict
        if hasattr(record, 'requestId'):
            message_dict["requestId"] = record.requestId

        if hasattr(record, 'documentId'):
            message_dict["documentId"] = record.documentId

        return json.dumps(message_dict, default=str)
    
class MongoDBHandler(logging.Handler):
    def __init__(self, mongo_uri, database_name, collection_name):
        super().__init__()
        self.client = MongoClient(mongo_uri)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]

    def emit(self, record):
        try:
            log_entry = self.format(record)
            log_dict = record.__dict__
            log_dict['message'] = log_entry
            self.collection.insert_one(log_dict)
        except Exception as e:
            print(f"Error inserting log entry: {e}")

    def close(self):
        self.client.close()

def setup_logger(LOG_NAME, requestId=None, doc_id=None,
                 LOG_FILE_INFO='file.log',
                 LOG_FILE_ERROR='file.err',
                 connection_string=MONGO_CONNECTION_STRING,
                 mongo_db=MONGO_DB,
                 info_collection=MONGO_INFO_LOG_COLLECTION,
                 error_collection=MONGO_ERROR_LOG_COLLECTION):
    
    log = logging.getLogger(LOG_NAME)
    json_formatter = JsonFormatter({"level": "levelname", 
                                    "message": "message", 
                                    "loggerName": "name", 
                                    "processName": "processName",
                                    "processID": "process", 
                                    "threadName": "threadName", 
                                    "threadID": "thread",
                                    "timestamp": "asctime",
                                    "requestId": "requestId",
                                    "documentId": "documentId"})
    
    log.setLevel(logging.INFO)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(json_formatter)
    log.addHandler(stream_handler)

    # MongoDB handler for INFO level logs
    
    mongo_handler_info = MongoDBHandler(connection_string, mongo_db, info_collection)
    mongo_handler_info.setFormatter(json_formatter)
    mongo_handler_info.setLevel(logging.INFO)
    mongo_handler_info.addFilter(lambda record: record.levelno <= logging.INFO)
    log.addHandler(mongo_handler_info)

    # MongoDB handler for ERROR level logs
    mongo_handler_error = MongoDBHandler(connection_string, mongo_db, error_collection)
    mongo_handler_error.setFormatter(json_formatter)
    mongo_handler_error.setLevel(logging.ERROR)
    log.addHandler(mongo_handler_error)

    # Add requestId to all logs
    log = logging.LoggerAdapter(log, {"requestId": requestId, "documentId": doc_id})

    return log

# Usage example:
# log = setup_logger('my_logger', requestId='12345')
# log.info("This is an info message")
# log.error("This is an error message")