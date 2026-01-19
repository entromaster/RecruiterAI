from opencensus.ext.azure.log_exporter import AzureLogHandler
from opencensus.ext.azure.trace_exporter import AzureExporter
from opencensus.ext.flask.flask_middleware import FlaskMiddleware
from opencensus.trace.samplers import ProbabilitySampler
import logging
from agentic_rag.src import (
    INSTRUMENTATION_KEY,
)

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