"""
 Copyright 2024 Adobe
 All Rights Reserved.

 NOTICE: Adobe permits you to use, modify, and distribute this file in
 accordance with the terms of the Adobe license agreement accompanying it.
"""

import logging
import os
from datetime import datetime
from .log_setup import setup_logger
from src import ADOBE_CLIENT_ID, ADOBE_CLIENT_SECRET
from monads.monad_class import monad_wrapper, async_monad_wrapper

from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
from adobe.pdfservices.operation.io.stream_asset import StreamAsset
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.table_structure_type import TableStructureType
logging.basicConfig(level=logging.INFO)


@monad_wrapper
def run_extract_pdf(filename, adobe_dir, logger_name, request_id):
    try:
        #app_logger = setup_logger(logger_name, requestId=request_id)
        logger_name.info("Adobe API credentials are \nADOBE_CLIENT_ID: {} \nADOBE_CLIENT_SECRET: {}".format(ADOBE_CLIENT_ID, ADOBE_CLIENT_SECRET))
        print("Adobe API credentials are \nADOBE_CLIENT_ID: {} \nADOBE_CLIENT_SECRET: {}".format(ADOBE_CLIENT_ID, ADOBE_CLIENT_SECRET))
        file = open(filename, "rb")
        input_stream = file.read()
        file.close()
        credentials = ServicePrincipalCredentials(client_id=ADOBE_CLIENT_ID, client_secret=ADOBE_CLIENT_SECRET)
        pdf_services = PDFServices(credentials=credentials)
        input_asset = pdf_services.upload(input_stream=input_stream, mime_type=PDFServicesMediaType.PDF)
        extract_pdf_params = ExtractPDFParams(
            elements_to_extract=[ExtractElementType.TEXT, ExtractElementType.TABLES],
            table_structure_type=TableStructureType.CSV,
            styling_info=True,
        )
        extract_pdf_job = ExtractPDFJob(input_asset=input_asset, extract_pdf_params=extract_pdf_params)
        location = pdf_services.submit(extract_pdf_job)
        pdf_services_response = pdf_services.get_job_result(location, ExtractPDFResult)
        result_asset: CloudAsset = pdf_services_response.get_result().get_resource()
        stream_asset: StreamAsset = pdf_services.get_content(result_asset)
        output_file_path = adobe_dir  # + "ExtractTextInfoWithStylingInfoFromPDF.zip" Disabling the addition of this thing
        with open(output_file_path, "wb") as file:
            file.write(stream_asset.get_input_stream())

        logger_name.info("Saved the results of Adobe API.")
        print("Saved the results of Adobe API.")
    except ServiceApiException as ser_api:
        logger_name.error(f"Could not connect with adobe API. ServiceApiException:: {ser_api}")
        print(f"Could not connect with adobe API. ServiceApiException:: {ser_api}")
        raise ServiceApiException(message="Adobe API ServiceApiException.")
    except ServiceUsageException as ser_use:
        logger_name.error(f"Could not connect with adobe API. ServiceUsageException:: {ser_use}")
        print(f"Could not connect with adobe API. ServiceUsageException:: {ser_use}")
        raise ServiceUsageException(message="Adobe API ServiceUsageException.")
    except SdkException as sdk_ex:
        logger_name.error(f"Could not connect with adobe API. SdkException SDK Exception:: {sdk_ex}")
        print(f"Could not connect with adobe API. SdkException SDK Exception:: {sdk_ex}")
        raise SdkException(message="Adobe API limit quota exhausted.")
    except Exception as e:
        logger_name.error(f"Unexpected Error happened while processing PDF: {str(e)}")
        print(f"Unexpected error occurred while processing PDF: {str(e)}")
        raise Exception(f"Unexpected error in PDF processing: {str(e)}")
