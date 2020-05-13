# -*- coding: utf-8 -*-
import logging

import boto3
from boto3.exceptions import Boto3Error
from botocore.exceptions import BotoCoreError, ClientError

# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

API_EXCEPTIONS = (Boto3Error, BotoCoreError, ClientError)

# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


def get_client(api_configuration_preset):
    client = boto3.client(
        service_name="comprehendmedical",
        aws_access_key_id=api_configuration_preset.get("aws_access_key"),
        aws_secret_access_key=api_configuration_preset.get("aws_secret_key"),
        region_name=api_configuration_preset.get("aws_region"),
    )
    logging.info("Credentials loaded")
    return client
