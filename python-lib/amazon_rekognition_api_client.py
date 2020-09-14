# -*- coding: utf-8 -*-
"""Module with a wrapper class around the Amazon Rekognition API"""

import logging
from typing import AnyStr
import boto3
from boto3.exceptions import Boto3Error
from botocore.exceptions import BotoCoreError, ClientError
from PIL import UnidentifiedImageError


API_EXCEPTIONS = (Boto3Error, BotoCoreError, ClientError, UnidentifiedImageError)


def get_client(aws_access_key_id: AnyStr, aws_secret_access_key: AnyStr, aws_region_name: AnyStr) -> boto3.client:
    client = boto3.client(
        service_name="rekognition",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region_name,
    )
    logging.info("Credentials loaded")
    return client
