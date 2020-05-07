# -*- coding: utf-8 -*-
import logging
from typing import AnyStr, Dict, List
from enum import Enum

import pandas as pd
import boto3
from boto3.exceptions import Boto3Error
from botocore.exceptions import BotoCoreError, ClientError

from plugin_io_utils import (
    API_COLUMN_NAMES_DESCRIPTION_DICT,
    ErrorHandlingEnum,
    build_unique_column_names,
    generate_unique,
    safe_json_loads,
    move_api_columns_to_end,
)


# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

API_EXCEPTIONS = (Boto3Error, BotoCoreError, ClientError)

API_SUPPORT_BATCH = False
BATCH_RESULT_KEY = None
BATCH_ERROR_KEY = None
BATCH_INDEX_KEY = None
BATCH_ERROR_MESSAGE_KEY = None
BATCH_ERROR_TYPE_KEY = None

VERBOSE = False


class MedicalEntityTypeEnum(Enum):
    ANATOMY = "Anatomy"
    MEDICAL_CONDITION = "Medical condition"
    MEDICATION = "Medication"
    PROTECTED_HEALTH_INFORMATION = "Protected health information"
    TEST_TREATMENT_PROCEDURE = "Test treatment procedure"
    TIME_EXPRESSION = "Time expression"


class MedicalPHITypeEnum(Enum):
    AGE = "Age"
    DATE = "Date"
    NAME = "Mame"
    PHONE_OR_FAX = "Phone or fax"
    EMAIL = "Email"
    ID = "ID"


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


def get_client(api_configuration_preset, service_name: AnyStr):
    client = boto3.client(
        service_name=service_name,
        aws_access_key_id=api_configuration_preset.get("aws_access_key"),
        aws_secret_access_key=api_configuration_preset.get("aws_secret_key"),
        region_name=api_configuration_preset.get("aws_region"),
    )
    logging.info("Credentials loaded")
    return client


class MedicalPhiAPIFormatter:
    """
    Formatter class for Protected Health Information API responses:
    - make sure response is valid JSON
    - expand results to multiple columns
    - compute column descriptions
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        column_prefix: AnyStr = "medical_phi_api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    ):
        self.input_df = input_df
        self.column_prefix = column_prefix
        self.error_handling = error_handling
        self.api_column_names = build_unique_column_names(input_df, column_prefix)
        self.column_description_dict = self._compute_column_description()

    def _compute_column_description(self):
        column_description_dict = {
            v: API_COLUMN_NAMES_DESCRIPTION_DICT[k]
            for k, v in self.api_column_names._asdict().items()
        }
        for entity_enum in MedicalPHITypeEnum:
            entity_type_column = generate_unique(
                "entity_type_" + str(entity_enum.value).lower() + "_text",
                self.input_df.keys(),
                self.column_prefix,
            )
            column_description_dict[
                entity_type_column
            ] = "List of '{}' PHI entities extracted by the API".format(
                str(entity_enum.value)
            )
        return column_description_dict

    def format_row(self, row: Dict) -> Dict:
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)
        entities = response.get("Entities", [])
        for entity_enum in MedicalPHITypeEnum:
            entity_type_column = generate_unique(
                "entity_type_" + str(entity_enum.value).lower() + "_text",
                row.keys(),
                self.column_prefix,
            )
            row[entity_type_column] = [
                e.get("Text", "")
                for e in entities
                if e.get("Type", "") == entity_enum.name
            ]
            if len(row[entity_type_column]) == 0:
                row[entity_type_column] = ""
        return row

    def format_df(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Formatting API results...")
        df = df.apply(func=self.format_row, axis=1)
        df = move_api_columns_to_end(df, self.api_column_names)
        logging.info("Formatting API results: Done.")
        return df


class MedicalEntityAPIFormatter:
    """
    Formatter class for Medical Entity Recognition API responses:
    - make sure response is valid JSON
    - expand results to multiple columns
    - compute column descriptions
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        entity_types: List,
        column_prefix: AnyStr = "medical_entity_api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    ):
        self.input_df = input_df
        self.entity_types = entity_types
        self.column_prefix = column_prefix
        self.error_handling = error_handling
        self.api_column_names = build_unique_column_names(input_df, column_prefix)
        self.column_description_dict = self._compute_column_description()

    def _compute_column_description(self):
        column_description_dict = {
            v: API_COLUMN_NAMES_DESCRIPTION_DICT[k]
            for k, v in self.api_column_names._asdict().items()
        }
        for entity_enum in MedicalEntityTypeEnum:
            entity_type_column = generate_unique(
                "entity_type_" + str(entity_enum.value).lower() + "_text",
                self.input_df.keys(),
                self.column_prefix,
            )
            column_description_dict[
                entity_type_column
            ] = "List of '{}' medical entities extracted by the API".format(
                str(entity_enum.value)
            )
        return column_description_dict

    def format_row(self, row: Dict) -> Dict:
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)
        entities = response.get("Entities", [])
        for entity_enum in MedicalEntityTypeEnum:
            entity_type_column = generate_unique(
                "entity_type_" + str(entity_enum.value).lower() + "_text",
                row.keys(),
                self.column_prefix,
            )
            row[entity_type_column] = [
                e.get("Text", "")
                for e in entities
                if e.get("Category", "") == entity_enum.name
            ]
            if len(row[entity_type_column]) == 0:
                row[entity_type_column] = ""
        return row

    def format_df(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Formatting API results...")
        df = df.apply(func=self.format_row, axis=1)
        df = move_api_columns_to_end(df, self.api_column_names)
        logging.info("Formatting API results: Done.")
        return df
