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


class KeyPhraseExtractionAPIFormatter:
    """
    Formatter class for Key Phrase Extraction API responses:
    - make sure response is valid JSON
    - extract a given number of key phrases
    - compute column descriptions
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        num_key_phrases: int,
        column_prefix: AnyStr = "keyphrase_api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    ):
        self.input_df = input_df
        self.num_key_phrases = num_key_phrases
        self.column_prefix = column_prefix
        self.error_handling = error_handling
        self.api_column_names = build_unique_column_names(input_df, column_prefix)
        self.column_description_dict = self._compute_column_description()

    def _compute_column_description(self):
        column_description_dict = {
            v: API_COLUMN_NAMES_DESCRIPTION_DICT[k]
            for k, v in self.api_column_names._asdict().items()
        }
        for n in range(self.num_key_phrases):
            keyphrase_column = generate_unique(
                "keyphrase_" + str(n + 1) + "_text",
                self.input_df.keys(),
                self.column_prefix,
            )
            confidence_column = generate_unique(
                "keyphrase_" + str(n + 1) + "_confidence",
                self.input_df.keys(),
                self.column_prefix,
            )
            column_description_dict[
                keyphrase_column
            ] = "Keyphrase {} extracted by the API".format(str(n + 1))
            column_description_dict[
                confidence_column
            ] = "Confidence score in Keyphrase {} from 0 to 1".format(str(n + 1))
        return column_description_dict

    def format_row(self, row: Dict) -> Dict:
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)
        key_phrases = sorted(
            response.get("KeyPhrases", []), key=lambda x: x.get("Score"), reverse=True
        )
        for n in range(self.num_key_phrases):
            keyphrase_column = generate_unique(
                "keyphrase_" + str(n + 1) + "_text", row.keys(), self.column_prefix
            )
            confidence_column = generate_unique(
                "keyphrase_" + str(n + 1) + "_confidence",
                row.keys(),
                self.column_prefix,
            )
            if len(key_phrases) > n:
                row[keyphrase_column] = key_phrases[n].get("Text", "")
                row[confidence_column] = key_phrases[n].get("Score")
            else:
                row[keyphrase_column] = ""
                row[confidence_column] = None
        return row

    def format_df(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Formatting API results...")
        df = df.apply(func=self.format_row, axis=1)
        df = move_api_columns_to_end(df, self.api_column_names)
        logging.info("Formatting API results: Done.")
        return df
