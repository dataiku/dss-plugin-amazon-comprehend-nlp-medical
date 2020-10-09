# -*- coding: utf-8 -*-
"""Module with classes to format results from the Amazon Comprehend Medical API"""

import logging
from typing import AnyStr, Dict, List
from enum import Enum

import pandas as pd

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


class MedicalEntityTypeEnum(Enum):
    ANATOMY = "Anatomy"
    MEDICAL_CONDITION = "Medical condition"
    MEDICATION = "Medication"
    PROTECTED_HEALTH_INFORMATION = "Protected health information"
    TEST_TREATMENT_PROCEDURE = "Test treatment procedure"
    TIME_EXPRESSION = "Time expression"


class MedicalPHITypeEnum(Enum):
    ADDRESS = "Address"
    AGE = "Age"
    DATE = "Date"
    NAME = "Name"
    PHONE_OR_FAX = "Phone or fax"
    EMAIL = "Email"
    ID = "ID"


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


class GenericAPIFormatter:
    """
    Geric Formatter class for API responses:
    - initialize with generic parameters
    - compute generic column descriptions
    - apply format_row to dataframe
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        column_prefix: AnyStr = "api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    ):
        self.input_df = input_df
        self.column_prefix = column_prefix
        self.error_handling = error_handling
        self.api_column_names = build_unique_column_names(input_df, column_prefix)
        self.column_description_dict = {
            v: API_COLUMN_NAMES_DESCRIPTION_DICT[k] for k, v in self.api_column_names._asdict().items()
        }

    def format_row(self, row: Dict) -> Dict:
        return row

    def format_df(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Formatting API results...")
        df = df.apply(func=self.format_row, axis=1)
        df = move_api_columns_to_end(df, self.api_column_names, self.error_handling)
        logging.info("Formatting API results: Done.")
        return df


class MedicalPhiAPIFormatter(GenericAPIFormatter):
    """
    Formatter class for Protected Health Information API responses:
    - make sure response is valid JSON
    - expand results to multiple columns
    - compute column descriptions
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        minimum_score: float,
        column_prefix: AnyStr = "medical_phi_api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    ):
        super().__init__(input_df, column_prefix, error_handling)
        self.minimum_score = float(minimum_score)
        self._compute_column_description()

    def _compute_column_description(self):
        for entity_enum in MedicalPHITypeEnum:
            entity_type_column = generate_unique(
                "entity_type_" + str(entity_enum.value).lower() + "_text", self.input_df.keys(), self.column_prefix,
            )
            self.column_description_dict[entity_type_column] = "List of '{}' PHI entities extracted by the API".format(
                str(entity_enum.value)
            )

    def format_row(self, row: Dict) -> Dict:
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)
        entities = response.get("Entities", [])
        discarded_entities = [
            e
            for e in entities
            if float(e.get("Score", 0)) < self.minimum_score
            and e.get("Type", "") in [e.name for e in MedicalEntityTypeEnum]
        ]
        if len(discarded_entities) != 0:
            logging.info("Discarding {} entities below the minimum score threshold".format(len(discarded_entities)))
        for entity_enum in MedicalPHITypeEnum:
            entity_type_column = generate_unique(
                "entity_type_" + str(entity_enum.value).lower() + "_text", row.keys(), self.column_prefix,
            )
            row[entity_type_column] = [
                e.get("Text", "")
                for e in entities
                if e.get("Type", "") == entity_enum.name and float(e.get("Score", 0)) >= self.minimum_score
            ]
            if len(row[entity_type_column]) == 0:
                row[entity_type_column] = ""
        return row


class MedicalEntityAPIFormatter(GenericAPIFormatter):
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
        minimum_score: float,
        column_prefix: AnyStr = "medical_entity_api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    ):
        super().__init__(input_df, column_prefix, error_handling)
        self.entity_types = entity_types
        self.minimum_score = float(minimum_score)
        self._compute_column_description()

    def _compute_column_description(self):
        for entity_enum in MedicalEntityTypeEnum:
            entity_type_column = generate_unique(
                "entity_type_" + str(entity_enum.value).lower() + "_text", self.input_df.keys(), self.column_prefix,
            )
            self.column_description_dict[
                entity_type_column
            ] = "List of '{}' medical entities extracted by the API".format(str(entity_enum.value))

    def format_row(self, row: Dict) -> Dict:
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)
        entities = response.get("Entities", [])
        discarded_entities = [
            e
            for e in entities
            if float(e.get("Score", 0)) < self.minimum_score
            and e.get("Category", "") in [e.name for e in MedicalEntityTypeEnum]
        ]
        if len(discarded_entities) != 0:
            logging.info("Discarding {} entities below the minimum score threshold".format(len(discarded_entities)))
        for entity_enum in MedicalEntityTypeEnum:
            if entity_enum in self.entity_types:
                entity_type_column = generate_unique(
                    "entity_type_" + str(entity_enum.value).lower() + "_text", row.keys(), self.column_prefix,
                )
                row[entity_type_column] = [
                    e.get("Text", "")
                    for e in entities
                    if e.get("Category", "") == entity_enum.name and float(e.get("Score", 0)) >= self.minimum_score
                ]
                if len(row[entity_type_column]) == 0:
                    row[entity_type_column] = ""
        return row
