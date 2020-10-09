# -*- coding: utf-8 -*-
"""Module with functions to parallelize API calls with error handling"""

import logging
import inspect
import math
from typing import Callable, AnyStr, List, Tuple, NamedTuple, Dict, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from more_itertools import chunked, flatten
from tqdm.auto import tqdm as tqdm_auto

from plugin_io_utils import ErrorHandlingEnum, build_unique_column_names


# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

DEFAULT_PARALLEL_WORKERS = 4
DEFAULT_BATCH_SIZE = 10
DEFAULT_API_SUPPORT_BATCH = False
DEFAULT_VERBOSE = False


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


def api_call_single_row(
    api_call_function: Callable,
    api_column_names: NamedTuple,
    row: Dict,
    api_exceptions: Union[Exception, Tuple[Exception]],
    error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    verbose: bool = DEFAULT_VERBOSE,
    **api_call_function_kwargs
) -> Dict:
    """
    Wraps a single-row API calling function to:
    - ensure it has a 'row' parameter which is a dict
      (for batches of rows, use the api_call_batch function below)
    - return the row with a new 'response' key containing the function result
    - handles errors from the function with two methods:
        * (default) do not fail on API-related exceptions, just log it
        and return the row with new error keys
        * fail if there is an error and raise it
    """
    if error_handling == ErrorHandlingEnum.FAIL:
        response = api_call_function(row=row, **api_call_function_kwargs)
        row[api_column_names.response] = response
    else:
        for k in api_column_names:
            row[k] = ""
        try:
            response = api_call_function(row=row, **api_call_function_kwargs)
            row[api_column_names.response] = response
        except api_exceptions as e:
            logging.warning(str(e))
            error_type = str(type(e).__qualname__)
            module = inspect.getmodule(e)
            if module is not None:
                error_type = str(module.__name__) + "." + error_type
            row[api_column_names.error_message] = str(e)
            row[api_column_names.error_type] = error_type
            row[api_column_names.error_raw] = str(e.args)
    return row


def api_call_batch(
    api_call_function: Callable,
    api_column_names: NamedTuple,
    batch: List[Dict],
    batch_api_response_parser: Callable,
    api_exceptions: Union[Exception, Tuple[Exception]],
    error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    verbose: bool = DEFAULT_VERBOSE,
    **api_call_function_kwargs
) -> List[Dict]:
    """
    Wraps a batch API calling function to:
    - ensure it has a 'batch' parameter which is a list of dict
    - return the batch with a new 'response' key in each dict
      containing the function result
    - handles errors from the function with two methods:
        * (default) do not fail on API-related exceptions, just log it
        and return the batch with new error keys in each dict (using batch_api_parser)
        * fail if there is an error and raise it
    """
    if error_handling == ErrorHandlingEnum.FAIL:
        response = api_call_function(batch=batch, **api_call_function_kwargs)
        batch = batch_api_response_parser(batch=batch, response=response, api_column_names=api_column_names)
        errors = [row[api_column_names.error_message] for row in batch if row[api_column_names.error_message] != ""]
        if len(errors) != 0:
            raise Exception("API returned errors: " + str(errors))
    else:
        try:
            response = api_call_function(batch=batch, **api_call_function_kwargs)
            batch = batch_api_response_parser(batch=batch, response=response, api_column_names=api_column_names)
        except api_exceptions as e:
            logging.warning(str(e))
            error_type = str(type(e).__qualname__)
            module = inspect.getmodule(e)
            if module is not None:
                error_type = str(module.__name__) + "." + error_type
            for row in batch:
                row[api_column_names.response] = ""
                row[api_column_names.error_message] = str(e)
                row[api_column_names.error_type] = error_type
                row[api_column_names.error_raw] = str(e.args)
    return batch


def convert_api_results_to_df(
    input_df: pd.DataFrame,
    api_results: List[Dict],
    api_column_names: NamedTuple,
    error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    verbose: bool = DEFAULT_VERBOSE,
) -> pd.DataFrame:
    """
    Helper function to the "api_parallelizer" main function.
    Combine API results (list of dict) with input dataframe,
    and convert it to a dataframe.
    """
    if error_handling == ErrorHandlingEnum.FAIL:
        columns_to_exclude = [v for k, v in api_column_names._asdict().items() if "error" in k]
    else:
        columns_to_exclude = []
        if not verbose:
            columns_to_exclude = [api_column_names.error_raw]
    output_schema = {**{v: str for v in api_column_names}, **dict(input_df.dtypes)}
    output_schema = {k: v for k, v in output_schema.items() if k not in columns_to_exclude}
    record_list = [{col: result.get(col) for col in output_schema.keys()} for result in api_results]
    api_column_list = [c for c in api_column_names if c not in columns_to_exclude]
    output_column_list = list(input_df.columns) + api_column_list
    output_df = pd.DataFrame.from_records(record_list).astype(output_schema).reindex(columns=output_column_list)
    assert len(output_df.index) == len(input_df.index)
    return output_df


def api_parallelizer(
    input_df: pd.DataFrame,
    api_call_function: Callable,
    api_exceptions: Union[Exception, Tuple[Exception]],
    column_prefix: AnyStr,
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
    api_support_batch: bool = DEFAULT_API_SUPPORT_BATCH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    verbose: bool = DEFAULT_VERBOSE,
    **api_call_function_kwargs
) -> pd.DataFrame:
    """
    Apply an API call function in parallel to a pandas.DataFrame.
    The DataFrame is passed to the function as row dictionaries.
    Parallelism works by:
    - (default) sending multiple concurrent threads
    - if the API supports it, sending batches of row
    """
    df_iterator = (i[1].to_dict() for i in input_df.iterrows())
    len_iterator = len(input_df.index)
    log_msg = "Calling remote API endpoint with {} rows...".format(len_iterator)
    if api_support_batch:
        log_msg += ", chunked by {}".format(batch_size)
        df_iterator = chunked(df_iterator, batch_size)
        len_iterator = math.ceil(len_iterator / batch_size)
    logging.info(log_msg)
    api_column_names = build_unique_column_names(input_df.columns, column_prefix)
    pool_kwargs = api_call_function_kwargs.copy()
    more_kwargs = [
        "api_call_function",
        "error_handling",
        "api_exceptions",
        "api_column_names",
    ]
    for k in more_kwargs:
        pool_kwargs[k] = locals()[k]
    for k in ["fn", "row", "batch"]:  # Reserved pool keyword arguments
        pool_kwargs.pop(k, None)
    api_results = []
    with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
        if api_support_batch:
            futures = [pool.submit(api_call_batch, batch=batch, **pool_kwargs) for batch in df_iterator]
        else:
            futures = [pool.submit(api_call_single_row, row=row, **pool_kwargs) for row in df_iterator]
        for f in tqdm_auto(as_completed(futures), total=len_iterator):
            api_results.append(f.result())
    if api_support_batch:
        api_results = flatten(api_results)
    output_df = convert_api_results_to_df(input_df, api_results, api_column_names, error_handling, verbose)
    num_api_error = sum(output_df[api_column_names.response] == "")
    num_api_success = len(input_df.index) - num_api_error
    logging.info("Remote API call results: {} rows succeeded, {} rows failed.".format(num_api_success, num_api_error))
    return output_df
