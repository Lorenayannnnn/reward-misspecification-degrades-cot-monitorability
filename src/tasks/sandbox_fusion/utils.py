# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import concurrent.futures
import json
import logging
import os
import threading
import time
import traceback
import uuid
from typing import Any, Optional

import requests

DEFAULT_TIMEOUT = 6  # compile and run timeout per assert
MAX_RETRIES = 2
INITIAL_RETRY_DELAY = 1
API_TIMEOUT = 8
# Per-attempt HTTP budget is 2*timeout + API_TIMEOUT = 20s, so the worst case over retries
# (~41s) stays under _VERIFY_TIMEOUT in src/tasks/coding.py.

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = [
    "python",
    "cpp",
    "nodejs",
    "go",
    "go_test",
    "java",
    "php",
    "csharp",
    "bash",
    "typescript",
    "sql",
    "rust",
    "cuda",
    "lua",
    "R",
    "perl",
    "D_ut",
    "ruby",
    "scala",
    "julia",
    "pytest",
    "junit",
    "kotlin_script",
    "jest",
    "verilog",
    "python_gpu",
    "lean",
    "swift",
    "racket",
]


def call_sandbox_api(
    sandbox_fusion_url: str,
    code: str,
    stdin: Optional[str],
    compile_timeout: int,
    run_timeout: int,
    memory_limit_mb: int,
    language: str = "python",
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """
    Run ``code`` in the remote sandbox, retrying transient failures. Returns ``(response_json, None)`` on
    success or ``(None, error_message)`` after retries.
    """
    request_id = str(uuid.uuid4())
    log_prefix = f"[Request ID: {request_id}] "

    if language not in SUPPORTED_LANGUAGES:
        error_msg = f"{log_prefix}Unsupported language: {language}"
        logger.error(error_msg)
        return None, error_msg

    payload = json.dumps(
        {
            "compile_timeout": compile_timeout,
            "run_timeout": run_timeout,
            "code": code,
            "stdin": stdin,
            "memory_limit_MB": memory_limit_mb,
            "language": language,
            "files": {},
            "fetch_files": [],
        }
    )
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    request_timeout = compile_timeout + run_timeout + API_TIMEOUT

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                sandbox_fusion_url,
                headers=headers,
                data=payload,
                timeout=request_timeout,
            )

            # Gateway errors mean the server is overloaded; retry with backoff.
            if response.status_code in (502, 503, 504):
                last_error = (
                    f"{log_prefix}API Request Error: HTTP {response.status_code} on attempt "
                    f"{attempt + 1}/{MAX_RETRIES}"
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(INITIAL_RETRY_DELAY * (attempt + 1))
                continue

            response.raise_for_status()

            return response.json(), None

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            # Connection resets and read timeouts from an overloaded sandbox are retried
            # rather than failing the constraint outright.
            last_error = f"{log_prefix}API Request Error (transient, attempt {attempt + 1}/{MAX_RETRIES}): {e}"
            if attempt < MAX_RETRIES - 1:
                time.sleep(INITIAL_RETRY_DELAY * (attempt + 1))
            continue
        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"
            break
        except json.JSONDecodeError as e:
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}"
            break
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"
            break

    # Strip the internal request id; the error becomes -1 in check_correctness.
    return None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed after retries"


# Wrapper for the fn_name path (stdin holds JSON-encoded arguments). Rendered with
# .format(), so literal braces are doubled.
_PY_FN_WRAPPER = """
import traceback
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json

# === User's Original Code START ===
{generation}
# === User's Original Code END ===

_SANDBOX_FN_NAME = "{fn_name}"

def _execute_user_function():
    # --- Input Parsing ---
    _raw_input_str = sys.stdin.read()
    _args = []
    if _raw_input_str.strip(): # If there's input
        try:
            _args = [json.loads(line) for line in _raw_input_str.split('\\n')]
        except json.JSONDecodeError as _je:
            sys.stderr.write(f"WrapperError: Invalid JSON input for '{{_SANDBOX_FN_NAME}}': {{_je}}\\nInput was: "
                              f"{{_raw_input_str[:200]}}\\n")
            return None, True # result, error_occurred

    # --- Function Location and Execution ---
    try:
        _target_callable = None
        # Try global scope first
        if _SANDBOX_FN_NAME in globals():
            _target_callable = globals()[_SANDBOX_FN_NAME]
        # Else, if 'Solution' class exists, try to get its method
        elif 'Solution' in globals():
            _Solution_class = globals()['Solution']
            # Attempt to instantiate and get method.
            # Errors (e.g., Solution not a class, instantiation fails, method missing)
            # will be caught by the broad except block below.
            _solution_instance = _Solution_class()
            _target_callable = getattr(_solution_instance, _SANDBOX_FN_NAME)

        if not _target_callable:
            sys.stderr.write(f"WrapperError: Function or method '{{_SANDBOX_FN_NAME}}' not found.\\n")
            return None, True # result, error_occurred

        _fn_result = _target_callable(*_args)
        return _fn_result, False # result, no_error
    except Exception: # Catches errors from Solution instantiation, getattr, or function call
        sys.stderr.write(f"Error during setup or execution of '{{_SANDBOX_FN_NAME}}':\\n{{traceback.format_exc()}}\\n")
        return None, True # result, error_occurred

if __name__ == '__main__':
    _result, _error_occurred = _execute_user_function()

    if not _error_occurred:
        # Serialize result to stdout
        if isinstance(_result, (dict, list, tuple)) or _result is None or isinstance(_result, bool):
            print(json.dumps(_result))
        elif isinstance(_result, (int, float, str)):
            print(str(_result)) # Ensure string conversion for print
        else:
            # For other types, default to string representation.
            print(str(_result))
    # Optional: To explicitly exit with an error code if the sandbox relies on it
    # else:
    #    sys.exit(1)
"""


def _call_api(
    case_index: int,
    code: str,
    stdin: Optional[str],
    sandbox_fusion_url: str,
    timeout: int,
    memory_limit_mb: int,
    language: str,
    concurrent_semaphore: Optional[threading.Semaphore],
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """
    Make one sandbox call, behind the shared semaphore when given.
    """
    api_response = None
    error_msg = None
    try:
        if concurrent_semaphore:
            with concurrent_semaphore:
                api_response, error_msg = call_sandbox_api(
                    sandbox_fusion_url=sandbox_fusion_url,
                    code=code,
                    stdin=stdin,
                    compile_timeout=timeout,
                    run_timeout=timeout,
                    memory_limit_mb=memory_limit_mb,
                    language=language,
                )
        else:
            api_response, error_msg = call_sandbox_api(
                sandbox_fusion_url=sandbox_fusion_url,
                code=code,
                stdin=stdin,
                compile_timeout=timeout,
                run_timeout=timeout,
                memory_limit_mb=memory_limit_mb,
                language=language,
            )
    except Exception as e:
        error_msg = f"API Request Exception during check_correctness for case {case_index + 1}: {e}"
        traceback.print_exc()
    return api_response, error_msg


def _process_single_case(
    case_index: int,
    stdin_data: Any,
    expected_output: Any,
    sandbox_fusion_url: str,
    generation: str,
    timeout: int,
    memory_limit_mb: int,
    language: str,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    fn_name: Optional[str] = None,
) -> tuple[int, dict[str, Any]]:
    """
    Run one test case and return ``(result_code, metadata)``.
    """
    current_generation_code = generation
    if fn_name and language == "python":
        current_generation_code = _PY_FN_WRAPPER.format(generation=generation, fn_name=fn_name)

    stdin = None if stdin_data is None else str(stdin_data)
    api_response, error_msg = _call_api(case_index, current_generation_code, stdin,
                                        sandbox_fusion_url, timeout, memory_limit_mb,
                                        language, concurrent_semaphore)

    metadata = {
        "case_index": case_index,
        "input": stdin,
        "expected_output": str(expected_output) if expected_output else None,
        "api_request_error": error_msg,
        "api_response": None,
        "status": "unknown",
        "stdout": None,
        "stderr": None,
        "exit_code": None,
        "duration": None,
        "compile_duration": None,
        "compile_stderr": None,
        "api_status": None,
        "compile_status": None,
        "run_status": None,
    }
    if error_msg:
        metadata["status"] = "api_error"
        result_status = -1
    elif api_response:
        result_status = _decode_api_response(metadata, api_response, expected_output)
    else:  # should not happen with call_sandbox_api as written
        metadata["status"] = "unknown_api_state"
        result_status = -1
    return result_status, metadata


def _decode_api_response(metadata: dict[str, Any], api_response: dict[str, Any], expected_output: Any):
    """
    Fill ``metadata`` from the API response and return the result code
    (True/False, -1 API or sandbox error, -2 runtime error, -3 timeout, -4 compile error).
    """
    metadata["api_response"] = api_response
    metadata["api_status"] = api_response.get("status")
    compile_result = api_response.get("compile_result")
    run_result = api_response.get("run_result")

    if compile_result:
        metadata["compile_status"] = compile_result.get("status")
        metadata["compile_duration"] = compile_result.get("execution_time")
        metadata["compile_stderr"] = compile_result.get("stderr")

    if run_result:
        metadata["run_status"] = run_result.get("status")
        metadata["stdout"] = run_result.get("stdout")
        metadata["stderr"] = run_result.get("stderr")
        metadata["exit_code"] = run_result.get("return_code")
        metadata["duration"] = run_result.get("execution_time")

    api_status = metadata["api_status"]
    if api_status == "SandboxError":
        metadata["status"] = "sandbox_error"
        return -1
    if api_status == "Failed":
        is_compile_error = compile_result and (
            metadata["compile_status"] in ["Error", "TimeLimitExceeded"]
            or (metadata["compile_status"] == "Finished" and compile_result.get("return_code") != 0)
        )
        if is_compile_error:
            if metadata["compile_status"] == "TimeLimitExceeded":
                metadata["status"] = "compile_timeout"
            else:
                metadata["status"] = "compile_error"
            return -4
        if run_result:
            is_runtime_error = (
                metadata["run_status"] == "TimeLimitExceeded"
                or metadata["run_status"] == "Error"
                or (metadata["run_status"] == "Finished" and run_result.get("return_code") != 0)
            )
            if is_runtime_error:
                if metadata["run_status"] == "TimeLimitExceeded":
                    metadata["status"] = "timeout"
                    return -3
                metadata["status"] = "runtime_error"
                return -2
            metadata["status"] = "unknown_failure"
            return -1
        metadata["status"] = "unknown_failure_state"
        return -1
    if api_status == "Success":
        if run_result and metadata["run_status"] == "Finished":
            actual_output = metadata["stdout"] if metadata["stdout"] is not None else ""
            # Compare with trailing newlines stripped.
            if expected_output is None or str(actual_output).rstrip("\n") == str(expected_output).rstrip("\n"):
                metadata["status"] = "success"
                return True
            metadata["status"] = "wrong_answer"
            return False
        metadata["status"] = "unexpected_success_state"
        return -1
    metadata["status"] = f"unknown_api_status_{api_status}"
    return -1


def check_correctness(
    sandbox_fusion_url: str,
    in_outs: Optional[dict],
    generation: str,
    timeout: int = DEFAULT_TIMEOUT,
    memory_limit_mb: int = 1024,
    language: str = "python",
    concurrent_semaphore: Optional[threading.Semaphore] = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """
    Run ``generation`` against every test case in the remote sandbox. Returns ``(results, metadata_list)``
    ordered like ``inputs``; result codes are True/False, -1 API or sandbox error, -2 runtime error,
    -3 timeout, -4 compile error.
    """
    if not in_outs or "inputs" not in in_outs or "outputs" not in in_outs:
        return [-1], [{"error": "Invalid input/output data"}]

    inputs = in_outs["inputs"]
    expected_outputs = in_outs["outputs"]
    fn_name = in_outs.get("fn_name")
    num_cases = len(inputs)
    assert_cases = in_outs.get("assert_case", [""] * num_cases)
    results = [None] * num_cases
    metadata_list = [None] * num_cases

    if num_cases == 0:
        return [], []

    if len(inputs) != len(expected_outputs):
        return [-1] * num_cases, [{"error": "Input/output count mismatch", "case_index": i} for i in range(num_cases)]

    if len(assert_cases) != num_cases:
        return [-1] * num_cases, [{"error": "Input/output count mismatch", "case_index": i} for i in range(num_cases)]

    # The semaphore, not the pool size, bounds sandbox concurrency.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(32, os.cpu_count() * 5)) as executor:
        future_to_index = {
            executor.submit(
                _process_single_case,
                i,
                stdin_data,
                expected_outputs[i],
                sandbox_fusion_url,
                generation + "\n\n" + assert_cases[i],
                timeout,
                memory_limit_mb,
                language,
                concurrent_semaphore,
                fn_name,
            ): i
            for i, stdin_data in enumerate(inputs)
        }

        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result_status, metadata = future.result()
                results[index] = result_status
                metadata_list[index] = metadata
            except Exception as exc:
                logger.error(f"Test case {index} generated an exception: {exc}")
                traceback.print_exc()
                results[index] = -1
                metadata_list[index] = {
                    "case_index": index,
                    "input": str(inputs[index]),
                    "expected_output": str(expected_outputs[index]) if expected_outputs[index] else None,
                    "api_request_error": f"Internal execution error: {exc}",
                    "status": "internal_error",
                }

    # Upstream marks every case after the first -4 as -4. That is dropped here: each case is
    # its own program, so one compile failure (or compile timeout on a loaded sandbox) says
    # nothing about the others, and propagating it would fabricate the mix pass_any_fail_any pays for.
    return results, metadata_list
