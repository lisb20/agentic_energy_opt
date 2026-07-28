"""
Evaluator for the function minimization example
"""

import importlib.util
import numpy as np
import time
import concurrent.futures
import traceback
import signal
from openevolve.evaluation_result import EvaluationResult
import sys
import yaml
import os
from matplotlib import pyplot as plt
import base64
import pickle
import copy
import json
from pathlib import Path

RELEASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_DIR / "simulator"))
from flow import Flow, get_generator_power
from RL_env_v2 import myenv, calculate_grid_cost

import time

def run_with_timeout(func, args=(), kwargs=None, timeout_seconds=5):
    """
    Run a function with a timeout using concurrent.futures

    Args:
        func: Function to run
        args: Arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        timeout_seconds: Timeout in seconds

    Returns:
        Result of the function or raises TimeoutError
    """
    results = []
    for arg in kwargs or []:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, **arg)
            try:
                result = future.result(timeout=timeout_seconds)
                results.append(result)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")
    return results

def evaluate(program_path):
    """
    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics
    """
    print("EVALUATOR: Evaluating program", program_path)
    try:
        state_path = RELEASE_DIR / "cache" / "state_to_opt.pkl"
        if state_path.exists():
            with state_path.open("rb") as f:
                cache_env = pickle.load(f)
                
            env = cache_env["env"]
            flow = cache_env["flow"]
            mask = cache_env["mask"]
            task_specific_args = cache_env["task_specific_args"]
            print("loading cache, task_specific_args", task_specific_args)
            domain = cache_env["task_type"]
            if domain == "PV":
                from PV import PVTaskExecutor
                executor = PVTaskExecutor(**task_specific_args)
            elif domain == "V2G":
                from V2G import V2GTaskExecutor
                executor = V2GTaskExecutor(**task_specific_args)
        else:
            raise FileNotFoundError(
                f"Optimization state not found: {state_path}. Start optimization through the MCP tool first."
            )
            

    except Exception as e:
        print(f"Error: {str(e)}")
        return EvaluationResult(
            metrics={"combined_score": -20, "error": "InitializationError"},
            artifacts={"error_type": type(e).__name__, "error_message": str(e)},
        )
    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        # Check if the required function exists
        if not hasattr(program, "Action"):
            print(f"Error: program does not have 'Action' function")
            
            error_artifacts = {
                "error_type": "MissingFunction",
                "error_message": "Program is missing required 'Action' function",
                "suggestion": "Make sure your program includes a function named 'Action' that returns a np array solution"
            }
            metrics = {
                    "combined_score": -20,
                    "error": "Missing Action function",
                }
            return EvaluationResult(
                metrics=metrics,
                artifacts=error_artifacts
            )

    except Exception as e:
        error_artifacts = {
            "error_type": "LoadError",
            "error_message": f"Could not load program from {program_path}",
            "suggestion": "Check that the program file exists and is valid Python code"
        }
        metrics = {
            "combined_score": -20,
            "error": "LoadError",
        }

        return EvaluationResult(
            metrics=metrics,
            artifacts=error_artifacts
        )
    
    try:
        inputs_args = executor.build_inputs(env, flow, action=program.Action)
        result = run_with_timeout(func = program.Action, kwargs= inputs_args, timeout_seconds=100)
        sim_result = executor.post_process(result, env, flow, mask)
        sim_result = executor.analysis_result(sim_result, env, flow, mask)

        metrics, artifacts = executor.compute_metrics(sim_result, env, flow, mask)
        return EvaluationResult(
            metrics=metrics,
            artifacts=artifacts
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        error_artifacts = {
            "error_type": "RuntimeError",
            "error_message": f"Could not run program from {program_path}",
            "suggestion": "Check that the program file exists and is valid Python code"
        }
        metrics = {
            "combined_score": -20,
            "error": "RuntimeError",
        }
        return EvaluationResult(
            metrics=metrics,
            artifacts=error_artifacts
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python evaluator.py PROGRAM_PATH")
    result = evaluate(sys.argv[1])
    print(result.metrics)
    
    
