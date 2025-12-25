"""
Utility functions for fetching and processing data from Weights & Biases (wandb).
"""

import wandb
import numpy as np
from typing import List, Union, Optional, Tuple, Literal


def fetch_single_run_data(run_wandb_path: str, run_id: str, metric_name: str, max_samples: int = 1000000) -> np.ndarray:
    """
    Fetch metric data for a single run from Weights & Biases.

    Args:
        run_id: Unique identifier for the wandb run
        metric_name: Name of the metric to fetch
        max_samples: Maximum number of samples to fetch (default: 10000)

    Returns:
        numpy.ndarray: Array containing the metric values

    Raises:
        ValueError: If run_id or metric_name is empty
        wandb.errors.CommError: If there's an error communicating with wandb
        KeyError: If the specified metric is not found in the run
    """
    if not run_id or not metric_name:
        raise ValueError("run_id and metric_name cannot be empty")

    try:
        run_path = f"{run_wandb_path}/{run_id}"
        run = wandb.Api().run(run_path)
        history = run.history(keys=[metric_name], samples=max_samples)
        return np.array(history[metric_name])
    except wandb.errors.CommError as e:
        raise wandb.errors.CommError(f"Failed to fetch data for run {run_id}: {str(e)}")
    except KeyError:
        raise KeyError(f"Metric '{metric_name}' not found in run {run_id}")


def normalize_array_length(arrays: List[np.ndarray], mode: Literal["pad", "truncate"] = "pad", pad_value: float = float("nan")) -> List[np.ndarray]:
    """
    Normalize arrays to have the same length by either padding or truncating.

    Args:
        arrays: List of arrays to normalize
        mode: 'pad' to pad shorter arrays with pad_value, 'truncate' to cut to shortest length
        pad_value: Value to use for padding (default: NaN)

    Returns:
        List of arrays with the same length
    """
    if not arrays:
        return arrays

    lengths = [len(arr) for arr in arrays]
    target_length = max(lengths) if mode == "pad" else min(lengths)

    normalized = []
    for arr in arrays:
        if len(arr) == target_length:
            normalized.append(arr)
        elif len(arr) < target_length:
            # Pad array
            padding = np.full(target_length - len(arr), pad_value)
            normalized.append(np.concatenate([arr, padding]))
        else:
            # Truncate array
            normalized.append(arr[:target_length])

    return normalized


def fetch_multiple_runs_data(
    run_wandb_path: str, run_ids: List[str], metric_name: str, length_mode: Literal["pad", "truncate"] = "pad", pad_value: float = float("nan")
) -> Tuple[np.ndarray, List[int]]:
    """
    Fetch metric data for multiple runs from Weights & Biases and normalize lengths.

    Args:
        run_ids: List of unique identifiers for wandb runs
        metric_name: Name of the metric to fetch
        length_mode: How to handle different lengths ('pad' or 'truncate')
        pad_value: Value to use for padding if length_mode is 'pad'

    Returns:
        Tuple containing:
            - numpy.ndarray: 2D array containing metric values for each run
            - List[int]: Original lengths of each array before normalization

    Raises:
        ValueError: If run_ids is empty or metric_name is empty
        wandb.errors.CommError: If there's an error communicating with wandb

    Example:
        >>> data, lengths = fetch_multiple_runs_data(
        ...     ["abc123", "def456"],
        ...     "loss",
        ...     length_mode='pad'
        ... )
        >>> print(f"Original lengths: {lengths}")
        >>> print(f"Normalized shape: {data.shape}")
    """
    if not run_ids:
        raise ValueError("run_ids cannot be empty")
    if not metric_name:
        raise ValueError("metric_name cannot be empty")

    try:
        # Fetch all arrays
        arrays = []
        original_lengths = []
        for run_id in run_ids:
            arr = fetch_single_run_data(run_wandb_path, run_id, metric_name)
            arrays.append(arr)
            original_lengths.append(len(arr))

        # Normalize lengths
        normalized = normalize_array_length(arrays, mode=length_mode, pad_value=pad_value)
        return np.array(normalized)

    except Exception as e:
        raise RuntimeError(f"Failed to fetch data for multiple runs: {str(e)}")
