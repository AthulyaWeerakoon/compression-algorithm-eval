from memory_profiler import memory_usage
import numpy as np
import time
import math
from typing import List, Dict 
from .types import Quantizer


def profile_memory(func, *args, **kwargs):
    """
    Run a function and return (result, peak_memory_MB, elapsed_time_sec).
    Uses memory_profiler.memory_usage with max_usage=True.
    """
    t_start = time.perf_counter()
    result, peak_mem = None, None

    def wrapper():
        nonlocal result
        result = func(*args, **kwargs)
        return result

    peak_mem = memory_usage((wrapper,), max_usage=True, retval=False, max_iterations=1)
    elapsed = time.perf_counter() - t_start
    return result, peak_mem, elapsed



def build_time_series_dataset(series, input_size, output_size, shuffle=True):
    """
    Build training data for time-series prediction.

    Args:
        series (array-like): 1D sequence of values (list or np.array).
        input_size (int): Number of timesteps in input window.
        output_size (int): Number of timesteps in output window.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        X (np.ndarray): Shape (n_samples, input_size).
        Y (np.ndarray): Shape (n_samples, output_size).
    """
    series = np.array(series)
    n_samples = len(series) - input_size - output_size + 1
    if n_samples <= 0:
        raise ValueError("Series too short for given input_size and output_size")

    X, Y = [], []
    for i in range(n_samples):
        X.append(series[i: i + input_size])
        Y.append(series[i + input_size: i + input_size + output_size])

    X = np.array(X)
    Y = np.array(Y)

    if shuffle:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X, Y = X[indices], Y[indices]

    return X, Y


def largest_remainder_quantize(counts, M):
    """
    Quantize a list of raw counts into integer frequencies summing to M.
    Ensures all frequencies are >= 1 and preserves distribution ratios as closely as possible.

    Args:
        counts (list[int]): Raw symbol counts (positive integers).
        M (int): Target total frequency sum.

    Returns:
        list[int]: Quantized integer frequencies summing to M.
    """

    if not counts:
        raise ValueError("Empty counts list.")
    N = len(counts)
    assert M >= N, f"M={M} too small to allocate at least 1 to each of {N} symbols."

    total = sum(counts)
    if total <= 0:
        raise ValueError("Total count must be positive.")

    # Compute proportional ideal frequencies
    r = [c / total * M for c in counts]

    # Take floors first
    f = [max(1, math.floor(x)) for x in r]

    rem = M - sum(f)

    if rem > 0:
        # Distribute remaining counts to largest fractional parts
        frac = sorted(enumerate(r), key=lambda kv: (kv[1] - math.floor(kv[1])), reverse=True)
        i = 0
        while rem > 0:
            idx = frac[i % N][0]
            f[idx] += 1
            rem -= 1
            i += 1
    elif rem < 0:
        # Remove from smallest fractional parts, ensuring >= 1
        frac = sorted(enumerate(r), key=lambda kv: (kv[1] - math.floor(kv[1])))
        i = 0
        while rem < 0:
            idx = frac[i % N][0]
            if f[idx] > 1:
                f[idx] -= 1
                rem += 1
            i += 1

    assert sum(f) == M, "Normalization failed: total frequencies do not sum to M."

    return f
