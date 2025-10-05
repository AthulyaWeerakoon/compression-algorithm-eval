from memory_profiler import memory_usage
import numpy as np
import time
from typing import List,Dict 
from util.types import Quantizer
from collections import Counter

def profile_memory(func, *args, **kwargs):
    """
    Run a function and return (result, peak_memory_MB, elapsed_time_sec).
    Uses memory_profiler.memory_usage with max_usage=True.
    """
    t_start = time.perf_counter()
    result, peak_mem = None, None

    def wrapper():
        nonlocal result
        result = func(*args, **kwargs,)
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


def build_frequency_table(values: List[float],quantizer : Quantizer) -> Dict[int, int]:
    """
    Convert the data into a frequency table
    """
    
    symbols = [quantizer.value_to_symbol(i) for i in values]
    freq_table = Counter(symbols)
    return freq_table   


def build_cdf(freq_table: Dict[int, int]):
    """
    Build CDF table 
    """
    cdf ={}
    cum = 0
    for symbol, frequency in sorted(freq_table.items()):
        cdf[symbol]=cum
        cum += frequency

    return cum , cdf