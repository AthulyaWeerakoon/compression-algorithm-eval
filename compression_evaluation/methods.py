import numpy as np
import matplotlib.pyplot as plt
import time
import math
from memory_profiler import memory_usage
from collections import Counter
from .types import ParametricDistribution
from .distributions import Gaussian, Beta, Gamma, InverseGamma, ScaledInvChiSquared
from typing import Sequence


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


def build_frequency_table(data_list):
    """
    Build a frequency table (dictionary) from a list of symbols.
    
    Args:
        data_list (list): List of symbols (integers or strings)
    
    Returns:
        dict: {symbol: frequency count}
    """
    if not data_list:
        return {}

    freq_dict = dict(Counter(data_list))
    return freq_dict


def compute_posterior_mixture(priors: list[ParametricDistribution], data: np.ndarray) -> list[ParametricDistribution]:
    """
        Compute the posterior distribution mixture given a list of conjugate priors and observed data.
        Returns a new ParametricDistribution instance.
    """
    return [compute_single_posterior(prior, data) for prior in priors]


def compute_single_posterior(prior: ParametricDistribution, data: np.ndarray) -> ParametricDistribution:
    """
    Compute the posterior distribution given a conjugate prior and observed data.
    Returns a new ParametricDistribution instance.
    """
    n = len(data)
    if n == 0:
        return prior  # no update possible

    # gaussian likelihood with known variance (conjugate prior: gaussian)
    if prior.name == "Gaussian":
        mu0, var0 = prior.parameters()
        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=1)

        # assume known sample variance = sample_var
        # precision form for conjugate update
        tau0 = 1.0 / var0
        tau = 1.0 / sample_var

        post_var = 1.0 / (tau0 + n * tau)
        post_mean = post_var * (tau0 * mu0 + n * tau * sample_mean)

        return Gaussian(post_mean, post_var, prior.weight)

    # beta-bernoulli model
    elif prior.name == "Beta":
        alpha, beta = prior.parameters()
        successes = np.sum(data)
        failures = n - successes
        return Beta(alpha + successes, beta + failures, prior.weight)

    # gamma-poisson model
    elif prior.name == "Gamma":
        alpha, beta = prior.parameters()
        return Gamma(alpha + np.sum(data), beta + n, prior.weight)

    # inverse-gamma (as prior for gaussian variance)
    elif prior.name == "Inverse-Gamma":
        alpha, beta = prior.parameters()
        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=1)
        return InverseGamma(alpha + n / 2, beta + 0.5 * n * sample_var, prior.weight)

    # scaled inverse chi-squared (alternative variance prior)
    elif prior.name == "Scaled-Inverse-Chi-Squared":
        nu, sigma_sq = prior.parameters()
        sample_var = np.var(data, ddof=1)
        return ScaledInvChiSquared(nu + n, (nu * sigma_sq + n * sample_var) / (nu + n), prior.weight)

    else:
        raise NotImplementedError(f"Posterior update not implemented for {prior.name}")


def approximate_mixture_map_pdf(distributions: list[ParametricDistribution]) -> float:
    """
    Approximate MAP for a mixture by evaluating weighted PDF at each component's mode.

    This works for any parametric distribution as long as it implements .pdf(x).
    """
    max_val = -np.inf
    best_mode = None

    for p in distributions:
        w = getattr(p, "weight", 1.0)
        mode = p.mode()

        if hasattr(p, "pdf"):
            pdf_val = p.pdf(mode)
        else:
            # fallback
            pdf_val = 1.0

        weighted_val = w * pdf_val
        if weighted_val > max_val:
            max_val = weighted_val
            best_mode = mode

    return best_mode


def plot_mixture_over_hist(
        data: Sequence[float],
        mixture: Sequence[ParametricDistribution],
        bin_size: float,
        x_min: float | None = None,
        x_max: float | None = None,
        figsize: tuple = (10, 5),
        show: bool = True,
        title: str | None = None
) -> None:
    """
    Plot histogram (using bin_size) and overlay a provided mixture of ParametricDistribution components.
    - Each component must implement .pdf(x) (vectorized or scalar), .weight, .name, .parameters(), .mode().
    - The densities are scaled to histogram counts using len(data) * bin_size.

    Parameters
    ----------
    data : 1D array-like
        Input samples.
    mixture : sequence of ParametricDistribution
        Already-fit components with .pdf(x) and .weight.
    bin_size : float
        Width of histogram bins.
    x_min, x_max : optional floats
        Range to plot; defaults to data min/max.
    figsize : tuple
        Figure size.
    show : bool
        If True, calls plt.show().
    title : str, optional
        Plot title.
    """
    data = np.asarray(data).ravel()
    if data.size == 0:
        raise ValueError("data must contain at least one value")

    if x_min is None:
        x_min = float(np.min(data))
    if x_max is None:
        x_max = float(np.max(data))
    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min")

    # create bins from min to max using bin_size
    bins = np.arange(x_min, x_max + bin_size, bin_size)
    hist_counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # prepare x values for pdf plotting
    x_vals = np.linspace(x_min, x_max, 1000)
    pdf_total = np.zeros_like(x_vals, dtype=float)
    n = data.size

    plt.figure(figsize=figsize)
    plt.bar(bin_centers, hist_counts, width=bin_size, color='g', alpha=0.6, label='Data histogram')

    for comp in mixture:
        w = float(getattr(comp, "weight", 1.0))
        # evaluate pdf: allow vectorized or scalar implementations
        try:
            comp_pdf = np.asarray(comp.pdf(x_vals), dtype=float)
        except Exception:
            # fallback to scalar evaluation
            comp_pdf = np.asarray([float(comp.pdf(x)) for x in x_vals], dtype=float)

        # scale to histogram counts
        comp_pdf_scaled = w * comp_pdf * n * bin_size
        pdf_total += comp_pdf_scaled
        plt.plot(x_vals, comp_pdf_scaled, '--', linewidth=2)

    # total mixture curve (scaled)
    plt.plot(x_vals, pdf_total, 'r-', linewidth=2, label='Mixture (total)')

    if title is None:
        plt.title('Mixture fit over data')
    else:
        plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)

    if show:
        plt.show()
