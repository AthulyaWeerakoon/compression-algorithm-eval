"""POC plotting helpers. Each function returns a Figure and never calls show()."""
import numpy as np
import matplotlib.pyplot as plt
from .poc import checked_ticks


def plot_training_series(values):
    a=np.asarray(values)
    fig,ax=plt.subplots()
    ax.plot(np.arange(len(a)),a)
    ax.set_xlabel("Sample"); ax.set_ylabel("Value")
    return fig


def plot_training_distribution(values, quantum):
    a=np.asarray(values,dtype=np.float64); q=float(quantum)
    if a.ndim!=1 or not a.size or not np.isfinite(a).all() or not np.isfinite(q) or q<=0:
        raise ValueError("finite training values and positive quantum required")
    ticks=checked_ticks(a,q,"training distribution")
    vals,counts=np.unique(ticks,return_counts=True)
    fig,ax=plt.subplots(); ax.bar(vals*q,counts,width=q)
    ax.set_xlabel("Value"); ax.set_ylabel("Frequency")
    return fig


def plot_residual_counts(alphabet, quantum):
    q=float(quantum)
    if not np.isfinite(q) or q<=0: raise ValueError("positive finite quantum required")
    counts=np.asarray(alphabet.counts)
    if counts.ndim!=1 or len(counts)!=alphabet.high_tick-alphabet.low_tick+1:
        raise ValueError("invalid trained alphabet counts")
    ticks=np.arange(alphabet.low_tick,alphabet.high_tick+1,dtype=np.int64)
    fig,ax=plt.subplots(); ax.bar(ticks*q,counts,width=q)
    ax.set_xlabel("Signed residual"); ax.set_ylabel("Training frequency")
    return fig
