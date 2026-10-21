import os
from decimal import Decimal
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from compression_evaluation.poc import (
    chronological_bounds, impute_training_only, infer_decimal_quantum, make_windows,
    checked_ticks, snap_to_grid, SignedGridQuantizer, fit_training_alphabet,
    make_codec_pair, evaluate_suffix, benchmark_pairs, pinned_allowed_cpu,
)
from compression_evaluation.poc_plots import plot_training_series, plot_training_distribution, plot_residual_counts


def constant(_): return np.zeros((1,1))


def test_split_impute_lattice_and_windows():
    assert chronological_bounds(11)==(7,9)
    np.testing.assert_array_equal(impute_training_only([1,np.nan,np.nan,4,5],3),[1,1,1,4,5])
    with pytest.raises(ValueError): impute_training_only([np.nan,1,2],2)
    with pytest.raises(ValueError): impute_training_only([1,2,np.nan],2)
    assert infer_decimal_quantum(["0.10","-0.20","0.30"])==Decimal("0.10")
    assert infer_decimal_quantum(["0.05","-0.10","0.15"])==Decimal("0.05")
    assert infer_decimal_quantum(["nan",None,"", "0.5"])==Decimal("0.5")
    with pytest.raises(ValueError): infer_decimal_quantum(["not-a-number"])
    with pytest.raises(ValueError): infer_decimal_quantum(["0","0.0"])
    X,y=make_windows(np.arange(12),8)
    assert X.shape==(4,8) and y.shape==(4,1)


def test_optional_pandas_missing_scalar():
    pd = pytest.importorskip("pandas")
    assert infer_decimal_quantum([pd.NA, "0.5"]) == Decimal("0.5")

def test_snap_ticks_quantizer_and_alphabet():
    np.testing.assert_allclose(snap_to_grid([31.37,-1.25],.1),[31.4,-1.2])
    with pytest.raises(ValueError): checked_ticks([.11],.1)
    with pytest.raises(ValueError): checked_ticks([True],1)
    with pytest.raises(ValueError): checked_ticks([2**53+1],1)
    q=SignedGridQuantizer(.1,-2,2)
    assert q.value_to_symbol(-.2)==0 and q.symbol_to_value(4)==.2
    with pytest.raises(ValueError): q.value_to_symbol(.3)
    alpha=fit_training_alphabet([0,.1,.2],[0,.1,.1],.1,total_frequency=32)
    assert sum(alpha.frequencies)==32 and not alpha.counts.flags.writeable
    assert alpha.counts.sum()==3
    with pytest.raises(ValueError): fit_training_alphabet([0,.1],[0],.1)
    with pytest.raises(ValueError): fit_training_alphabet([0,1e8],[0,0],.1,total_frequency=32)


def test_codec_roundtrip_evaluator_once_and_metrics():
    alpha=fit_training_alphabet(np.linspace(-2,2,41),np.zeros(41),.1,total_frequency=128)
    values=np.arange(15,dtype=float)*.1
    e,d=make_codec_pair(values[:8],constant,alpha,.1)
    np.testing.assert_array_equal(d.decode(e.encode(values[8:])),values[8:])
    from compression_evaluation.evaluator import Evaluator
    original=Evaluator.evaluate
    with patch.object(Evaluator,"evaluate",autospec=True,side_effect=lambda self,data:original(self,data)) as call:
        study=evaluate_suffix(values,constant,alpha,.1)
        assert call.call_count==1
    assert len(study.symbols)==len(values)-8
    assert study.bootstrap_bytes==study.metrics["compressed_size_bytes"]+64
    assert study.bootstrap_ratio==len(values)*8/study.bootstrap_bytes
    assert study.bootstrap_bits_per_value==8*study.bootstrap_bytes/len(values)


def test_public_series_apis_reject_integers_lost_by_float64():
    alpha=fit_training_alphabet([0,1],[0,0],1,total_frequency=32)
    valid=np.array([0,1,0,1,0,1,0,1,0],dtype=np.int64)
    assert len(evaluate_suffix(valid,constant,alpha,1).symbols)==1
    overflow=valid.copy(); overflow[-1]=2**53+1
    with pytest.raises(ValueError,match="exact float64 precision"):
        evaluate_suffix(overflow,constant,alpha,1)

    factories={k:(lambda:make_codec_pair(np.zeros(8),constant,alpha,1)) for k in "NS"}
    result=benchmark_pairs(factories,valid[8:],1,warmup_count=0)
    assert result.complete
    with pytest.raises(ValueError,match="exact float64 precision"):
        benchmark_pairs(factories,np.array([2**53+1],dtype=np.int64),1,warmup_count=0)


def test_benchmark_warmup_order_and_fake_deadline():
    alpha=fit_training_alphabet(np.linspace(-2,2,41),np.zeros(41),.1,total_frequency=128)
    factories={k:(lambda:make_codec_pair(np.zeros(8),constant,alpha,.1)) for k in "NS"}
    with patch("compression_evaluation.poc.time.perf_counter",side_effect=[0]*30), patch("compression_evaluation.poc.time.perf_counter_ns",side_effect=range(100,1000,10)):
        result=benchmark_pairs(factories,np.arange(10)*.1,.1,warmup_count=2)
    assert result.complete and [r.kind for r in result.runs]==list("NSSNNS")
    with pytest.raises(ValueError): benchmark_pairs(factories,np.arange(10)*.1,.1,order="NN",warmup_count=0)
    with patch("compression_evaluation.poc.time.perf_counter",side_effect=[0,2]):
        incomplete=benchmark_pairs(factories,np.arange(10)*.1,.1,warmup_count=0,deadline_seconds=1)
    assert not incomplete.complete and incomplete.encode_speedup is None


def test_affinity_restore_and_plots_no_show():
    original=os.sched_getaffinity(0) if hasattr(os,"sched_getaffinity") else None
    if original is not None:
        with pytest.raises(RuntimeError):
            with pinned_allowed_cpu(): raise RuntimeError
        assert os.sched_getaffinity(0)==original
    alpha=fit_training_alphabet([0,.1,-.1],[0,0,0],.1,total_frequency=32)
    with patch("matplotlib.pyplot.show",side_effect=AssertionError("show")):
        figs=[plot_training_series([1,2,3]),plot_training_distribution([-.1,0,.1],.1),plot_residual_counts(alpha,.1)]
        with pytest.raises(ValueError): plot_training_distribution([.11],.1)
    assert all(isinstance(f,plt.Figure) for f in figs)
    for f in figs: plt.close(f)
