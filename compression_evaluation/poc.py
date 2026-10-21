from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import math
import os
import time
from typing import Callable, Iterable, Mapping

import numpy as np

from .classes import ANSCoder, PCDecoder, PCEncoder, SimpleFrequencyTable, StaticResidualRegressor
from .methods import largest_remainder_quantize
from .types import RegressorEnvelop


def chronological_bounds(n: int) -> tuple[int, int]:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0: raise ValueError("n must be nonnegative integer")
    return 7 * int(n) // 10, 9 * int(n) // 10


def impute_training_only(values, train_end: int, validation_end: int | None = None) -> np.ndarray:
    a=np.asarray(values,dtype=np.float64)
    if a.ndim!=1 or not a.size or isinstance(train_end,bool) or not isinstance(train_end,(int,np.integer)) or not 0<train_end<=a.size: raise ValueError("invalid series or training boundary")
    ve=a.size if validation_end is None else validation_end
    if not train_end<=ve<=a.size or np.isinf(a).any(): raise ValueError("invalid boundary or infinite input")
    missing=np.isnan(a)
    if missing[0] or missing[train_end:].any(): raise ValueError("leading or validation/test missing values are not permitted")
    out=a.copy()
    for i in range(1,int(train_end)):
        if missing[i]: out[i]=out[i-1]
    return out


def make_windows(values,input_size:int=8)->tuple[np.ndarray,np.ndarray]:
    a=np.asarray(values,dtype=np.float64)
    if a.ndim!=1 or isinstance(input_size,bool) or not isinstance(input_size,(int,np.integer)) or input_size<=0 or len(a)<=input_size or not np.isfinite(a).all(): raise ValueError("invalid window input")
    w=np.lib.stride_tricks.sliding_window_view(a,input_size+1)
    return w[:,:-1].copy(),w[:,-1:].copy()


def infer_decimal_quantum(training_source_strings:Iterable)->Decimal:
    vals=[]
    for x in training_source_strings:
        if x is None or str(x).strip().casefold() == "<na>" or (isinstance(x,str) and (not x.strip() or x.strip().casefold() in {"nan", "nat"})): continue
        try:
            if bool(x!=x): continue
        except (TypeError,ValueError): pass
        try: d=Decimal(str(x).strip())
        except (InvalidOperation,ValueError) as e: raise ValueError("invalid decimal source value") from e
        if not d.is_finite(): raise ValueError("source values must be finite")
        vals.append(d)
    if not vals: raise ValueError("no observed training source values")
    exp=min(d.as_tuple().exponent for d in vals); scale=10**max(0,-exp)
    from math import gcd
    lattice=0
    for d in vals: lattice=gcd(lattice,abs(int(d*scale)))
    if not lattice: raise ValueError("all training values are zero")
    return Decimal(lattice)/Decimal(scale)


def _ticks(values,quantum):
    raw=np.asarray(values)
    if raw.dtype.kind not in "iuf": raise ValueError("real numeric values required")
    # Float conversion can erase integer units above 2**53 (including uint64
    # and Python integers); reject rather than validate a changed source value.
    if raw.dtype.kind in "iu" and np.any(raw.astype(object) > 2**53):
        raise ValueError("integer values exceed exact float64 precision")
    if raw.dtype.kind in "iu" and np.any(raw.astype(object) < -(2**53)):
        raise ValueError("integer values exceed exact float64 precision")
    if raw.dtype.kind == "f" and raw.dtype.itemsize > 8:
        obj=raw.astype(object)
        if any(np.isfinite(x) and float(x) != x for x in obj.flat): raise ValueError("values are not exactly representable as float64")
    a=raw.astype(np.float64); q=float(quantum)
    if not math.isfinite(q) or q<=0 or not np.isfinite(a).all(): raise ValueError("finite data and positive quantum required")
    scaled=a/q; rnd=np.rint(scaled); limit=float(2**63)
    if not np.isfinite(scaled).all() or np.any(rnd < -limit) or np.any(rnd >= limit) or np.any(np.abs(scaled-rnd)>1e-12): raise ValueError("values not safely on int64 grid")
    return rnd.astype(np.int64)


def checked_ticks(values,quantum,label="values"):
    try: return _ticks(values,quantum)
    except (TypeError,ValueError,OverflowError) as e: raise ValueError(f"{label}: {e}") from e


def snap_to_grid(values,quantum):
    raw=np.asarray(values)
    if raw.dtype.kind not in "iuf": raise ValueError("predictions must be real numeric")
    a=raw.astype(np.float64); q=float(quantum)
    if not np.isfinite(a).all() or not math.isfinite(q) or q<=0: raise ValueError("finite predictions and positive quantum required")
    scaled=a/q; rnd=np.rint(scaled); limit=float(2**63)
    if not np.isfinite(scaled).all() or np.any(rnd < -limit) or np.any(rnd >= limit): raise ValueError("prediction ticks exceed int64")
    return rnd*q


class SignedGridQuantizer:
    def __init__(self,quantum,low_tick,high_tick):
        if isinstance(low_tick,bool) or isinstance(high_tick,bool) or not isinstance(low_tick,(int,np.integer)) or not isinstance(high_tick,(int,np.integer)) or low_tick>high_tick: raise ValueError("invalid tick bounds")
        if int(low_tick)<np.iinfo(np.int64).min or int(high_tick)>np.iinfo(np.int64).max: raise ValueError("tick bounds exceed int64")
        if not math.isfinite(float(quantum)) or quantum<=0: raise ValueError("positive finite quantum required")
        self.quantum=float(quantum); self.low_tick=int(low_tick); self.high_tick=int(high_tick)
        self.levels=self.high_tick-self.low_tick+1
        if self.levels<=0 or self.levels>np.iinfo(np.int64).max: raise ValueError("unsupported alphabet size")
    def value_to_symbol(self,value):
        tick=int(checked_ticks([value],self.quantum)[0])
        if not self.low_tick<=tick<=self.high_tick: raise ValueError("value outside trained alphabet")
        return tick-self.low_tick
    def symbol_to_value(self,symbol):
        if isinstance(symbol,bool) or not isinstance(symbol,(int,np.integer)) or not 0<=symbol<self.levels: raise ValueError("symbol outside alphabet")
        return (self.low_tick+int(symbol))*self.quantum
    def symbol_range(self): return self.levels


@dataclass(frozen=True)
class TrainedAlphabet:
    quantizer:SignedGridQuantizer; frequencies:tuple[int,...]; counts:np.ndarray; low_tick:int; high_tick:int
    def __post_init__(self):
        c=np.asarray(self.counts,dtype=np.int64).copy(); c.flags.writeable=False
        object.__setattr__(self,"counts",c); object.__setattr__(self,"frequencies",tuple(map(int,self.frequencies)))


def fit_training_alphabet(targets,predictions,quantum,pad_ticks=5,total_frequency=1024):
    t=checked_ticks(targets,quantum,"targets"); p=checked_ticks(snap_to_grid(predictions,quantum),quantum,"predictions")
    if not t.size or t.shape!=p.shape or isinstance(pad_ticks,bool) or not isinstance(pad_ticks,(int,np.integer)) or pad_ticks<0: raise ValueError("matching nonempty arrays and valid padding required")
    if isinstance(total_frequency,bool) or not isinstance(total_frequency,(int,np.integer)) or total_frequency<1: raise ValueError("invalid total frequency")
    lim=np.iinfo(np.int64); diffs=[int(x)-int(y) for x,y in zip(t,p)]
    if any(x<lim.min or x>lim.max for x in diffs): raise ValueError("residual subtraction exceeds int64")
    r=np.asarray(diffs,dtype=np.int64); lo=int(r.min())-int(pad_ticks); hi=int(r.max())+int(pad_ticks)
    if lo<lim.min or hi>lim.max: raise ValueError("alphabet bounds exceed int64")
    span=hi-lo+1
    if span>int(total_frequency): raise ValueError("alphabet span exceeds available positive frequencies")
    counts=np.bincount(r-lo,minlength=span).astype(np.int64); smooth=counts+1
    freq=tuple(largest_remainder_quantize(smooth.tolist(),int(total_frequency)))
    return TrainedAlphabet(SignedGridQuantizer(quantum,lo,hi),freq,counts,lo,hi)


def make_codec_pair(bootstrap:np.ndarray,predictor:Callable[[np.ndarray],np.ndarray],alphabet:TrainedAlphabet,quantum:float,*,input_size:int=8)->tuple[PCEncoder,PCDecoder]:
    b=np.asarray(bootstrap,dtype=np.float64)
    if isinstance(input_size,bool) or not isinstance(input_size,int) or input_size<=0 or b.shape!=(input_size,) or not np.isfinite(b).all(): raise ValueError("bootstrap must be finite exact-size vector")
    checked_ticks(b,quantum,"bootstrap")
    class Envelope(RegressorEnvelop):
        def predict(self,n,input_window):
            w=np.asarray(input_window,dtype=np.float64)
            if n!=1 or w.shape!=(input_size,): raise ValueError("expected one exact window")
            out=np.asarray(predictor(w.reshape(1,-1).copy()))
            if out.shape!=(1,1) or out.dtype.kind not in "iuf" or not np.isfinite(out).all(): raise ValueError("predictor must return finite numeric (1,1)")
            return snap_to_grid(out,quantum).reshape(1)
    e,d=StaticResidualRegressor(Envelope(),input_size),StaticResidualRegressor(Envelope(),input_size)
    e._window=b.copy(); d._window=b.copy()
    coder=ANSCoder(SimpleFrequencyTable(list(alphabet.frequencies)),alphabet.quantizer)
    return PCEncoder(e,coder,alphabet.quantizer,1),PCDecoder(d,coder,alphabet.quantizer,1)


@dataclass(frozen=True)
class ValidationStudy:
    metrics:dict; symbols:np.ndarray; counts:np.ndarray; entropy_bits:float; cross_entropy_bits:float; mismatch_gap_bits:float
    zero_fraction:float; plus_one_fraction:float; minus_one_fraction:float; mean_abs_residual:float; occupied_bins:int
    bootstrap_bytes:int; bootstrap_ratio:float; bootstrap_bits_per_value:float


def evaluate_suffix(values,predictor,alphabet:TrainedAlphabet,quantum:float,*,input_size:int=8)->ValidationStudy:
    from .evaluator import Evaluator
    ticks=checked_ticks(values,quantum,"values"); a=np.asarray(values,dtype=np.float64)
    if a.ndim!=1 or len(a)<=input_size: raise ValueError("series too short")
    suffix=a[input_size:]; enc,dec=make_codec_pair(a[:input_size],predictor,alphabet,quantum,input_size=input_size); recorded=[]
    class Recording:
        def value_to_symbol(self,x):
            s=alphabet.quantizer.value_to_symbol(x); recorded.append(s); return s
        def symbol_to_value(self,s): return alphabet.quantizer.symbol_to_value(s)
        def symbol_range(self): return alphabet.quantizer.symbol_range()
    class EncoderGuard:
        def encode(self,data): return enc.encode(data)
    class DecoderGuard:
        def decode(self,stream):
            out=np.asarray(dec.decode(stream),dtype=np.float64)
            if not np.array_equal(checked_ticks(out,quantum,"decoded"),ticks[input_size:]): raise ValueError("decoded grid mismatch")
            return out
    enc.quantizer=Recording()
    metrics=Evaluator(EncoderGuard(),DecoderGuard()).evaluate(suffix)
    symbols=np.asarray(recorded,dtype=np.int64)
    if symbols.size!=suffix.size: raise ValueError("encoder symbol history length mismatch")
    if not np.array_equal(enc.predictor._window,dec.predictor._window): raise ValueError("histories diverged")
    counts=np.bincount(symbols,minlength=len(alphabet.frequencies)); probs=counts[counts>0]/len(symbols)
    entropy=float(-np.sum(probs*np.log2(probs))); model=np.asarray(alphabet.frequencies,dtype=float)/sum(alphabet.frequencies)
    cross=float(-np.sum((counts/len(symbols))[counts>0]*np.log2(model[counts>0])))
    residual=np.asarray([alphabet.quantizer.symbol_to_value(int(s)) for s in symbols]);
    def fraction(tick): return float(np.mean(symbols==tick)) if 0<=tick<len(alphabet.frequencies) else 0.0
    try: z=alphabet.quantizer.value_to_symbol(0.)
    except ValueError: z=-1
    try: plus=alphabet.quantizer.value_to_symbol(quantum)
    except ValueError: plus=-1
    try: minus=alphabet.quantizer.value_to_symbol(-quantum)
    except ValueError: minus=-1
    b=int(metrics["compressed_size_bytes"])+64; n=len(a)
    return ValidationStudy(metrics,symbols.copy(),counts,entropy,cross,cross-entropy,fraction(z),fraction(plus),fraction(minus),float(np.mean(np.abs(residual))),int(np.count_nonzero(counts)),b,float(n*8/b),float(8*b/n))


@dataclass(frozen=True)
class TimedRun:
    kind:str; encode_s:float; decode_s:float; payload_bytes:int
@dataclass(frozen=True)
class BenchmarkResult:
    runs:tuple[TimedRun,...]; complete:bool; reason:str|None; encode_medians:dict|None; decode_medians:dict|None
    encode_iqr:dict|None; decode_iqr:dict|None; encode_speedup:float|None; decode_speedup:float|None


def benchmark_pairs(factories:Mapping[str,Callable],suffix:np.ndarray,quantum:float,*,order="NSSNNS",warmup_count=256,deadline_seconds=1500.)->BenchmarkResult:
    from .classes import PCDecoder,PCEncoder
    from statistics import median
    checked_ticks(suffix,quantum,"suffix"); a=np.asarray(suffix,dtype=np.float64)
    if a.ndim!=1 or not a.size or set(factories)!={"N","S"} or not order or any(k not in "NS" for k in order) or not {"N","S"}.issubset(set(order)) or isinstance(warmup_count,bool) or warmup_count<0 or not np.isfinite(deadline_seconds) or deadline_seconds<=0: raise ValueError("invalid benchmark settings")
    start=time.perf_counter(); runs=[]; used=set()
    def pair(kind):
        e,d=factories[kind]()
        if not isinstance(e,PCEncoder) or not isinstance(d,PCDecoder) or e is d or e.predictor is d.predictor or e.predictor._window is d.predictor._window: raise ValueError("factory must return distinct codec pair")
        if not np.array_equal(e.predictor._window,d.predictor._window): raise ValueError("pair bootstrap histories differ")
        return e,d
    def run(kind,data,timed):
        e,d=pair(kind)
        te=time.perf_counter_ns(); payload=e.encode(data); enc=(time.perf_counter_ns()-te)/1e9
        if not isinstance(payload,(bytes,bytearray)) or not payload: raise ValueError("invalid payload")
        td=time.perf_counter_ns(); rec=np.asarray(d.decode(payload)); dec=(time.perf_counter_ns()-td)/1e9
        if not np.isfinite(enc) or not np.isfinite(dec) or not np.array_equal(checked_ticks(rec,quantum),checked_ticks(data,quantum)): raise ValueError("invalid roundtrip or timing")
        if not np.array_equal(e.predictor._window,d.predictor._window): raise ValueError("codec histories diverged")
        if timed: runs.append(TimedRun(kind,enc,dec,len(payload)))
    if warmup_count:
        for kind in "NS":
            prefix=a[:min(len(a),warmup_count)]
            run(kind,prefix,False); used.add(kind)
            if time.perf_counter()-start>=deadline_seconds: return BenchmarkResult(tuple(runs),False,"deadline",None,None,None,None,None,None)
        if time.perf_counter()-start>=deadline_seconds: return BenchmarkResult(tuple(runs),False,"deadline",None,None,None,None,None,None)
    for kind in order:
        if time.perf_counter()-start>=deadline_seconds: return BenchmarkResult(tuple(runs),False,"deadline",None,None,None,None,None,None)
        run(kind,a,True)
        if time.perf_counter()-start>=deadline_seconds: return BenchmarkResult(tuple(runs),False,"deadline",None,None,None,None,None,None)
    def agg(attr): return {k:float(np.median([getattr(r,attr) for r in runs if r.kind==k])) for k in "NS"}
    def iqr(attr): return {k:float(np.percentile([getattr(r,attr) for r in runs if r.kind==k],75)-np.percentile([getattr(r,attr) for r in runs if r.kind==k],25)) for k in "NS"}
    em,dm=agg("encode_s"),agg("decode_s")
    return BenchmarkResult(tuple(runs),True,None,em,dm,iqr("encode_s"),iqr("decode_s"),em["N"]/em["S"],dm["N"]/dm["S"])


@contextmanager
def pinned_allowed_cpu():
    if not hasattr(os,"sched_getaffinity") or not hasattr(os,"sched_setaffinity"): yield None; return
    original=os.sched_getaffinity(0)
    if not original: yield None; return
    cpu=min(original)
    try: os.sched_setaffinity(0,{cpu}); yield cpu
    finally: os.sched_setaffinity(0,original)
