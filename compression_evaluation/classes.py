from .types import *
from typing import Sequence, List
import numpy as np
from .methods import compute_posterior_mixture, approximate_mixture_map_pdf


class UniformQuantizer(Quantizer):
    """
    Uniform scalar quantizer for residuals.

    Maps residual -> integer symbol via rounding: q = round((residual - min_residual) / step).
    Symbol mapping is from 0 to levels - 1.
    Reconstruction: residual_hat = symbol * step + min_residual
    """

    def __init__(self, step: float, levels: int = 65536, min_residual: float = None, clip: bool = True):
        assert step > 0
        assert levels >= 2
        self.step = float(step)
        self.levels = int(levels)
        self.clip = bool(clip)

        # If min_residual is not provided, default to symmetric around zero
        if min_residual is None:
            self.min_residual = - (levels // 2) * step
        else:
            self.min_residual = float(min_residual)

        self.max_residual = self.min_residual + (levels - 1) * step
        self.min_sym = 0
        self.max_sym = levels - 1

    def value_to_symbol(self, residual: float) -> int:
        q = int(np.round((residual - self.min_residual) / self.step))
        if self.clip:
            q = max(self.min_sym, min(q, self.max_sym))
        else:
            if not (self.min_sym <= q <= self.max_sym):
                raise ValueError("Quantized symbol out of range.")
        return q

    def symbol_to_value(self, symbol: int) -> float:
        return symbol * self.step + self.min_residual

    def symbol_range(self) -> int:
        return self.levels

    def get_step_size(self) -> float:
        return self.step


class UniformQuantizerByRange(Quantizer):
    """
    Range-based uniform quantizer.

    Computes step size automatically from [min_val, max_val] and number of levels:
        step = (max_val - min_val) / (levels - 1)

    - residual_to_symbol: maps values in [min_val, max_val] to integers [0 .. levels-1]
    - symbol_to_residual: reconstructs a float value from integer symbol
    """

    def __init__(self, min_val: float, max_val: float, levels: int = 65536, clip: bool = True):
        assert max_val > min_val, "max_val must be greater than min_val"
        assert levels >= 2, "levels must be >= 2"

        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.levels = int(levels)
        self.step = (self.max_val - self.min_val) / (self.levels - 1)
        self.clip = bool(clip)

    def value_to_symbol(self, residual: float) -> int:
        """Quantize value in [min_val, max_val] to integer symbol."""
        q = int(np.round((residual - self.min_val) / self.step))
        if self.clip:
            q = max(0, min(q, self.levels - 1))
        else:
            if not (0 <= q < self.levels):
                raise ValueError("Value outside quantization range.")
        return q

    def symbol_to_value(self, symbol: int) -> float:
        """De-quantize integer symbol back to representative value."""
        if not (0 <= symbol < self.levels):
            raise ValueError("Symbol out of range.")
        return self.min_val + symbol * self.step

    def symbol_range(self) -> int:
        return self.levels

    def get_step_size(self) -> float:
        return self.step


class PCEncoder(Encoder):
    """
    LPC-style encoder with fixed predictor length and pluggable Coder and Quantizer.

    Workflow:
      1. predictor.predict(n_predictions)
      2. residual = actual - pred
      3. q_sym = quantizer.residual_to_symbol(residual)
      4. predictor.update(reconstructed_residuals)
      5. coder.encode_symbols(q_syms)
    """

    def __init__(
            self,
            predictor: Predictor,
            coder: Coder,
            quantizer: Quantizer = None,
            n_predictions: int = 1,
    ):
        assert hasattr(predictor, "predict") and hasattr(predictor, "update"), \
            "predictor must implement predict(n:int) and update(seq[float])"
        assert hasattr(coder, "encode_symbols") and hasattr(coder, "decode_symbols"), \
            "coder must implement encode_symbols and decode_symbols"
        self.predictor = predictor
        self.coder = coder
        self.quantizer = quantizer if quantizer is not None else UniformQuantizer(step=1.0)
        self.n_predictions = int(n_predictions)
        if self.n_predictions <= 0:
            raise ValueError("n_predictions must be positive")

    def encode(self, data: Sequence[float]) -> bytes:
        try:
            xs = np.asarray(data)
        except (TypeError, ValueError) as exc:
            raise ValueError("data must be a nonempty finite numeric 1D series") from exc
        if xs.ndim != 1 or xs.size == 0 or xs.dtype.kind not in "iuf" or not np.all(np.isfinite(xs)):
            raise ValueError("data must be a nonempty finite numeric 1D series")
        with np.errstate(over="ignore", invalid="ignore"):
            xs = xs.astype(float)
        if not np.all(np.isfinite(xs)):
            raise ValueError("data must be representable as finite float64")
        n = len(xs)
        all_syms = []

        for i in range(0, n, self.n_predictions):
            # Handle the last incomplete block properly
            pred_count = min(self.n_predictions, n - i)
            preds = np.asarray(self.predictor.predict(pred_count), dtype=float)
            if preds.shape != (pred_count,):
                raise ValueError(f"predictor.predict({pred_count}) must return shape {(pred_count,)}")

            actual = xs[i:i + pred_count]
            residuals = actual - preds
            syms = [self.quantizer.value_to_symbol(float(r)) for r in residuals]

            # Reconstructed residuals for feedback update
            recon_residuals = [self.quantizer.symbol_to_value(sym) for sym in syms]
            self.predictor.update(recon_residuals)

            all_syms.extend(syms)

        return self.coder.encode_symbols(all_syms)


class PCDecoder(Decoder):
    """
    LPC-style decoder with fixed prediction length and pluggable Coder and Quantizer.

    Workflow:
      1. preds = predictor.predict(n_predictions)
      2. q_syms = coder.decode_symbols(bitstream)
      3. reconstructed = preds + quantizer.symbol_to_residual(sym)
      4. predictor.update(reconstructed_residuals)
    """

    def __init__(
            self,
            predictor: Predictor,
            coder: Coder,
            quantizer: Quantizer = None,
            n_predictions: int = 1,
    ):
        assert hasattr(predictor, "predict") and hasattr(predictor, "update"), \
            "predictor must implement predict(n:int) and update(seq[float])"
        assert hasattr(coder, "encode_symbols") and hasattr(coder, "decode_symbols"), \
            "coder must implement encode_symbols and decode_symbols"
        self.predictor = predictor
        self.coder = coder
        self.quantizer = quantizer if quantizer is not None else UniformQuantizer(step=1.0)
        self.n_predictions = int(n_predictions)
        if self.n_predictions <= 0:
            raise ValueError("n_predictions must be positive")

    def decode(self, bitstream: bytes) -> List[float]:
        if not isinstance(bitstream, (bytes, bytearray)) or not bitstream:
            raise ValueError("bitstream must be nonempty bytes")
        syms = self.coder.decode_symbols(bitstream)
        if not syms:
            raise ValueError("decoded stream is empty")
        recon = []
        idx = 0
        total_syms = len(syms)

        while idx < total_syms:
            pred_count = min(self.n_predictions, total_syms - idx)
            preds = np.asarray(self.predictor.predict(pred_count), dtype=float)
            if preds.shape != (pred_count,):
                raise ValueError(f"predictor.predict({pred_count}) must return shape {(pred_count,)}")

            block_syms = syms[idx:idx + pred_count]
            residuals = [self.quantizer.symbol_to_value(sym) for sym in block_syms]
            block_recon = [float(p) + float(r) for p, r in zip(preds, residuals)]

            recon.extend(block_recon)
            self.predictor.update(residuals)
            idx += pred_count

        return recon


class MultivariatePCEncoder(Encoder):
    """
    Multivariate LPC-style predictive encoder.

    Assumptions about the predictor:
      - predictor.predict(n) returns a 2D array of shape (D, n)
      - predictor.update(residuals) consumes a 2D array of shape (D, n)
      - predictor internally maintains any sliding window / state

    Workflow (per block):
      1. preds = predictor.predict(n)
      2. residuals = actual - preds
      3. residuals quantized independently per sequence
      4. predictor.update(reconstructed_residuals)
      5. symbols entropy-coded per sequence
    """

    def __init__(
        self,
        predictor: MultivariatePredictor,
        coders: list,
        quantizers: list,
        n_predictions: int = 1,
    ):
        assert hasattr(predictor, "predict") and hasattr(predictor, "update"), \
            "predictor must implement predict(n) and update(residuals_2d)"
        if len(coders) == 0 or len(coders) != len(quantizers):
            raise ValueError("Must provide matching nonempty coders and quantizers")

        self.predictor = predictor
        self.coders = coders
        self.quantizers = quantizers
        self.D = len(coders)
        self.n_predictions = int(n_predictions)
        if self.n_predictions <= 0:
            raise ValueError("n_predictions must be positive")

    def encode(self, data: np.ndarray) -> list:
        try:
            data = np.asarray(data)
        except (TypeError, ValueError) as exc:
            raise ValueError("data must be finite numeric (T, D) with configured D") from exc
        if (
            data.ndim != 2 or not data.shape[0] or data.shape[1] != self.D
            or data.dtype.kind not in "iuf" or not np.all(np.isfinite(data))
        ):
            raise ValueError("data must be finite numeric (T, D) with configured D")
        try:
            data = data.astype(float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("data must be representable as float64") from exc
        if not np.all(np.isfinite(data)):
            raise ValueError("data must be representable as finite float64")
        T, D = data.shape
        all_syms = [[] for _ in range(D)]
        for t in range(0, T, self.n_predictions):
            n = min(self.n_predictions, T-t)
            try:
                preds = np.asarray(self.predictor.predict(n), dtype=float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("predictor output must be a numeric array") from exc
            if preds.shape != (D, n):
                raise ValueError(f"predictor must return shape {(D,n)}")
            residuals = data[t:t+n] - preds.T
            reconstructed = np.zeros_like(residuals)
            for d in range(D):
                syms = [self.quantizers[d].value_to_symbol(float(x)) for x in residuals[:, d]]
                all_syms[d].extend(syms)
                reconstructed[:, d] = [self.quantizers[d].symbol_to_value(x) for x in syms]
            self.predictor.update(reconstructed.T)
        return [self.coders[d].encode_symbols(all_syms[d]) for d in range(D)]

class MultivariatePCDecoder(Decoder):
    """
    Multivariate LPC-style predictive decoder.

    Assumptions about the predictor:
      - predictor.predict(n) returns a 2D array of shape (D, n)
      - predictor.update(residuals_2d) consumes a 2D array of shape (D, n)
      - predictor internally maintains any sliding window / state

    Bit-exact inverse of MultivariatePCEncoder.
    """

    def __init__(
        self,
        predictor: MultivariatePredictor,
        coders: list,
        quantizers: list,
        n_predictions: int = 1,
    ):
        assert hasattr(predictor, "predict") and hasattr(predictor, "update"), \
            "predictor must implement predict(n) and update(residuals_2d)"
        if len(coders) == 0 or len(coders) != len(quantizers):
            raise ValueError("Must provide matching nonempty coders and quantizers")

        self.predictor = predictor
        self.coders = coders
        self.quantizers = quantizers
        self.D = len(coders)
        self.n_predictions = int(n_predictions)
        if self.n_predictions <= 0:
            raise ValueError("n_predictions must be positive")

    def decode(self, bitstreams: list) -> np.ndarray:
        if not isinstance(bitstreams, list) or len(bitstreams) != self.D or any(not isinstance(x, (bytes, bytearray)) or not x for x in bitstreams):
            raise ValueError("expected exactly D nonempty byte streams")
        syms = [self.coders[d].decode_symbols(bitstreams[d]) for d in range(self.D)]
        if not syms[0] or any(len(x) != len(syms[0]) for x in syms):
            raise ValueError("decoded channel lengths must be equal and nonzero")
        T = len(syms[0]); recon = np.zeros((T,self.D), dtype=float)
        for t in range(0,T,self.n_predictions):
            n=min(self.n_predictions,T-t)
            preds=np.asarray(self.predictor.predict(n),dtype=float)
            if preds.shape != (self.D,n): raise ValueError(f"predictor must return shape {(self.D,n)}")
            residuals=np.empty((n,self.D),dtype=float)
            for d in range(self.D): residuals[:,d]=[self.quantizers[d].symbol_to_value(x) for x in syms[d][t:t+n]]
            recon[t:t+n]=preds.T+residuals
            self.predictor.update(residuals.T)
        return recon

class SimpleFrequencyTable(FrequencyTable):
    """
    Simple frequency table implementing the ANS FrequencyTable protocol.
    Assumes that symbols are integers which start from 0, and corresponds to the index of each frequency in the
    initialization frequency list
    """

    def __init__(self, freqs: list[int]):
        if any(f < 0 for f in freqs):
            raise ValueError("Frequencies must be non-negative")
        self._freqs = freqs
        self._cumulative = [0]
        for f in freqs:
            self._cumulative.append(self._cumulative[-1] + f)
        self._total = self._cumulative[-1]

    def freq(self, symbol: int) -> int:
        return self._freqs[symbol]

    def cum_freq(self, symbol: int) -> int:
        return self._cumulative[symbol]

    def symbol_from_cum(self, cum_value: int) -> int:
        """Binary search for symbol corresponding to cumulative frequency."""
        if not (0 <= cum_value < self._total):
            raise ValueError("cum_value out of range")
        low, high = 0, len(self._freqs) - 1
        while low <= high:
            mid = (low + high) // 2
            if self._cumulative[mid + 1] <= cum_value:
                low = mid + 1
            elif self._cumulative[mid] > cum_value:
                high = mid - 1
            else:
                return mid
        raise RuntimeError("Failed to find symbol for cumulative frequency")

    @property
    def total(self) -> int:
        return self._total


class ANSCoder(Coder):
    """Simple rANS encoder"""

    def __init__(self, freq_table: FrequencyTable, quantizer: Quantizer | None = None):
        self.ft = freq_table
        self.quantizer = quantizer

    def encode_symbols(self, data: Sequence[int]) -> bytes:
        state = 1
        ft = self.ft

        # Encode in reverse
        for symbol in reversed(data):
            freq = ft.freq(symbol)
            cum = ft.cum_freq(symbol)
            state = (state // freq) * ft.total + cum + (state % freq)

        # Just turn final state into bytes (big endian)
        return state.to_bytes((state.bit_length() + 7) // 8, 'big')

    def decode_symbols(self, bitstream: bytes) -> List[int]:
        ft = self.ft
        # Convert bytes back to integer state
        state = int.from_bytes(bitstream, 'big')

        decoded = []
        while state > 1:  # until state returns to initial region
            x = state % ft.total
            s = ft.symbol_from_cum(x)
            freq = ft.freq(s)
            cum = ft.cum_freq(s)
            state = freq * (state // ft.total) + (x - cum)
            decoded.append(s)

        return decoded


class StaticResidualRegressor(Predictor):
    """
    A residual-based predictive model that maintains a fixed-size input window.

    It uses a wrapped regressor (e.g., linear, AR, neural net) to predict the next
    value(s) from the current window, then updates the window using quantized residuals.
    """

    def __init__(self, regressor: RegressorEnvelop, input_size: int):
        self._regressor = regressor
        self._window_size = input_size
        self._window = np.zeros(input_size, dtype=float)
        self._predictions = np.zeros(0, dtype=float)

    def predict(self, n: int = 1) -> np.ndarray:
        """
        Make `n` predictions from the current window using the wrapped regressor.
        """
        preds = self._regressor.predict(n, self._window)
        self._predictions = preds
        return preds

    def update(self, quantized_residuals):
        """
        Update the input window by adding residuals to the previous predictions.
        The window rolls forward with the most recent reconstructed values.
        """
        quantized_residuals = np.asarray(quantized_residuals, dtype=float)

        # reconstruct predicted + residual values
        assert len(self._predictions) == len(quantized_residuals), "Residual count must match the prediction length"
        reconstructed = self._predictions + quantized_residuals

        # roll window and insert new reconstructed values
        total_new = len(reconstructed)
        if total_new >= self._window_size:
            # if residuals exceed window size, keep only last window_size values
            self._window = reconstructed[-self._window_size:]
        else:
            self._window = np.roll(self._window, -total_new)
            self._window[-total_new:] = reconstructed


class MultiVariateStaticResidualRegressor(MultivariatePredictor):
    """
    A residual-based predictive model for batched (multi-window) inputs.

    Each row of the internal window buffer represents a separate time series window.
    The wrapped regressor (conforming to MultiVariateRegressorEnvelop) predicts the
    next `n` values for each window, and the regressor updates its windows using
    quantized residuals added to the predicted outputs.
    """

    def __init__(self, regressor: MultiVariateRegressorEnvelop, num_windows: int, input_size: int):
        """
        Initialize the batched residual regressor.

        Args:
            regressor (MultiVariateRegressorEnvelop): The wrapped regressor.
            num_windows (int): Number of concurrent windows (batch size).
            input_size (int): Number of samples per input window.
        """
        self._regressor = regressor
        self._num_windows = num_windows
        self._window_size = input_size

        # Internal states
        self._windows = np.zeros((num_windows, input_size), dtype=float)
        self._predictions = np.zeros((num_windows, 0), dtype=float)

    def predict(self, n: int = 1) -> np.ndarray:
        """
        Make `n` predictions for each input window.

        Returns:
            np.ndarray: 2D array of shape [num_windows, n]
        """
        preds = self._regressor.predict(n, self._windows)

        assert isinstance(preds, np.ndarray), "Regressor must return a numpy array"
        assert preds.ndim == 2, f"Expected 2D array output, got {preds.ndim}D"
        assert preds.shape == (self._num_windows, n), (
            f"Expected prediction shape ({self._num_windows}, {n}), got {preds.shape}"
        )

        self._predictions = preds
        return preds

    def update(self, quantized_residuals: np.ndarray):
        """
        Update all input windows with quantized residuals.

        Args:
            quantized_residuals (np.ndarray): Array of shape [num_windows, n]
                                              corresponding to last predictions.
        """
        quantized_residuals = np.asarray(quantized_residuals, dtype=float)

        assert quantized_residuals.shape == self._predictions.shape, (
            f"Residual shape {quantized_residuals.shape} must match prediction shape {self._predictions.shape}"
        )

        reconstructed = self._predictions + quantized_residuals

        n = reconstructed.shape[1]
        if n >= self._window_size:
            self._windows = reconstructed[:, -self._window_size:]
        else:
            self._windows = np.roll(self._windows, -n, axis=1)
            self._windows[:, -n:] = reconstructed

        assert self._windows.shape == (self._num_windows, self._window_size), (
            f"Updated window shape mismatch: expected ({self._num_windows}, {self._window_size}), got {self._windows.shape}"
        )


class ANSEncoder(Encoder):
    """Simple rANS encoder"""

    def __init__(self, freq_table: FrequencyTable, quantizer: Quantizer):
        self.ft = freq_table
        self.quantizer = quantizer

    def encode(self, data: Sequence[int]) -> bytes:
        state = 1
        ft = self.ft

        symbols = [self.quantizer.value_to_symbol(val) for val in data]

        # Encode in reverse
        for symbol in reversed(symbols):
            freq = ft.freq(symbol)
            cum = ft.cum_freq(symbol)
            state = (state // freq) * ft.total + cum + (state % freq)

        # Just turn final state into bytes (big endian)
        return state.to_bytes((state.bit_length() + 7) // 8, 'big')


class ANSDecoder(Decoder):
    """Simple rANS decoder"""

    def __init__(self, freq_table: FrequencyTable, quantizer: Quantizer):
        self.ft = freq_table
        self.quantizer = quantizer

    def decode(self, bitstream: bytes) -> List[int]:
        ft = self.ft
        # Convert bytes back to integer state
        state = int.from_bytes(bitstream, 'big')

        decoded = []
        while state > 1:  # until state returns to initial region
            x = state % ft.total
            s = ft.symbol_from_cum(x)
            freq = ft.freq(s)
            cum = ft.cum_freq(s)
            state = freq * (state // ft.total) + (x - cum)
            decoded.append(s)

        decoded_value = [self.quantizer.symbol_to_value(sym) for sym in decoded]
        return decoded_value


class DictionaryFrequencyTable(FrequencyTable):
    """
    Frequency table using a dictionary {symbol: frequency} for ANS.
    """

    def __init__(self, freqs: dict[int, int]):
        if any(f < 0 for f in freqs.values()):
            raise ValueError("Frequencies must be non-negative")

        self._freqs = freqs.copy()
        self._symbols = sorted(freqs.keys())
        self._cumulative = {0: 0}
        cum = 0
        self._cum_list = []
        for sym in self._symbols:
            self._cumulative[sym] = cum
            self._cum_list.append((sym, cum))
            cum += freqs[sym]

        self._total = cum

    def freq(self, symbol: int) -> int:
        return self._freqs[symbol]

    def cum_freq(self, symbol: int) -> int:
        return self._cumulative[symbol]

    def symbol_from_cum(self, cum_value: int) -> int:
        """Binary search for symbol corresponding to cumulative frequency."""
        if not (0 <= cum_value < self._total):
            raise ValueError("cum_value out of range")

        low, high = 0, len(self._cum_list) - 1
        while low <= high:
            mid = (low + high) // 2
            sym, cum = self._cum_list[mid]
            next_cum = self._cum_list[mid + 1][1] if mid + 1 < len(self._cum_list) else self._total

            if cum_value < cum:
                high = mid - 1
            elif cum_value >= next_cum:
                low = mid + 1
            else:
                return sym

        raise RuntimeError("Failed to find symbol for cumulative frequency")

    @property
    def total(self) -> int:
        return self._total


class BpcMapPredictor(RegressorEnvelop):
    def __init__(self, prior_mixture: list[ParametricDistribution]):
        self._prior_mixture = prior_mixture

    def predict(self, n: int, input_window: np.ndarray) -> np.ndarray:
        posterior = compute_posterior_mixture(self._prior_mixture, input_window)
        return np.full(n, approximate_mixture_map_pdf(posterior))
