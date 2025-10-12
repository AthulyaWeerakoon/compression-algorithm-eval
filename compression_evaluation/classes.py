from lib2to3.pygram import Symbols
from .types import Predictor, Coder, Quantizer, Encoder, Decoder, FrequencyTable, RegressorEnvelop
from typing import Sequence, List
import numpy as np


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

    def encode(self, data: Sequence[float]) -> bytes:
        xs = np.asarray(data, dtype=float).ravel()
        n = len(xs)
        all_syms = []

        for i in range(0, n, self.n_predictions):
            # Handle the last incomplete block properly
            pred_count = min(self.n_predictions, n - i)
            preds = np.asarray(self.predictor.predict(pred_count), dtype=float)
            assert preds.shape[0] == pred_count, \
                f"predictor.predict({pred_count}) must return {pred_count} predictions, got {preds.shape[0]}"

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

    def decode(self, bitstream: bytes) -> List[float]:
        syms = self.coder.decode_symbols(bitstream)
        recon = []
        idx = 0
        total_syms = len(syms)

        while idx < total_syms:
            pred_count = min(self.n_predictions, total_syms - idx)
            preds = np.asarray(self.predictor.predict(pred_count), dtype=float)

            block_syms = syms[idx:idx + pred_count]
            residuals = [self.quantizer.symbol_to_value(sym) for sym in block_syms]
            block_recon = [float(p) + float(r) for p, r in zip(preds, residuals)]

            recon.extend(block_recon)
            self.predictor.update(residuals)
            idx += pred_count

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

    def __init__(self, freq_table: FrequencyTable, quantizer: Quantizer ):
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
