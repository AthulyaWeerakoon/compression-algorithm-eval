from util.types import Predictor, Coder, Quantizer, Encoder, Decoder, FrequencyTable
from typing import Sequence, List, Optional
import numpy as np


class UniformQuantizer(Quantizer):
    """
    Uniform scalar quantizer for residuals.

    Maps residual -> integer symbol via rounding: q = round(residual / step).
    Symbol mapping has an offset so output is non-negative (0..levels-1).
    Reconstruction: residual_hat = (q - offset) * step.
    """

    def __init__(self, step: float, levels: int = 65536, clip: bool = True):
        assert step > 0, "step must be positive"
        assert levels >= 2, "levels must be >= 2"
        self.step = float(step)
        self.levels = int(levels)
        self.offset = levels // 2  # map signed ints to non-negative symbols
        self.min_sym = 0
        self.max_sym = levels - 1
        self.clip = bool(clip)

    def residual_to_symbol(self, residual: float) -> int:
        q = int(np.round(residual / self.step))
        sym = q + self.offset
        if self.clip:
            sym = max(self.min_sym, min(sym, self.max_sym))
        else:
            if not (self.min_sym <= sym <= self.max_sym):
                raise ValueError("Quantized symbol out of range.")
        return int(sym)

    def symbol_to_residual(self, symbol: int) -> float:
        q = int(symbol) - self.offset
        return q * self.step

    def symbol_range(self) -> int:
        return self.levels


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


class LPCEncoder(Encoder):
    """
    LPC-style encoder that uses a pluggable Predictor and a pluggable Coder.

    Workflow (per sample or per-batch):
      1. pred = predictor.predict(n=1 or batch_size)
      2. residual = actual - pred
      3. q_sym = quantizer.residual_to_symbol(residual)  # quantize BEFORE updating predictor
      4. predictor.update([q_sym ...])                     # predictor receives quantized residuals
      5. collect q_sym into list to pass to coder.encode_symbols

    Important: predictor.update is called *before* coder.encode_symbols so the predictor at the encoder
    has the same state as after encoding; decoder will update *after* decoding the symbol, restoring sync.
    """

    def __init__(
            self,
            predictor: Predictor,
            coder: Coder,
            quantizer: Quantizer = None,
            batch_size: int = 1,
    ):
        assert isinstance(predictor, Predictor) or (hasattr(predictor, "predict") and hasattr(predictor, "update")), \
            "predictor must implement predict(n:int) and update(seq[int])"
        assert isinstance(coder, Coder) or (hasattr(coder, "encode_symbols") and hasattr(coder, "decode_symbols")), \
            "coder must implement encode_symbols and decode_symbols"
        self.predictor = predictor
        self.coder = coder
        self.quantizer = quantizer if quantizer is not None else UniformQuantizer(step=1.0)
        self.batch_size = int(batch_size)

    def encode(self, data: Sequence[float]) -> bytes:
        """
        Encode a 1D sequence of numeric data into bytes.
        Returns: bytes (the output bitstream from coder)
        """
        xs = np.asarray(data, dtype=float).ravel()
        n = xs.shape[0]
        symbols = []
        i = 0
        while i < n:
            b = min(self.batch_size, n - i)
            preds = np.asarray(self.predictor.predict(b), dtype=float)
            assert preds.shape[0] == b, "predictor.predict(b) must return b predictions"

            actuals = xs[i:i + b]
            residuals = actuals - preds  # float residuals

            # Quantize residuals to symbols
            syms = [self.quantizer.residual_to_symbol(float(r)) for r in residuals]

            # Update predictor with quantized residuals BEFORE encoding (encoder-side update)
            self.predictor.update(syms)

            symbols.extend(syms)
            i += b

        # encode all symbols using coder
        bitstream = self.coder.encode_symbols(symbols)
        return bitstream


class LPCDecoder(Decoder):
    """
    LPC-style decoder that uses the same Predictor and Coder types as encoder.

    Workflow (per sample or per-batch):
      1. preds = predictor.predict(n=batch_size)   # prediction BEFORE seeing quantized residual
      2. q_syms = coder.decode_symbols(bitstream, n=batch_size)
      3. reconstruct = preds + quantizer.symbol_to_residual(q_sym)
      4. predictor.update(q_syms)                   # update AFTER decoding so next predictions match encoder
    """

    def __init__(
            self,
            predictor: Predictor,
            coder: Coder,
            quantizer: Quantizer = None,
            batch_size: int = 1,
    ):
        assert isinstance(predictor, Predictor) or (hasattr(predictor, "predict") and hasattr(predictor, "update")), \
            "predictor must implement predict(n:int) and update(seq[int])"
        assert isinstance(coder, Coder) or (hasattr(coder, "encode_symbols") and hasattr(coder, "decode_symbols")), \
            "coder must implement encode_symbols and decode_symbols"
        self.predictor = predictor
        self.coder = coder
        self.quantizer = quantizer if quantizer is not None else UniformQuantizer(step=1.0)
        self.batch_size = int(batch_size)

    def decode(self, bitstream: bytes, n_values: Optional[int] = None) -> List[float]:
        """
        Decode bitstream into a list of reconstructed values.
        - bitstream: bytes as produced by coder.encode_symbols
        - n_values: number of values to decode (required for many coders). If None and coder supports
          implicit length, decoder may attempt to derive it (but best to pass it).
        """
        if n_values is None:
            raise ValueError("n_values must be provided for decoding (number of symbols/values encoded).")

        total = int(n_values)
        recon = []
        i = 0
        # To keep API simple, ask coder to decode in chunks of batch_size
        while i < total:
            b = min(self.batch_size, total - i)
            # pred BEFORE decoding these symbols:
            preds = np.asarray(self.predictor.predict(b), dtype=float)
            assert preds.shape[0] == b, "predictor.predict(b) must return b predictions"

            syms = self.coder.decode_symbols(bitstream, b)  # NOTE: coder must consume next b symbols from stream
            # If coder decodes entire stream at once, adapt accordingly (this protocol assumes ordered decode).
            if len(syms) != b:
                # Some coders return entire decoded list at once; handle that:
                # If coder returned more, take first b and keep rest as "remaining"
                # (left as user implementation detail).
                raise RuntimeError(f"Coder.decode_symbols returned {len(syms)} symbols, expected {b}")

            # reconstruct and update predictor AFTER decoding
            for idx, sym in enumerate(syms):
                residual_hat = self.quantizer.symbol_to_residual(sym)
                value_hat = float(preds[idx]) + float(residual_hat)
                recon.append(value_hat)

            # Now update predictor with quantized residuals (as ints)
            self.predictor.update(syms)

            i += b

        return recon

class SimpleFrequencyTable(FrequencyTable):
    """
    Simple frequency table implementing the ANS FrequencyTable protocol.
    Assumes that symbols are integers which start from 0, and correspond
    to indices of each frequency in the initialization frequency list.
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
        symbol = int(symbol)  # ✅ ensure symbol is an integer
        if not (0 <= symbol < len(self._freqs)):
            raise ValueError(f"Symbol {symbol} out of range (max {len(self._freqs)-1})")
        return self._freqs[symbol]

    def cum_freq(self, symbol: int) -> int:
        symbol = int(symbol)  # ✅ ensure symbol is an integer
        if not (0 <= symbol < len(self._freqs)):
            raise ValueError(f"Symbol {symbol} out of range (max {len(self._freqs)-1})")
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


class ANSEnoder(Encoder):
    """Simple rANS encoder"""

    def __init__(self, freq_table: FrequencyTable):
        self.ft = freq_table

    def encode(self, data: Sequence[int]) -> bytes:
        state = 1
        ft = self.ft

        # Encode in reverse
        for symbol in reversed(data):
            freq = ft.freq(symbol)
            cum = ft.cum_freq(symbol)
            state = (state // freq) * ft.total + cum + (state % freq)

        # Just turn final state into bytes (big endian)
        return state.to_bytes((state.bit_length() + 7) // 8, 'big')


class ANSDeoder(Decoder):
    """Simple rANS encoder"""

    def __init__(self, freq_table: FrequencyTable):
        self.ft = freq_table

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

        return decoded