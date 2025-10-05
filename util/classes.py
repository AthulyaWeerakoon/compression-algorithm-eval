from util.types import Predictor, Coder, Quantizer, Encoder, Decoder
from typing import Sequence, List, Optional, Dict
import numpy as np
from util.methods import build_cdf


class UniformQuantizer(Quantizer):
    """
    Uniform scalar quantizer for values.

    Maps value -> integer symbol via rounding: q = round(value / step).
    Symbol mapping has an offset so output is non-negative (0..levels-1).
    Reconstruction: value_hat = (q - offset) * step.
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

    def value_to_symbol(self, value: float) -> int:
        q = int(np.round(value / self.step))
        sym = q + self.offset
        if self.clip:
            sym = max(self.min_sym, min(sym, self.max_sym))
        else:
            if not (self.min_sym <= sym <= self.max_sym):
                raise ValueError("Quantized symbol out of range.")
        return int(sym)

    def symbol_to_value(self, symbol: int) -> float:
        q = int(symbol) - self.offset
        return q * self.step

    def symbol_range(self) -> int:
        return self.levels


class UniformQuantizerByRange(Quantizer):
    """
    Range-based uniform quantizer.

    Computes step size automatically from [min_val, max_val] and number of levels:
        step = (max_val - min_val) / (levels - 1)

    - value_to_symbol: maps values in [min_val, max_val] to integers [0 .. levels-1]
    - symbol_to_value: reconstructs a float value from integer symbol
    """

    def __init__(self, min_val: float, max_val: float, levels: int = 65536, clip: bool = True):
        assert max_val > min_val, "max_val must be greater than min_val"
        assert levels >= 2, "levels must be >= 2"

        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.levels = int(levels)
        self.step = (self.max_val - self.min_val) / (self.levels - 1)
        self.clip = bool(clip)

    def value_to_symbol(self, value: float) -> int:
        """Quantize value in [min_val, max_val] to integer symbol."""
        q = int(np.round((value - self.min_val) / self.step))
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
      2. value = actual - pred
      3. q_sym = quantizer.value_to_symbol(value)  # quantize BEFORE updating predictor
      4. predictor.update([q_sym ...])                     # predictor receives quantized values
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
            values = actuals - preds  # float values

            # Quantize values to symbols
            syms = [self.quantizer.value_to_symbol(float(r)) for r in values]

            # Update predictor with quantized values BEFORE encoding (encoder-side update)
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
      1. preds = predictor.predict(n=batch_size)   # prediction BEFORE seeing quantized value
      2. q_syms = coder.decode_symbols(bitstream, n=batch_size)
      3. reconstruct = preds + quantizer.symbol_to_value(q_sym)
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
                value_hat = self.quantizer.symbol_to_value(sym)
                value_hat = float(preds[idx]) + float(value_hat)
                recon.append(value_hat)

            # Now update predictor with quantized values (as ints)
            self.predictor.update(syms)

            i += b

        return recon
    


class StaticANSEncoder(Encoder):
    """
    StaticANSEncoder encodes a sequence of integer symbols using static Asymmetric Numeral Systems (ANS).
    It uses a provided frequency table to build a cumulative distribution function (CDF) for efficient encoding.
    The encode method processes a list of symbols and returns the final ANS state representing the encoded data.
    """
    def __init__(self, freq_table: Dict[int, int]):
        self.freq_table = freq_table
        self.total ,self.cdf = build_cdf(freq_table)

    def encode(self, data_list: List[int]) -> bytes:
        """
        Encodes a list of integer symbols into a single ANS state integer using a static frequency table.

        Parameters:
            data_list (List[int]): List of integer symbols to encode.

        Returns:
            bytes: The final ANS bitstream.

        Encoding algorithm:
            Processes symbols in reverse order, updating the ANS state using the frequency table and cumulative distribution function (CDF).
        """
        state = 1
        for symbol in reversed(data_list):
            f = self.freq_table[symbol]
            c = self.cdf[symbol]
            state = (state // f) * self.total + c + (state % f)

        n_bytes = (state.bit_length() + 7) // 8   
        return state.to_bytes(n_bytes, byteorder="big")    



class StaticANSDecoder(Decoder):
    def __init__(self, freq_table: Dict[int,int]):
        self.freq_table = freq_table
        self.total = sum(freq_table.values())
        self.cdf_ranges = {}
        self.n, self.cdf = build_cdf(freq_table)
        cum = 0
        for symbol, frequency in sorted(freq_table.items()):
            self.cdf_ranges[symbol] = (cum, cum + frequency)
            cum += frequency

    def decode(self, bitstream: bytes, n_symbol: int) -> List[int]:
        """
        Decodes an ANS bitstream (as bytes) into the original list of symbols.

        Parameters:
            bitstream (bytes): The ANS bitstream to decode, as produced by StaticANSEncoder.encode.

        Returns:
            List[int]: The decoded list of integer symbols, in original order.

        Decoding algorithm:
            Iteratively extracts symbols from the ANS bitstream using the frequency table and CDF ranges,
            reconstructing the original sequence in reverse, then returns it in correct order.
        """
        state = int.from_bytes(bitstream, byteorder="big")  
        result = []
        for _ in range(n_symbol):
            x = state % self.total
            for sym, (lo, hi) in self.cdf_ranges.items():
                if lo <= x < hi:
                    result.append(sym)
                    f = self.freq_table[sym]
                    state = f * (state // self.total) + (x - lo)
                    break
        return result[::-1]
        
