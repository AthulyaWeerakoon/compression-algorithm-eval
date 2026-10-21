from typing import Any
from decimal import Decimal, localcontext
import numpy as np
from .types import Encoder, Decoder
from .methods import profile_memory


class Evaluator:
    """
    Evaluator class used to evaluate required metrics for the study.
    Expects an encoder and a decoder implementing the required protocols.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder):
        # Runtime duck typing check
        assert hasattr(encoder, "encode") and callable(encoder.encode), \
            "Encoder must implement encode(data) -> bitstream"
        assert hasattr(decoder, "decode") and callable(decoder.decode), \
            "Decoder must implement decode(bitstream) -> data"

        self.encoder = encoder
        self.decoder = decoder

    def evaluate(self, data_series: Any) -> dict:
        """Evaluate a numeric series; compressed size measures payload only."""
        try:
            source = np.asarray(data_series)
            if source.dtype.kind not in "iuf" or source.ndim not in (1, 2) or any(d == 0 for d in source.shape):
                raise ValueError("data must be a nonempty real numeric 1D or 2D array")
            if not np.all(np.isfinite(source)):
                raise ValueError("data must be finite")
            if source.dtype.kind in "iu" and not np.all(source.astype(np.float64).astype(object) == source.astype(object)):
                raise ValueError("integer values must be exactly representable as float64")
            data_np = source.astype(np.float64)
            if not np.all(np.isfinite(data_np)):
                raise ValueError("data is not representable as finite float64")
        except (TypeError, OverflowError, ValueError) as exc:
            raise ValueError("data must be a nonempty real numeric 1D or 2D array") from exc
        count = int(data_np.size)
        uncompressed = int(data_series.nbytes) if isinstance(data_series, np.ndarray) else 8 * count

        bitstream, peak_encode_mem, t_encode = profile_memory(self.encoder.encode, data_series)
        if isinstance(bitstream, (bytes, bytearray)) and bitstream:
            compressed = len(bitstream)
        elif isinstance(bitstream, list) and bitstream and all(isinstance(b, (bytes, bytearray)) and b for b in bitstream):
            compressed = sum(map(len, bitstream))
        else:
            raise ValueError("encoder must return nonempty bytes or a list of nonempty byte streams")

        reconstructed, peak_decode_mem, t_decode = profile_memory(self.decoder.decode, bitstream)
        try:
            recon = np.asarray(reconstructed)
            if recon.dtype.kind not in "iuf" or recon.shape != data_np.shape or not np.all(np.isfinite(recon)):
                raise ValueError("decoder output must be finite numeric data with matching shape")
            recon = recon.astype(np.float64)
        except (TypeError, OverflowError, ValueError) as exc:
            raise ValueError("invalid decoder output") from exc
        fallback = False
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            difference = data_np - recon
            difference_squares = difference ** 2
            source_squares = data_np ** 2
            mse = float(np.mean(difference_squares))
            power = float(np.mean(source_squares))
            fallback = (
                not np.all(np.isfinite(difference))
                or not np.all(np.isfinite(difference_squares))
                or not np.all(np.isfinite(source_squares))
                or not np.isfinite(mse) or not np.isfinite(power)
                or np.any((difference != 0) & (difference_squares == 0))
                or np.any((data_np != 0) & (source_squares == 0))
                or (np.any(difference_squares > 0) and mse == 0)
                or (np.any(source_squares > 0) and power == 0)
            )
        if fallback:
            with localcontext() as ctx:
                ctx.prec = max(ctx.prec, 2800 + len(str(count)))
                dec_source = [Decimal.from_float(float(x)) for x in data_np.flat]
                dec_recon = [Decimal.from_float(float(x)) for x in recon.flat]
                dec_mse = sum(((x-y) ** 2 for x, y in zip(dec_source, dec_recon)), Decimal(0)) / count
                dec_power = sum((x ** 2 for x in dec_source), Decimal(0)) / count
                max_float = Decimal.from_float(float(np.finfo(np.float64).max))
                if dec_mse > max_float or dec_power > max_float:
                    raise ValueError("derived metric exceeds float64 range")
                mse, power = float(dec_mse), float(dec_power)
                if dec_mse > 0 and mse == 0:
                    raise ValueError("positive MSE underflows float64")
                if dec_mse == 0:
                    snr = float("inf")
                elif dec_power == 0:
                    snr = float("-inf")
                else:
                    snr = float(Decimal(10) * (dec_power.ln() - dec_mse.ln()) / Decimal(10).ln())
        else:
            snr = float("inf") if mse == 0 else (float("-inf") if power == 0 else float(10 * (np.log10(power) - np.log10(mse))))
        ratio, bps = uncompressed / compressed, 8 * compressed / count
        if not np.isfinite(mse) or not np.isfinite(ratio) or not np.isfinite(bps) or np.isnan(snr):
            raise ValueError("derived metric overflow")
        return {"encode_time_sec": t_encode, "encode_mem_mb": peak_encode_mem,
                "decode_time_sec": t_decode, "decode_mem_mb": peak_decode_mem,
                "compression_ratio": ratio, "bits_per_symbol": bps, "mse": mse, "snr_db": snr,
                "uncompressed_size_bytes": uncompressed, "compressed_size_bytes": compressed, "symbol_count": count}