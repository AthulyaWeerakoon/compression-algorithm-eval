from typing import Any
import numpy as np
from util.types_protocol import Encoder, Decoder
from util.methods import profile_memory


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
        results = {}

        # --- Encode ---
        bitstream, peak_encode_mem, t_encode = profile_memory(self.encoder.encode, data_series)
        results["encode_time_sec"] = t_encode
        results["encode_mem_mb"] = peak_encode_mem

        # Type sanity check
        assert isinstance(bitstream, (bytes, bytearray, list, np.ndarray)), \
            "Encoder must return a bitstream (bytes, bytearray, list, or ndarray)"

        # --- Decode ---
        reconstructed, peak_decode_mem, t_decode = profile_memory(self.decoder.decode, bitstream)
        results["decode_time_sec"] = t_decode
        results["decode_mem_mb"] = peak_decode_mem

        # --- Metrics ---
        data_np = np.array(data_series, dtype=float)
        recon_np = np.array(reconstructed, dtype=float)
        assert data_np.shape == recon_np.shape, "Decoded data shape mismatch"

        # Compression metrics
        uncompressed_size = len(data_np) * data_np.dtype.itemsize
        compressed_size = len(bitstream) if isinstance(bitstream, (bytes, bytearray)) else len(bitstream)
        results["compression_ratio"] = uncompressed_size / compressed_size if compressed_size > 0 else float("inf")
        results["bits_per_symbol"] = (compressed_size * 8) / len(data_np)

        # Distortion metrics
        mse = np.mean((data_np - recon_np) ** 2)
        results["mse"] = mse
        results["snr_db"] = 10 * np.log10(np.mean(data_np ** 2) / mse) if mse > 0 else float("inf")

        return results
