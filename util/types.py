from typing import Protocol, runtime_checkable, Any, Sequence, List, Optional
import numpy as np


@runtime_checkable
class Encoder(Protocol):
    """
    Protocol representing an Encoder interface for compression experiments.

    Any class implementing this protocol must provide an `encode` method that
    takes a data array and returns an encoded bitstream. This protocol allows
    the Evaluator class to perform compression metrics and evaluation without
    enforcing a specific implementation.
    """

    def encode(self, data: Any) -> Any:
        """
        Encodes a data array into a bitstream.

        Args:
            data (Any): The data array to encode.

        Returns:
            Any: Encoded bytestream of the data array.
        """
        ...


@runtime_checkable
class Decoder(Protocol):
    """
    Protocol representing a Decoder interface for compression experiments.

    Any class implementing this protocol must provide a `decode` method that
    takes a bitstream and reconstructs the original data array. This allows
    Evaluator to compute reconstruction metrics such as MSE and SNR.
    """

    def decode(self, bitstream: Any) -> Any:
        """
        Decodes a bitstream into a data array.

        Args:
            bitstream (Any): The bitstream to decode.

        Returns:
            Any: Decoded data array of the bitstream.
        """
        ...


@runtime_checkable
class Predictor(Protocol):
    """
    Predictor protocol for pluggable prediction models.

    Implementations must:
      - predict(n: int) -> Sequence[float]: return the next n predictions in order.
      - update(quantized_residuals: Sequence[int]) -> None: update internal state using
        the quantized residuals that were actually used (ints).
    """

    def predict(self, n: int = 1) -> Sequence[float]: ...

    def update(self, quantized_residuals: Sequence[int]) -> None: ...


@runtime_checkable
class Coder(Protocol):
    """
    Coder protocol (abstracts ANS/rANS/tANS) used to encode/decode integer symbols.

    Implementations must:
      - encode_symbols(symbols: Sequence[int]) -> bytes
      - decode_symbols(bitstream: bytes, n: int) -> List[int]
    The decode_symbols should return symbols in the same order as they were encoded.
    """

    def encode_symbols(self, symbols: Sequence[int]) -> bytes: ...

    def decode_symbols(self, bitstream: bytes, n: int) -> List[int]: ...


@runtime_checkable
class Quantizer(Protocol):
    def residual_to_symbol(self, residual: float) -> int:
        """Quantize residual (float) -> integer symbol."""
        ...

    def symbol_to_residual(self, symbol: int) -> float:
        """Dequantize symbol -> reconstructed residual (float)."""
        ...

    def symbol_range(self) -> int:
        """Return number of available discrete symbols (levels)."""
        ...
