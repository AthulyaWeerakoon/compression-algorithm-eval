from typing import Protocol, runtime_checkable, Any, Sequence, List
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
      - update(quantized_values: Sequence[int]) -> None: update internal state using
        the quantized values that were actually used (ints).
    """

    def predict(self, n: int = 1) -> Sequence[float]:
        """Return n number of predictions."""
        ...

    def update(self, quantized_values: Sequence[float]) -> None:
        """Update predictor memory with the residuals."""
        ...


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

    def decode_symbols(self, bitstream: bytes) -> List[int]: ...


@runtime_checkable
class Quantizer(Protocol):
    def value_to_symbol(self, value: float) -> int:
        """Quantize value (float) -> integer symbol."""
        ...

    def symbol_to_value(self, symbol: int) -> float:
        """Integer symbol -> reconstructed value (float)."""
        ...

    def symbol_range(self) -> int:
        """Return number of available discrete symbols (levels)."""
        ...


@runtime_checkable
class FrequencyTable(Protocol):
    """Protocol defining frequency interface for ANS."""

    def freq(self, symbol: int) -> int:
        """Return the frequency of the given symbol."""
        ...

    def cum_freq(self, symbol: int) -> int:
        """Return cumulative frequency up to (but not including) this symbol."""
        ...

    def symbol_from_cum(self, cum_value: int) -> int:
        """Return the symbol corresponding to the given cumulative frequency."""
        ...

    @property
    def total(self) -> int:
        """Total frequency count of all symbols."""
        ...


@runtime_checkable
class RegressorEnvelop(Protocol):
    """
    Protocol defining a minimal interface for wrapped regressors.

    Any regressor implementing this protocol must provide a `predict` method that:
    - takes the number of predictions `n` and a numpy input window (1D array)
    - returns a numpy array of `n` predicted values
    """

    def predict(self, n: int, input_window: np.ndarray) -> np.ndarray:
        """
        Predict the next `n` values based on the given input window.

        Args:
            n (int): Number of predictions to make.
            input_window (np.ndarray): 1D array of previous samples.

        Returns:
            np.ndarray: Array of `n` predicted values.
        """
        ...
