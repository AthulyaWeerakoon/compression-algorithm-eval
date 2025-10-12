import numpy as np
from compression_evaluation.classes import PCEncoder, PCDecoder, ANSCoder, SimpleFrequencyTable, \
    StaticResidualRegressor
from compression_evaluation.types import RegressorEnvelop, Quantizer


# Demo quantizer that returns positive symbols for preset array of values
class DemoQuantizer(Quantizer):
    def symbol_to_residual(self, symbol: int) -> float:
        return symbol - 1

    def residual_to_symbol(self, residual: float) -> int:
        return int(residual) + 1


# Demo predicto that always predict 1
class DemoRegressor(RegressorEnvelop):
    def predict(self, n: int, input_window: np.ndarray) -> np.ndarray:
        return np.ones(n)


if __name__ == "__main__":
    frequencies = [3, 4, 6, 1, 7, 4]
    stream = [0, 5, 2, 2, 1, 4, 2, 5, 2, 1, 4, 4, 5, 1, 4, 4, 5, 0, 4, 3, 4, 3, 0, 1, 2, 2]
    print("Original stream:", stream)

    # ANS encoder and decoder
    freq_table = SimpleFrequencyTable(frequencies)
    ans_coder = ANSCoder(freq_table)

    # LPC encoder and decoder
    regressor = DemoRegressor()
    predictor = StaticResidualRegressor(regressor, 5)
    quantizer = DemoQuantizer()
    lpc_encoder = PCEncoder(predictor, ans_coder, quantizer, 1)
    lpc_decoder = PCDecoder(predictor, ans_coder, quantizer, 1)

    encoded_stream = lpc_encoder.encode(stream)
    print("Encoded stream (bytes):", encoded_stream)

    reconstructed_stream = lpc_decoder.decode(encoded_stream)
    print("Reconstructed stream:", reconstructed_stream)

    


