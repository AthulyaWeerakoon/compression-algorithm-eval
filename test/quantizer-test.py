import numpy as np
from compression_evaluation.classes import UniformQuantizer  # change 'your_module' to the actual module name


def main():
    print("Testing UniformQuantizer...\n")

    # basic quantization
    q = UniformQuantizer(step=1.0, levels=8, min_residual=-4.0)
    residual = 0.0
    symbol = q.value_to_symbol(residual)
    reconstructed = q.symbol_to_value(symbol)
    print(f"Residual {residual} -> Symbol {symbol} -> Recon {reconstructed}")
    assert np.isclose(reconstructed, residual, atol=q.step)

    # clip behavior enabled
    q = UniformQuantizer(step=1.0, levels=8, min_residual=-4.0, clip=True)
    sym_low = q.value_to_symbol(-10.0)
    sym_high = q.value_to_symbol(10.0)
    print(f"Clipped low symbol: {sym_low}, high symbol: {sym_high}")
    assert sym_low == q.min_sym
    assert sym_high == q.max_sym

    # clip behavior disabled (should raise)
    q = UniformQuantizer(step=1.0, levels=8, min_residual=-4.0, clip=False)
    try:
        q.value_to_symbol(10.0)
    except ValueError:
        print("Out-of-range check passed (no clipping)")
    else:
        raise AssertionError("Expected ValueError for out-of-range residual")

    # round-trip quantization
    q = UniformQuantizer(step=0.5, levels=16, min_residual=-4.0)
    residuals = np.linspace(-4.25, 3.75, 17)
    for r in residuals:
        sym = q.value_to_symbol(r)
        recon = q.symbol_to_value(sym)
        print(f"{r:6.2f} -> {sym:4d} -> {recon:7.3f}")
        assert np.isclose(recon, r, atol=q.step)

    # default min_residual
    q = UniformQuantizer(step=2.0, levels=4)
    print(f"Default min_residual = {q.min_residual}, max_residual = {q.max_residual}")
    assert q.min_residual == -4.0
    assert np.isclose(q.max_residual, -4.0 + (q.levels - 1) * q.step)

    # symbol range
    q = UniformQuantizer(step=1.0, levels=256)
    assert q.symbol_range() == 256

    # assertion checks
    try:
        UniformQuantizer(step=0)
    except AssertionError:
        print("Assertion caught for step=0")
    else:
        raise AssertionError("Expected AssertionError for step=0")

    try:
        UniformQuantizer(step=1.0, levels=1)
    except AssertionError:
        print("Assertion caught for levels=1")
    else:
        raise AssertionError("Expected AssertionError for levels=1")

    print("\nAll tests passed")


if __name__ == "__main__":
    main()
