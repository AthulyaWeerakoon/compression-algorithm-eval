import math
from compression_evaluation.methods import largest_remainder_quantize


def approx_equal(a, b, tol=0.3):
    return abs(math.log2(a / b)) < tol


def run_tests():
    print("Running tests for largest_remainder_quantize()...\n")

    # --- Basic proportionality test ---
    counts = [5, 2, 1]
    M = 256
    freqs = largest_remainder_quantize(counts, M)

    assert all(v > 0 for v in freqs), "All frequencies must be > 0"
    assert sum(freqs) == M, "Sum of frequencies must equal M"
    assert freqs[0] > freqs[1] > freqs[2], "Order (A>B>C) must be preserved"

    expected_ratios = [counts[0] / counts[1], counts[1] / counts[2]]
    got_ratios = [freqs[0] / freqs[1], freqs[1] / freqs[2]]
    assert all(approx_equal(er, gr) for er, gr in zip(expected_ratios, got_ratios)), \
        "Ratios should be approximately preserved"
    print("Basic proportionality test passed.")

    # --- Empty and invalid counts ---
    try:
        largest_remainder_quantize([], M)
        assert False, "Empty counts should raise ValueError"
    except ValueError:
        print("Empty input correctly raised ValueError.")

    try:
        largest_remainder_quantize([0, 0], M)
        assert False, "Zero total should raise ValueError"
    except ValueError:
        print("Zero-total input correctly raised ValueError.")

    # --- Too small M ---
    try:
        largest_remainder_quantize([5, 3, 2], 2)
        assert False, "M < len(counts) should raise AssertionError"
    except AssertionError:
        print("Too-small M correctly raised AssertionError.")

    # --- Scaling invariance ---
    base = [50, 30, 20]
    scaled = [x * 100 for x in base]
    f1 = largest_remainder_quantize(base, M)
    f2 = largest_remainder_quantize(scaled, M)

    for (a, b) in [(0, 1), (1, 2), (0, 2)]:
        r1, r2 = f1[a] / f1[b], f2[a] / f2[b]
        assert approx_equal(r1, r2, tol=0.2), f"Scaled ratio mismatch for {a}/{b}"
    print("Scaling invariance test passed.")

    # --- Uniform distribution ---
    uniform = [1] * 10
    f = largest_remainder_quantize(uniform, 100)
    assert max(f) - min(f) <= 1, "Uniform counts should yield nearly equal frequencies"
    assert sum(f) == 100, "Sum must match M"
    print("Uniform distribution test passed.")

    print("\nAll tests passed")


if __name__ == "__main__":
    run_tests()
