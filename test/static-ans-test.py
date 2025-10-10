from compression_evaluation.classes import SimpleFrequencyTable, ANSCoder

if __name__ == "__main__":
    frequencies = [3, 4, 6, 1, 7, 4]
    stream = [0, 5, 2, 2, 1, 4, 2, 5, 2, 1, 4, 4, 5, 1, 4, 4, 5, 0, 4, 3, 4, 3, 0, 1, 2, 2]
    print("Original stream:", stream)

    freq_table = SimpleFrequencyTable(frequencies)
    coder = ANSCoder(freq_table)

    encoded_stream = coder.encode_symbols(stream)
    print("Encoded stream (bytes):", encoded_stream)

    reconstructed_stream = coder.decode_symbols(encoded_stream)
    print("Reconstructed stream:", reconstructed_stream)

    # --- Compression ratio ---
    original_size_bytes = len(stream) * 4
    compressed_size_bytes = len(encoded_stream)

    compression_ratio = original_size_bytes / compressed_size_bytes if compressed_size_bytes > 0 else 0
    print(f"Original size: {original_size_bytes} bytes")
    print(f"Compressed size: {compressed_size_bytes} bytes")
    print(f"Compression ratio: {compression_ratio:.2f}x smaller")

