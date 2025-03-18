import zlib
import argparse
import os

def compress_file(input_file: str, output_file: str):
    """Compresses a file using zlib and writes the compressed data to a new file."""
    try:
        with open(input_file, 'rb') as f:
            data = f.read()

        compressed_data = zlib.compress(data, level=zlib.Z_BEST_COMPRESSION)

        with open(output_file, 'wb') as f:
            f.write(compressed_data)

        original_size = os.path.getsize(input_file) / 1024  # Convert to KB
        compressed_size = os.path.getsize(output_file) / 1024  # Convert to KB

        print(f"Original file size: {original_size:.2f} KB")
        print(f"Compressed file size: {compressed_size:.2f} KB")
    except FileNotFoundError:
        print("Error: Input file not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress a file using zlib.")
    parser.add_argument("input_file", help="Path to the input file.")
    parser.add_argument("output_file", help="Path to save the compressed file.")
    args = parser.parse_args()

    compress_file(args.input_file, args.output_file)
