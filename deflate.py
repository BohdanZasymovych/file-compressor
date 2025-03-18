"""deflate.py"""
import os
import struct
import argparse
from dataclasses import dataclass
import heapq
from typing import Optional, Any


BLOCK_SIZES = {
    ".txt": 32768,
    ".html": 32768,
    ".htm": 32768,
    ".xml": 32768,
    ".json": 32768,
    ".csv": 32768,
    ".log": 32768,
    ".css": 32768,
    ".js": 32768,

    ".bmp": 16384,
    ".tif": 16384,
    ".tiff": 16384,
    ".raw": 16384,

    ".exe": 16384,
    ".dll": 16384,
    ".bin": 16384,
    ".dat": 16384,

    ".jpg": 4096,
    ".jpeg": 4096,
    ".png": 4096,
    ".gif": 4096,
    ".mp3": 4096,
    ".mp4": 4096,
    ".avi": 4096,
    ".zip": 4096,
    ".gz": 4096,
    ".rar": 4096,
}


# (code, min_value, max_value, extra_bits)
LENGTH_CODES = (
    (257, 3, 3, 0),
    (258, 4, 4, 0),
    (259, 5, 5, 0),
    (260, 6, 6, 0),
    (261, 7, 7, 0),
    (262, 8, 8, 0),
    (263, 9, 9, 0),
    (264, 10, 10, 0),
    (265, 11, 12, 1),
    (266, 13, 14, 1),
    (267, 15, 16, 1),
    (268, 17, 18, 1),
    (269, 19, 22, 2),
    (270, 23, 26, 2),
    (271, 27, 30, 2),
    (272, 31, 34, 2),
    (273, 35, 42, 3),
    (274, 43, 50, 3),
    (275, 51, 58, 3),
    (276, 59, 66, 3),
    (277, 67, 82, 4),
    (278, 83, 98, 4),
    (279, 99, 114, 4),
    (280, 115, 130, 4),
    (281, 131, 162, 5),
    (282, 163, 194, 5),
    (283, 195, 226, 5),
    (284, 227, 257, 5),
    (285, 258, 258, 0)
)


# (code, min_value, max_value, extra_bits)
DISTANCE_CODES = (
    (0, 1, 1, 0),
    (1, 2, 2, 0),
    (2, 3, 3, 0),
    (3, 4, 4, 0),
    (4, 5, 6, 1),
    (5, 7, 8, 1),
    (6, 9, 12, 2),
    (7, 13, 16, 2),
    (8, 17, 24, 3),
    (9, 25, 32, 3),
    (10, 33, 48, 4),
    (11, 49, 64, 4),
    (12, 65, 96, 5),
    (13, 97, 128, 5),
    (14, 129, 192, 6),
    (15, 193, 256, 6),
    (16, 257, 384, 7),
    (17, 385, 512, 7),
    (18, 513, 768, 8),
    (19, 769, 1024, 8),
    (20, 1025, 1536, 9),
    (21, 1537, 2048, 9),
    (22, 2049, 3072, 10),
    (23, 3073, 4096, 10),
    (24, 4097, 6144, 11),
    (25, 6145, 8192, 11),
    (26, 8193, 12288, 12),
    (27, 12289, 16384, 12),
    (28, 16385, 24576, 13),
    (29, 24577, 32768, 13)
)


# CODE_LENGTH_ORDER = (16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15)


# Static Huffman tree for the code length alphabet (symbols 0 to 18)
CODE_LENGTH_CODES = {
    0:  (9, 4),    # binary: 1001
    1:  (27, 5),   # binary: 11011
    2:  (28, 5),   # binary: 11100
    3:  (29, 5),   # binary: 11101
    4:  (30, 5),   # binary: 11110
    5:  (31, 5),   # binary: 11111
    6:  (0, 4),    # binary: 0000
    7:  (1, 4),    # binary: 0001
    8:  (2, 4),    # binary: 0010
    9:  (3, 4),    # binary: 0011
    10: (4, 4),    # binary: 0100
    11: (5, 4),    # binary: 0101
    12: (6, 4),    # binary: 0110
    13: (7, 4),    # binary: 0111
    14: (8, 4),    # binary: 1000
    15: (26, 5),   # binary: 11010
    16: (10, 4),   # binary: 1010
    17: (11, 4),   # binary: 1011
    18: (12, 4)    # binary: 1100
}


def get_length_code(length):
    """
    Given a match length (in the range 3 to 258), this function returns a tuple:
    (length_code, extra_bits_count, extra_bits_value)
    """
    if length < 3 or length > 258:
        raise ValueError

    for (code, min_len, max_len, extra_bits) in LENGTH_CODES:
        if min_len <= length <= max_len:
            extra_value = length - min_len
            return {'code': code, 'extra_bits': extra_bits, 'extra_value': extra_value}
    raise ValueError


def get_distance_code(distance):
    """
    Given a match distance (in the range 1 to 32768), this function returns a tuple:
    (distance_code, extra_bits_count, extra_bits_value)
    """
    if distance < 1 or distance > 32768:
        raise ValueError

    for (code, min_dist, max_dist, extra_bits) in DISTANCE_CODES:
        if min_dist <= distance <= max_dist:
            extra_value = distance - min_dist
            return {'code': code, 'extra_bits': extra_bits, 'extra_value': extra_value}

    raise ValueError


@dataclass
class HuffmanNode:
    """class representing node in the huffman tree"""
    freq: float
    symbol: Any = None
    left: Optional['HuffmanNode'] = None
    right: Optional['HuffmanNode'] = None

    def __lt__(self, other: 'HuffmanNode') -> bool:
        return self.freq <= other.freq


class HuffMan:
    """class to build huffman tree and get codes of literals, lengths and distances"""
    def __init__(self):
        pass

    def calculate_frequencies(self, compressed_file: list) -> dict:
        """calculate frequencies of literals, lengths and distances appearing in the lz77 output"""
        literals_length_freq = {}
        distances_freq = {}
        distances_number = 0
        for i in compressed_file:
            if isinstance(i, int):
                literals_length_freq[i] = literals_length_freq.get(i, 0)+1
            else:
                dist, length = i
                length_code = get_length_code(length)['code']
                dist_code = get_distance_code(dist)['code']
                literals_length_freq[length_code] = literals_length_freq.get(length_code, 0)+1
                distances_freq[dist_code] = distances_freq.get(dist_code, 0)+1
                distances_number += 1

        literals_length_freq = {symbol: num/(len(compressed_file)+1) for symbol, num in literals_length_freq.items()}
        literals_length_freq.update({256: 1/(len(compressed_file)+1)})
        return (
                literals_length_freq,
                {dist: num/distances_number for dist, num in distances_freq.items()}
                )

    @staticmethod
    def frequencies_to_nodes(frequencies: dict) -> list:
        """conver frequencies to  nodes of huffman tree, returns min heap with nodes"""
        heap = []
        for el, freq in frequencies.items():
            heapq.heappush(heap, HuffmanNode(freq=freq, symbol=el))
        return heap

    def build_tree(self, heap: list):
        """build huffman tree using heap of min heap of nodes"""
        # print(heap)
        # print('\n\n\n\n')
        if heap:
            while len(heap) >= 2:
                right_node = heapq.heappop(heap)
                left_node = heapq.heappop(heap)
                heapq.heappush(heap, HuffmanNode(freq=left_node.freq+right_node.freq,
                                                left=left_node,
                                                right=right_node))
            return heap[0]
        return None

    def tree_to_code_legth(self, root: 'HuffmanNode', cur_lengh: int=1, code_lengths: int=None) -> dict:
        """get code lengths of each symbol from huffman tree"""
        if root is None:
            return {}
        if code_lengths is None:
            code_lengths = {}
        if root.left is None and root.right is None:
            code_lengths[root.symbol] = cur_lengh
        else:
            self.tree_to_code_legth(root.left, cur_lengh+1, code_lengths)
            self.tree_to_code_legth(root.right, cur_lengh+1, code_lengths)
        return code_lengths

    @staticmethod
    def code_lengths_to_codes(codes_length: dict) -> dict:
        """conver code length to binary codes of symbols"""
        sorted_dict_items = sorted(codes_length.items(), key=lambda i: (i[1], i[0]))
        prev_code = -1
        prev_code_length = 1
        codes = {}
        for el, cur_code_length in sorted_dict_items:
            cur_code = (prev_code+1) << (cur_code_length-prev_code_length)
            codes[el] = (cur_code, cur_code_length)
            prev_code_length = cur_code_length
            prev_code = cur_code
        return codes


class LZ77:
    def __init__(self, lookahead_buffer_size=258):
        self.lookahead_buffer_size = lookahead_buffer_size

    def find_longest_match(self, block: bytes, position: int) -> tuple[int, int]:
        """
        Function finds longest match of string from start of lookahead window in buffer.
        Returns tuple containing offset and length.
        """
        best_offset, best_length = 0, 0
        cur_window_size = position
        cur_lookahead_buffer_size = min(len(block)-position, self.lookahead_buffer_size)

        for i in range(position):
            if block[i] == block[position]:
                cur_length = 1
                window_pos = i + cur_length%(position-i)
                while (cur_length < cur_lookahead_buffer_size
                       and block[window_pos] == block[position+cur_length]):
                    cur_length += 1
                    window_pos = i + cur_length%(position-i)
                    if cur_length > best_length:
                        best_length = cur_length
                        best_offset = cur_window_size-i

        return (best_offset, best_length)

    def compress(self, filepath: str) -> list[list]:
        """
        Function returns list of tuples, representing copies of previouse symbols, and literals
        """
        #read file by blocks
        with open(filepath, 'rb') as file:
            extention = os.path.splitext(filepath)[1]
            block_size = BLOCK_SIZES.get(extention, 32768)
            block = file.read(block_size)
            blocks = []
            while block:
                blocks.append(block)
                block = file.read(block_size)

        compressed_file = []
        for block in blocks:
            compressed_block = []
            i = 0
            while i < len(block):
                offset, length = self.find_longest_match(block, i)
                if length > 2:
                    compressed_block.append((offset, length))
                    i += length
                else:
                    compressed_block.append(ord(block[i:i+1]))
                    i += 1
            compressed_file.append(compressed_block)
        return compressed_file


class BitWriter:
    """class to make bit stream and write it to the file"""
    def __init__(self, literals_lengths_codes: dict, distances_codes: dict, is_last_block: bool):
        self.bit_stream = 0
        self.bit_count = 0
        self.is_last_block = is_last_block
        self.literals_lengths_codes = literals_lengths_codes
        self.distances_codes = distances_codes

    def write_bit_stream_to_file(self, output_file_path: str) -> None:
        """write bit stream to the output file"""
        with open(output_file_path, 'wb') as file:
            while self.bit_count >= 8:
                byte = self.bit_stream >> (self.bit_count - 8) & 0xFF
                self.bit_stream &= (1<<self.bit_count-8)-1
                self.bit_count -= 8
                file.write(struct.pack('B', byte))

            if self.bit_count:
                file.write(struct.pack('B', self.bit_stream << (8-self.bit_count)))

    def write_bits_to_stream(self, value: int, bit_length: int) -> None:
        """write bits to bit stream"""
        self.bit_stream = self.bit_stream << (bit_length) | value
        self.bit_count += bit_length

    def write_bit_stream(self, compressed_data: list, output_file_path: str) -> None:
        """write compressed data to the output file as bitestream"""
        self.write_bits_to_stream((1 if self.is_last_block else 0), 1)
        lit_len_lengths = self.extract_lengths(self.literals_lengths_codes)
        dist_lengths = self.extract_lengths(self.distances_codes)
        rle, hlit, hdist = self.rle_encode(lit_len_lengths, dist_lengths)
        # print(hlit, hdist, rle)
        self.write_trees_info(rle, hlit, hdist)
        for symbol in compressed_data:
            if isinstance(symbol, int):
                code, bit_length = self.literals_lengths_codes[symbol]
                self.write_bits_to_stream(code, bit_length)
            else:
                distance, length = symbol

                length_code_dict = get_length_code(length)
                length_code, length_code_bit_length = self.literals_lengths_codes[length_code_dict['code']]
                length_extra_value, length_extra_bits = length_code_dict['extra_value'], length_code_dict['extra_bits']
                length_code = (length_code << length_extra_bits) | length_extra_value
                length_code_bit_length += length_extra_bits

                distance_code_dict = get_distance_code(distance)
                distance_code, distance_code_bit_length = self.distances_codes[distance_code_dict['code']]
                distance_extra_value, distance_extra_bits = distance_code_dict['extra_value'], distance_code_dict['extra_bits']
                distance_code = (distance_code << distance_extra_bits) | distance_extra_value
                distance_code_bit_length += distance_extra_bits

                self.write_bits_to_stream(length_code, length_code_bit_length)
                self.write_bits_to_stream(distance_code, distance_code_bit_length)

        eob_code, eob_code_length = self.literals_lengths_codes[256]
        self.write_bits_to_stream(eob_code, eob_code_length)
        self.write_bit_stream_to_file(output_file_path)

    @staticmethod
    def extract_lengths(code_dict: dict) -> list:
        """Extract code lengths in order of their keys."""
        max_key = max(code_dict.keys()) if code_dict else 0
        lengths = [0] * (max_key + 1)
        for symbol, (_, code_length) in code_dict.items():
            lengths[symbol] = code_length
        return lengths

    def rle_encode(self, lit_lengths: list, dist_lengths: list) -> tuple[int, int]:
        """
        Encodes the header information (the description of literal/length and distance
        Huffman trees) for a DEFLATE dynamic block using a static Huffman tree for the
        code length alphabet.
        """
        nlit = 257
        for i in range(len(lit_lengths)):
            if lit_lengths[i] != 0:
                nlit = i + 1
        # nlit = max(nlit, 257)

        ndist = 1
        for i in range(len(dist_lengths)):
            if dist_lengths[i] != 0:
                ndist = i + 1
        # ndist = max(ndist, 1)

        hlit = nlit - 257
        hdist = ndist - 1

        combined = lit_lengths[:nlit] + dist_lengths[:ndist]

        rle = []
        i = 0
        while i < len(combined):
            curr = combined[i]
            j = i + 1
            while j < len(combined) and combined[j] == curr:
                j += 1
            run_length = j - i
            if curr == 0:
                while run_length >= 11:
                    count = min(run_length, 138)
                    rle.append((18, count - 11, 7))
                    run_length -= count
                if run_length >= 3:
                    rle.append((17, run_length - 3, 3))
                else:
                    for _ in range(run_length):
                        rle.append((0, 0, 0))
            else:
                rle.append((curr, 0, 0))
                run_length -= 1
                while run_length >= 3:
                    count = min(run_length, 6)
                    rle.append((16, count - 3, 2))
                    run_length -= count
                while run_length > 0:
                    rle.append((curr, 0, 0))
                    run_length -= 1
            i = j

        return rle, hlit, hdist

    def write_trees_info(self, rle: list, hlit: int, hdist: int):
        """write info about literals/length tree to the bit stream"""
        self.write_bits_to_stream(2, 2)
        self.write_bits_to_stream(hlit, 5)
        self.write_bits_to_stream(hdist, 5)

        for sym, extra, extra_bits in rle:
            code, code_len = CODE_LENGTH_CODES[sym]
            self.write_bits_to_stream(code, code_len)
            if extra_bits > 0:
                self.write_bits_to_stream(extra, extra_bits)


def read_file_bits(filepath):
    with open(filepath, 'rb') as f:
        data = f.read()
        bits = ''.join(format(byte, '08b') for byte in data)
        print(bits)


def deflate_compress(input_file_path: str, output_file_path: str) -> None:
    hf = HuffMan()
    lz = LZ77()
    compressed_data = lz.compress(input_file_path)
    for i, block in enumerate(compressed_data):
        literals_lengths_frequencies, distances_frequencies = hf.calculate_frequencies(block)
        literals_lengths_tree = hf.build_tree(hf.frequencies_to_nodes(literals_lengths_frequencies))
        distances_tree = hf.build_tree(hf.frequencies_to_nodes(distances_frequencies))
        literals_lengths_codes = hf.code_lengths_to_codes(hf.tree_to_code_legth(literals_lengths_tree))
        distances_codes = hf.code_lengths_to_codes(hf.tree_to_code_legth(distances_tree))

        bit_writer = BitWriter(literals_lengths_codes, distances_codes, is_last_block = (i == len(compressed_data)-1))
        bit_writer.write_bit_stream(block, output_file_path)


def deflate_decompress(input_file_path: str, output_file_path: str) -> None:
    pass


def main():
    parser = argparse.ArgumentParser('Compress and decompress files')
    subparsers = parser.add_subparsers(dest="command", help="Choose whether to compress or decompress a file")

    compress_parser = subparsers.add_parser("compress", help="Compress a file")
    compress_parser.add_argument("input_file", help="Path to the file to compress")
    compress_parser.add_argument("output_file", help="Path to the compressed file")
    compress_parser.add_argument('--verbose', '--v', action='store_true', help='Show info about initial and compressed file sizes')
    compress_parser.add_argument('--content', '--c', action='store_true', help='Show content of compressed file')

    decompress_parser = subparsers.add_parser("decompress", help="Decompress a file")
    decompress_parser.add_argument("input_file", help="Path to the file to decompress")
    decompress_parser.add_argument("output_file", help="Path to the decompressed file")

    args = parser.parse_args()

    if args.command == 'compress':
        INPUT_FILE = args.input_file
        OUTPUT_FILE = args.output_file
        deflate_compress(INPUT_FILE, OUTPUT_FILE)
        print(f'Compressed data was written to {OUTPUT_FILE}')
        if args.verbose:
            print(f'{round(os.path.getsize(INPUT_FILE)/1024, 2)}kb -> {round(os.path.getsize(OUTPUT_FILE)/1024, 2)}kb')
        if args.content:
            read_file_bits(OUTPUT_FILE)

    if args.command == 'decompress':
        raise NotImplementedError


if __name__ == '__main__':
    main()
