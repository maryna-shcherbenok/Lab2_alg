from __future__ import annotations

from math import ceil, log
from pathlib import Path
import sys

MAX_LEN = 15
MAX_ELEMENTS = 1_000_000
MAX_OPERATIONS = 1_000_000
FALSE_POSITIVE_PROBABILITY = 0.01

# 64-bit arithmetic is used for deterministic non-cryptographic hashes.
MASK_64 = (1 << 64) - 1
FNV_OFFSET = 1_469_598_103_934_665_603
FNV_PRIME = 1_099_511_628_211
SECOND_SEED = 0x9E3779B97F4A7C15
SECOND_BASE = 131


def validate_word(word: str) -> None:
    """Check the task restrictions: 1..15 Latin letters."""
    if not 1 <= len(word) <= MAX_LEN:
        raise ValueError("Word length must be from 1 to 15 characters.")

    if any(not ("a" <= ch <= "z" or "A" <= ch <= "Z") for ch in word):
        raise ValueError("Only Latin letters A-Z and a-z are allowed.")


class BloomFilter:
    def __init__(
        self,
        expected_elements: int = MAX_ELEMENTS,
        false_positive_probability: float = FALSE_POSITIVE_PROBABILITY,
    ) -> None:
        if expected_elements <= 0:
            raise ValueError("Expected number of elements must be positive.")
        if not 0 < false_positive_probability < 1:
            raise ValueError("False-positive probability must be in (0, 1).")

        self.expected_elements = expected_elements
        self.false_positive_probability = false_positive_probability

        # m = -n * ln(p) / (ln 2)^2
        self.bit_count = ceil(
            -expected_elements * log(false_positive_probability) / (log(2) ** 2)
        )

        # k = (m / n) * ln 2
        self.hash_count = max(
            1,
            round((self.bit_count / expected_elements) * log(2)),
        )

        # Compact bit array: eight filter bits in one byte.
        self.bits = bytearray((self.bit_count + 7) // 8)

    def _base_hashes(self, word: str) -> tuple[int, int]:
        """Calculate two deterministic 64-bit non-cryptographic hashes."""
        h1 = FNV_OFFSET
        h2 = SECOND_SEED

        for ch in word:
            code = ord(ch)

            # FNV-1a style hash.
            h1 ^= code
            h1 = (h1 * FNV_PRIME) & MASK_64

            # Independent polynomial-style 64-bit hash.
            h2 = (h2 * SECOND_BASE + code) & MASK_64

        # An odd step prevents a zero step and improves the double-hash sequence.
        h2 |= 1
        return h1, h2

    def _positions(self, word: str):
        """Generate k Bloom-filter positions using double hashing."""
        h1, h2 = self._base_hashes(word)

        for i in range(self.hash_count):
            yield (h1 + i * h2) % self.bit_count

    def _set_bit(self, position: int) -> None:
        byte_index = position >> 3
        bit_index = position & 7
        self.bits[byte_index] |= 1 << bit_index

    def _get_bit(self, position: int) -> bool:
        byte_index = position >> 3
        bit_index = position & 7
        return bool(self.bits[byte_index] & (1 << bit_index))

    def add(self, word: str) -> None:
        """Insert a word by setting all k corresponding bits to 1."""
        validate_word(word)

        for position in self._positions(word):
            self._set_bit(position)

    def contains(self, word: str) -> bool:
        """
        False: the word is definitely absent.
        True: the word is probably present; a false positive is possible.
        """
        validate_word(word)

        for position in self._positions(word):
            if not self._get_bit(position):
                return False

        return True


def process_file(input_path: Path, output_stream=sys.stdout) -> None:
    bloom = BloomFilter()
    operation_count = 0
    terminated = False

    with input_path.open("r", encoding="utf-8-sig") as source:
        for line_number, raw_line in enumerate(source, 1):
            line = raw_line.strip()

            if not line:
                continue

            if line == "#":
                terminated = True
                break

            operation_count += 1
            if operation_count > MAX_OPERATIONS:
                raise ValueError("The number of operations exceeds 1,000,000.")

            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Line {line_number}: expected '+ word', '? word' or '#'."
                )

            operation, word = parts
            validate_word(word)

            if operation == "+":
                bloom.add(word)
            elif operation == "?":
                output_stream.write("Y\n" if bloom.contains(word) else "N\n")
            elif operation == "-":
                raise ValueError(
                    f"Line {line_number}: the classic Bloom filter does not support deletion."
                )
            else:
                raise ValueError(
                    f"Line {line_number}: unknown operation '{operation}'."
                )

    if not terminated:
        raise ValueError("Input data must end with '#'.")


def main() -> int:
    input_path = Path(__file__).with_name("input.txt")

    try:
        process_file(input_path)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
