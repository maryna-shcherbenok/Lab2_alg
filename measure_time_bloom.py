from statistics import mean
from time import perf_counter

from bloom_filter import BloomFilter


def make_word(number: int, prefix: str) -> str:
    """Create a deterministic unique word containing only Latin letters."""
    chars = []
    value = number

    while True:
        chars.append(chr(ord("a") + value % 26))
        value //= 26
        if value == 0:
            break

    return (prefix + "".join(reversed(chars)))[:15]


def one_run(n: int, query_count: int = 100_000):
    # Test-data generation is not included in the measured algorithm time.
    inserted = [make_word(i, "a") for i in range(n)]
    successful = [inserted[i % n] for i in range(min(query_count, n))]
    missing = [make_word(i, "z") for i in range(query_count)]

    bloom = BloomFilter()

    # Insertions.
    start = perf_counter()
    for word in inserted:
        bloom.add(word)
    add_time = perf_counter() - start

    # Collision/overlap diagnostics are calculated AFTER the insertion timer,
    # so they do not affect the measured insertion time.
    set_bits = sum(byte.bit_count() for byte in bloom.bits)
    hash_placements = n * bloom.hash_count

    # Each of the k*N hash positions attempts to set one bit.
    # If that bit was already set, the placement overlaps with a previous one.
    # This is a Bloom-filter position collision/overlap, not a hash-table
    # collision that needs chaining or another collision-resolution method.
    overlap_count = hash_placements - set_bits
    fill_rate = set_bits / bloom.bit_count
    overlap_rate = overlap_count / hash_placements if hash_placements else 0.0

    # Queries for words that were inserted.
    start = perf_counter()
    for word in successful:
        bloom.contains(word)
    successful_time = perf_counter() - start

    # Queries for words that were never inserted.
    start = perf_counter()
    false_positives = 0
    for word in missing:
        if bloom.contains(word):
            false_positives += 1
    missing_time = perf_counter() - start

    fp_rate = false_positives / len(missing)

    return (
        add_time,
        successful_time,
        missing_time,
        fp_rate,
        set_bits,
        hash_placements,
        overlap_count,
        fill_rate,
        overlap_rate,
    )


def main():
    sizes = (10_000, 100_000, 500_000, 1_000_000)
    repeats = 3

    print(
        "N\tadd_s\tquery_present_s\tquery_absent_s\t"
        "false_positive_rate\tset_bits\tfill_rate\t"
        "overlaps\toverlap_rate"
    )

    for n in sizes:
        samples = [one_run(n) for _ in range(repeats)]

        add_time = mean(sample[0] for sample in samples)
        successful_time = mean(sample[1] for sample in samples)
        missing_time = mean(sample[2] for sample in samples)
        fp_rate = mean(sample[3] for sample in samples)

        # These structural values are identical in all three runs because
        # the generated data and hash functions are deterministic.
        set_bits = samples[0][4]
        hash_placements = samples[0][5]
        overlap_count = samples[0][6]
        fill_rate = samples[0][7]
        overlap_rate = samples[0][8]

        print(
            f"{n}\t{add_time:.6f}\t{successful_time:.6f}\t"
            f"{missing_time:.6f}\t{fp_rate:.4%}\t"
            f"{set_bits}\t{fill_rate:.4%}\t"
            f"{overlap_count}\t{overlap_rate:.4%}"
        )

        print(
            f"  hash_placements={hash_placements:,}; "
            f"unique_set_bits={set_bits:,}; "
            f"overlapping_placements={overlap_count:,}"
        )


if __name__ == "__main__":
    main()
