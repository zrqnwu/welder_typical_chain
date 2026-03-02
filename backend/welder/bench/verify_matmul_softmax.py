#!/usr/bin/env python3
import argparse
import array
import json
import math
import struct
from pathlib import Path


def _read_dump(path: Path):
    meta_path = Path(str(path) + ".json")
    if not path.exists():
        raise FileNotFoundError(path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    meta = json.loads(meta_path.read_text())
    if int(meta.get("elem_bytes", 0)) != 4:
        raise ValueError(f"expected elem_bytes=4, got {meta.get('elem_bytes')}")
    rank = int(meta.get("rank", 0))
    if rank != 2:
        raise ValueError(f"expected rank=2 output dump, got rank={rank}")

    offset = int(meta["offset"])
    sizes = [int(x) for x in meta["sizes"]]
    strides = [int(x) for x in meta["strides"]]
    nbytes = int(meta["bytes"])
    if len(sizes) != 2 or len(strides) != 2:
        raise ValueError("expected sizes/strides to have length 2")
    if nbytes % 4 != 0:
        raise ValueError(f"dump byte size not divisible by 4: {nbytes}")

    data = path.read_bytes()
    if len(data) < nbytes:
        raise ValueError(
            f"dump file is smaller than metadata bytes: {len(data)} < {nbytes}"
        )
    flat = array.array("f")
    flat.frombytes(data[:nbytes])
    return flat, offset, sizes, strides


def _linear_val(flat: int) -> float:
    t = flat % 1024
    return t * 0.001 - 0.5


def _xorshift32(state: int) -> int:
    state &= 0xFFFFFFFF
    state ^= (state << 13) & 0xFFFFFFFF
    state ^= (state >> 17) & 0xFFFFFFFF
    state ^= (state << 5) & 0xFFFFFFFF
    return state & 0xFFFFFFFF


def _random_vals(count: int, seed: int):
    s = 1 if seed == 0 else int(seed)
    out = []
    for _ in range(count):
        s = _xorshift32(s)
        u = float(s) * (1.0 / 4294967296.0)
        out.append(u * 2.0 - 1.0)
    return out


def _f32_to_f16_bits(x: float) -> int:
    # Convert f32 -> IEEE-754 binary16 bits (round-to-nearest-even).
    u = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    sign = (u >> 31) & 0x1
    exp = (u >> 23) & 0xFF
    mant = u & 0x7FFFFF

    if exp == 0xFF:
        # NaN/Inf.
        if mant == 0:
            return (sign << 15) | 0x7C00
        payload = (mant >> 13) & 0x03FF
        return (sign << 15) | 0x7C00 | payload | 1

    exp16 = int(exp) - 127 + 15
    if exp16 >= 31:
        return (sign << 15) | 0x7C00  # overflow -> Inf

    if exp16 <= 0:
        if exp16 < -10:
            return (sign << 15)  # underflow -> signed zero
        mant |= 0x800000  # restore implicit 1
        shift = 1 - exp16
        mant16 = mant >> (shift + 13)
        remainder = mant & ((1 << (shift + 13)) - 1)
        halfway = 1 << (shift + 12)
        if remainder > halfway or (remainder == halfway and (mant16 & 1)):
            mant16 += 1
        return (sign << 15) | (mant16 & 0x03FF)

    mant16 = mant >> 13
    remainder = mant & 0x1FFF
    halfway = 0x1000
    if remainder > halfway or (remainder == halfway and (mant16 & 1)):
        mant16 += 1
        if mant16 == 0x400:
            mant16 = 0
            exp16 += 1
            if exp16 >= 31:
                return (sign << 15) | 0x7C00

    return (sign << 15) | ((exp16 & 0x1F) << 10) | (mant16 & 0x03FF)


def _f16_bits_to_f32(bits: int) -> float:
    sign = (bits >> 15) & 0x1
    exp = (bits >> 10) & 0x1F
    mant = bits & 0x03FF

    if exp == 0:
        if mant == 0:
            return -0.0 if sign else 0.0
        val = (mant / 1024.0) * (2.0 ** (-14))
        return -val if sign else val

    if exp == 31:
        if mant == 0:
            return float("-inf") if sign else float("inf")
        return float("nan")

    val = (1.0 + mant / 1024.0) * (2.0 ** (exp - 15))
    return -val if sign else val


def _to_f16_f32(x: float) -> float:
    return _f16_bits_to_f32(_f32_to_f16_bits(x))


def main():
    ap = argparse.ArgumentParser(
        description="Verify Matmul->Softmax output dump against a CPU reference (sampled rows)."
    )
    ap.add_argument("--out", required=True, help="Path to output dump (.bin).")
    ap.add_argument(
        "--m",
        type=int,
        default=0,
        help="Runtime M extent to verify (0 => use dump metadata M).",
    )
    ap.add_argument(
        "--n",
        type=int,
        default=0,
        help="Runtime N extent to verify (0 => use dump metadata N).",
    )
    ap.add_argument("--k", type=int, default=64, help="K dimension (default 64).")
    ap.add_argument(
        "--pattern",
        default="linear",
        choices=["zero", "linear", "random"],
        help="Input init pattern (must match welder-profiler --init).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed for --pattern=random (must match welder-profiler --seed).",
    )
    ap.add_argument(
        "--rows",
        default="0,1,2,3,7,15,31,63,127,255,511,1023",
        help="Comma-separated row indices to verify (default: a fixed set).",
    )
    ap.add_argument("--rtol", type=float, default=1e-2)
    ap.add_argument("--atol", type=float, default=1e-2)
    ap.add_argument(
        "--matmul-f16",
        action="store_true",
        help="Simulate f16 truncation for Matmul operands (A/B) before softmax.",
    )
    args = ap.parse_args()

    flat, offset, sizes, strides = _read_dump(Path(args.out))
    dump_m, dump_n = sizes
    s0, s1 = strides
    m = dump_m if args.m <= 0 else int(args.m)
    n = dump_n if args.n <= 0 else int(args.n)
    if m <= 0 or n <= 0:
        raise SystemExit(f"error: invalid runtime extents m={m} n={n}")
    if m > dump_m or n > dump_n:
        raise SystemExit(
            f"error: runtime extents exceed dump extents: "
            f"runtime=({m},{n}) dump=({dump_m},{dump_n})"
        )
    k = int(args.k)

    def out_at(i: int, j: int) -> float:
        idx = offset + i * s0 + j * s1
        return float(flat[idx])

    rows = []
    for tok in args.rows.split(","):
        tok = tok.strip()
        if not tok:
            continue
        rows.append(int(tok))
    rows = [r for r in rows if 0 <= r < m]
    if not rows:
        raise SystemExit("error: no valid --rows selected")

    # Build B[K,N].
    if args.pattern == "zero":
        b_vals = [[0.0 for _ in range(n)] for _ in range(k)]
    elif args.pattern == "linear":
        b_vals = [[_linear_val(kk * n + j) for j in range(n)] for kk in range(k)]
    else:
        rand = _random_vals(k * n, args.seed)
        b_vals = [rand[kk * n : (kk + 1) * n] for kk in range(k)]

    if args.matmul_f16:
        b_vals = [[_to_f16_f32(x) for x in row] for row in b_vals]

    # For random A, we only need to generate up to max_row*K elements.
    max_row = max(rows)
    a_prefix = None
    if args.pattern == "random":
        a_prefix = _random_vals((max_row + 1) * k, args.seed)

    max_abs = 0.0
    max_rel = 0.0
    ok = True

    for i in rows:
        # Build A row.
        if args.pattern == "zero":
            a_row = [0.0] * k
        elif args.pattern == "linear":
            base = i * k
            a_row = [_linear_val(base + kk) for kk in range(k)]
        else:
            assert a_prefix is not None
            a_row = a_prefix[i * k : (i + 1) * k]

        if args.matmul_f16:
            a_row = [_to_f16_f32(x) for x in a_row]

        # C row = A_row @ B.
        c_row = [0.0] * n
        for j in range(n):
            s = 0.0
            for kk in range(k):
                s += a_row[kk] * b_vals[kk][j]
            c_row[j] = s

        row_max = max(c_row)
        exps = [math.exp(x - row_max) for x in c_row]
        denom = sum(exps)
        inv = 1.0 / denom

        for j in range(n):
            out_v = out_at(i, j)
            if not math.isfinite(out_v):
                print("FAIL: output contains NaN/Inf")
                return 1
            ref_v = exps[j] * inv
            diff = abs(out_v - ref_v)
            max_abs = max(max_abs, diff)
            d = max(abs(ref_v), 1e-6)
            max_rel = max(max_rel, diff / d)
            if diff > (args.atol + args.rtol * abs(ref_v)):
                ok = False

    status = "PASS" if ok else "FAIL"
    print(
        f"{status}: max_abs={max_abs:.6e} max_rel={max_rel:.6e} shape=({m},{n}) rows={len(rows)}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
