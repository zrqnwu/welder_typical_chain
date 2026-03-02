#!/usr/bin/env python3
"""Thin wrapper for typical-chain verification.

Default verifier comes from vendored backend under this repo.
"""

import argparse
import os
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Path to dumped output binary")
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--pattern", default="linear")
    parser.add_argument("--seed", default="1")
    parser.add_argument("--rtol", default="1e-3")
    parser.add_argument("--atol", default="1e-3")
    parser.add_argument(
        "--backend-root",
        default=os.environ.get(
            "BACKEND_ROOT",
            "/home/zhangruiqi/welder_typical_chain/backend/welder",
        ),
    )
    parser.add_argument(
        "--legacy-root",
        default=None,
        help="Deprecated alias of --backend-root",
    )
    args = parser.parse_args()

    backend_root = args.backend_root
    if args.legacy_root:
        backend_root = args.legacy_root

    verifier = os.path.join(backend_root, "bench", "verify_matmul_softmax.py")
    if not os.path.isfile(verifier):
        print(f"error: verifier not found: {verifier}", file=sys.stderr)
        return 2

    cmd = [
        sys.executable,
        verifier,
        "--out",
        args.out,
        "--k",
        str(args.k),
        "--pattern",
        args.pattern,
        "--seed",
        str(args.seed),
        "--rtol",
        str(args.rtol),
        "--atol",
        str(args.atol),
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
