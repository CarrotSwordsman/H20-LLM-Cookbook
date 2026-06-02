#!/usr/bin/env python3
"""
Validate the H20 fused-MoE config files in `configs/{bf16,fp8_w8a8}/`.

Checks performed (each is fatal — exit code 1 if any fail):

1. **Filename convention**:
   - bf16:     `E={E},N={N},device_name=NVIDIA_H20.json`
   - fp8_w8a8: `E={E},N={N},device_name=NVIDIA_H20,dtype=fp8_w8a8.json`
2. **JSON schema**: top-level dict of batch-string -> per-batch config dict.
3. **Per-batch config keys**: must be exactly the 6 fields:
   `BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_warps, num_stages`.
4. **Value domains**: every field in the search space the cookbook actually
   tunes over (catches typos and stale tuner output).
5. **Batch grid completeness**: every config must contain the canonical 18
   batch sizes used by the upstream `benchmark_moe.py` and our tuners.

Usage:
    python scripts/check_configs.py
    python scripts/check_configs.py configs/bf16   # subset
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Filename patterns (named groups so we can sanity-check (E, N) against contents).
BF16_NAME_RE = re.compile(
    r"^E=(?P<E>\d+),N=(?P<N>\d+),device_name=NVIDIA_H20\.json$"
)
FP8_NAME_RE = re.compile(
    r"^E=(?P<E>\d+),N=(?P<N>\d+),device_name=NVIDIA_H20,dtype=fp8_w8a8\.json$"
)

REQUIRED_FIELDS = {
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "num_warps",
    "num_stages",
}

# Domains taken from tune_moe_h20.py / tune_moe_h20_fp8.py SEARCH_SPACE.
ALLOWED = {
    "BLOCK_SIZE_M": {16, 32, 64, 128},
    "BLOCK_SIZE_N": {32, 64, 128, 256},
    "BLOCK_SIZE_K": {32, 64, 128, 256},  # bf16 default uses 32; tuner space starts at 64
    "GROUP_SIZE_M": {1, 16, 32, 64},
    "num_warps": {4, 8},
    "num_stages": {3, 4, 5},
}

# Batch grid the tuner sweeps; per-batch JSON should cover all of these.
EXPECTED_BATCHES = {
    "1", "2", "4", "8", "16", "24", "32", "48", "64", "96",
    "128", "256", "512", "1024", "1536", "2048", "3072", "4096",
}


def check_one(path: Path, errors: list[str]) -> None:
    name = path.name
    is_fp8 = "dtype=fp8_w8a8" in name
    name_re = FP8_NAME_RE if is_fp8 else BF16_NAME_RE
    m = name_re.match(name)
    if not m:
        errors.append(
            f"{path}: filename does not match required pattern "
            f"({'fp8' if is_fp8 else 'bf16'})"
        )
        return

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        errors.append(f"{path}: invalid JSON — {exc}")
        return

    if not isinstance(data, dict):
        errors.append(f"{path}: top-level must be a dict")
        return

    missing = EXPECTED_BATCHES - set(data.keys())
    extra = set(data.keys()) - EXPECTED_BATCHES
    if missing:
        errors.append(
            f"{path}: missing batch keys {sorted(missing, key=int)}"
        )
    if extra:
        errors.append(
            f"{path}: unexpected batch keys {sorted(extra)}"
        )

    for batch_key, cfg in data.items():
        # batch key must be int-stringable
        try:
            int(batch_key)
        except ValueError:
            errors.append(f"{path}: batch key {batch_key!r} is not an int")
            continue

        if not isinstance(cfg, dict):
            errors.append(f"{path}[{batch_key}]: value must be a dict")
            continue

        cfg_keys = set(cfg.keys())
        if cfg_keys != REQUIRED_FIELDS:
            errors.append(
                f"{path}[{batch_key}]: keys mismatch — "
                f"missing={sorted(REQUIRED_FIELDS - cfg_keys)}, "
                f"extra={sorted(cfg_keys - REQUIRED_FIELDS)}"
            )

        for field, allowed in ALLOWED.items():
            if field not in cfg:
                continue
            v = cfg[field]
            if not isinstance(v, int):
                errors.append(
                    f"{path}[{batch_key}].{field}: expected int, got {type(v).__name__}"
                )
            elif v not in allowed:
                errors.append(
                    f"{path}[{batch_key}].{field}: value {v} not in allowed set "
                    f"{sorted(allowed)}"
                )


def gather(targets: list[Path]) -> list[Path]:
    files: list[Path] = []
    for t in targets:
        if t.is_file() and t.suffix == ".json":
            files.append(t)
        elif t.is_dir():
            files.extend(sorted(t.rglob("*.json")))
    return files


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        targets = [Path(p) for p in argv[1:]]
    else:
        targets = [REPO_ROOT / "configs" / "bf16",
                   REPO_ROOT / "configs" / "fp8_w8a8"]

    files = gather(targets)
    if not files:
        print(f"No JSON config files found under: {[str(t) for t in targets]}",
              file=sys.stderr)
        return 1

    errors: list[str] = []
    for f in files:
        check_one(f, errors)

    print(f"Checked {len(files)} config file(s).")
    if errors:
        print(f"\n{len(errors)} problem(s) found:\n", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("All configs OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
