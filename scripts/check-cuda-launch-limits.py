#!/usr/bin/env python3
"""Check CUDA launch dimensions and int32-sensitive index arithmetic.

This is a standalone debugging helper for the CUDA overflow bug family where a
tensor shape produces a launch grid that is legal in x but too large in y/z, or
where kernel index math would overflow int32 before being widened.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Callable, Iterable


CUDA_GRID_X_MAX = 2**31 - 1
CUDA_GRID_YZ_MAX = 65535
INT32_MAX = 2**31 - 1

CUDA_CPY_TILE_DIM_2D = 32
CUDA_CPY_BLOCK_NM = 8
CUDA_GET_ROWS_BACK_BLOCK_SIZE = 256
CUDA_QUANTIZE_BLOCK_SIZE = 256
QK8_1 = 32
FATTN_KQ_STRIDE = 256


@dataclass(frozen=True)
class CheckResult:
    name: str
    values: dict[str, int]
    failures: list[str]
    notes: list[str]

    @property
    def ok(self) -> bool:
        return not self.failures


def ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("ceil_div divisor must be positive")
    return (a + b - 1) // b


def require_non_negative(**values: int) -> None:
    for name, value in values.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")


def require_positive(**values: int) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")


def validate_grid(grid_x: int, grid_y: int, grid_z: int) -> list[str]:
    failures: list[str] = []
    if grid_x > CUDA_GRID_X_MAX:
        failures.append(f"grid.x={grid_x} exceeds CUDA limit {CUDA_GRID_X_MAX}")
    if grid_y > CUDA_GRID_YZ_MAX:
        failures.append(f"grid.y={grid_y} exceeds CUDA limit {CUDA_GRID_YZ_MAX}")
    if grid_z > CUDA_GRID_YZ_MAX:
        failures.append(f"grid.z={grid_z} exceeds CUDA limit {CUDA_GRID_YZ_MAX}")
    return failures


def validate_int32(name: str, value: int) -> list[str]:
    if value > INT32_MAX:
        return [f"{name}={value} exceeds int32 max {INT32_MAX}"]
    return []


def result(name: str, values: dict[str, int], failures: list[str], notes: Iterable[str] = ()) -> CheckResult:
    return CheckResult(name=name, values=values, failures=failures, notes=list(notes))


def check_raw(args: argparse.Namespace) -> CheckResult:
    require_non_negative(grid_x=args.grid_x, grid_y=args.grid_y, grid_z=args.grid_z)
    failures = validate_grid(args.grid_x, args.grid_y, args.grid_z)
    values = {"grid.x": args.grid_x, "grid.y": args.grid_y, "grid.z": args.grid_z}
    if args.int32_expr:
        for item in args.int32_expr:
            name, sep, value_text = item.partition("=")
            if sep != "=" or not name:
                raise ValueError("--int32-expr must use NAME=VALUE")
            value = int(value_text, 0)
            values[name] = value
            failures.extend(validate_int32(name, value))
    return result("raw", values, failures)


def check_cpy_transpose(args: argparse.Namespace) -> CheckResult:
    require_positive(ne00=args.ne00, ne01=args.ne01, ne02=args.ne02)
    require_non_negative(nb00=args.nb00, nb02=args.nb02)

    if args.nb00 <= args.nb02:
        ne00n = args.ne00
        ne01n = args.ne01
        ne02n = args.ne02
    else:
        ne00n = args.ne00
        ne01n = args.ne01 * args.ne02
        ne02n = 1

    ne = args.ne if args.ne is not None else ne00n * ne01n * ne02n
    require_positive(ne=ne)

    rows_per_batch = ne01n * ne00n
    grid_x = ceil_div(ne01n, CUDA_CPY_TILE_DIM_2D)
    grid_y = ceil_div(ne00n, CUDA_CPY_TILE_DIM_2D)
    grid_z_raw = ceil_div(ne // rows_per_batch, CUDA_CPY_BLOCK_NM)

    failures = validate_grid(grid_x, grid_y, grid_z_raw)
    values = {
        "grid.x": grid_x,
        "grid.y": grid_y,
        "grid.z.raw": grid_z_raw,
        "ne00n": ne00n,
        "ne01n": ne01n,
        "ne02n": ne02n,
        "ne": ne,
    }
    notes = []
    if grid_z_raw > CUDA_GRID_YZ_MAX:
        notes.append("cpy.cu fix chunks the z launch range and grid-strides over remaining slices")
    return result("cpy-transpose", values, failures, notes)


def check_get_rows_back(args: argparse.Namespace) -> CheckResult:
    require_positive(ne00=args.ne00, ne1=args.ne1)
    grid_x = ceil_div(args.ne00, CUDA_GET_ROWS_BACK_BLOCK_SIZE)
    grid_y_raw = args.ne1
    grid_z = 1
    failures = validate_grid(grid_x, grid_y_raw, grid_z)
    values = {"grid.x": grid_x, "grid.y.raw": grid_y_raw, "grid.z": grid_z}
    notes = []
    if grid_y_raw > CUDA_GRID_YZ_MAX:
        notes.append("getrows.cu fix clamps grid.y and grid-strides dst_row by gridDim.y")
    return result("get-rows-back", values, failures, notes)


def check_quantize_q8_1(args: argparse.Namespace) -> CheckResult:
    require_positive(ne0=args.ne0, ne1=args.ne1, ne2=args.ne2, ne3=args.ne3)
    if args.ne0 % QK8_1 != 0:
        raise ValueError(f"ne0 must be divisible by QK8_1 ({QK8_1})")

    grid_x = ceil_div(args.ne0, CUDA_QUANTIZE_BLOCK_SIZE)
    grid_y_raw = args.ne1
    grid_z_raw = args.ne2 * args.ne3

    failures = validate_grid(grid_x, grid_y_raw, grid_z_raw)
    failures.extend(validate_int32("quantize logical block span", grid_x * grid_y_raw * CUDA_QUANTIZE_BLOCK_SIZE // QK8_1))
    values = {
        "grid.x": grid_x,
        "grid.y.raw": grid_y_raw,
        "grid.z.raw": grid_z_raw,
        "logical_block_span": grid_x * grid_y_raw * CUDA_QUANTIZE_BLOCK_SIZE // QK8_1,
    }
    notes = []
    if grid_y_raw > CUDA_GRID_YZ_MAX or grid_z_raw > CUDA_GRID_YZ_MAX:
        notes.append("quantize.cu fix clamps y/z and grid-strides over i1/i23")
    return result("quantize-q8-1", values, failures, notes)


def check_fattn_mask(args: argparse.Namespace) -> CheckResult:
    iter_k = args.iter_k
    if iter_k is None:
        if args.k_ne1 is None:
            raise ValueError("fattn-mask requires either --iter-k or --k-ne1")
        require_positive(k_ne1=args.k_ne1)
        if args.k_ne1 % FATTN_KQ_STRIDE != 0:
            raise ValueError(f"k_ne1 must be divisible by FATTN_KQ_STRIDE ({FATTN_KQ_STRIDE})")
        iter_k = args.k_ne1 // FATTN_KQ_STRIDE

    require_positive(iter_k=iter_k, grid_x=args.grid_x, grid_y=args.grid_y, ncols1=args.ncols1)
    require_non_negative(s31=args.s31, s33=args.s33)
    grid_z = 1

    kv_max_sj = (iter_k - 1) * FATTN_KQ_STRIDE
    worst_mask_offset = 0
    if args.grid_y > 0 and args.grid_x > 0:
        worst_mask_offset = (args.grid_y - 1) * args.s33 + (args.grid_x - 1) * args.ncols1 * args.s31 + kv_max_sj // 2

    failures = validate_grid(args.grid_x, args.grid_y, grid_z)
    failures.extend(validate_int32("KV_max_sj", kv_max_sj))
    failures.extend(validate_int32("worst_mask_half2_offset", worst_mask_offset))
    values = {
        "grid.x": args.grid_x,
        "grid.y": args.grid_y,
        "grid.z": grid_z,
        "iter_k": iter_k,
        "KV_max_sj": kv_max_sj,
        "worst_mask_half2_offset": worst_mask_offset,
    }
    notes = []
    if kv_max_sj > INT32_MAX or worst_mask_offset > INT32_MAX:
        notes.append("fattn-common.cuh KQ mask scan needs int64-safe stride/index math for these values")
    return result("fattn-mask", values, failures, notes)


def print_result(check: CheckResult) -> None:
    print(f"{check.name}: {'PASS' if check.ok else 'FAIL'}")
    for key, value in check.values.items():
        print(f"  {key}: {value}")
    for failure in check.failures:
        print(f"  violation: {failure}")
    for note in check.notes:
        print(f"  note: {note}")


def run_self_test() -> int:
    tests: list[tuple[str, argparse.Namespace, Callable[[argparse.Namespace], CheckResult], str]] = [
        (
            "cpy grid.z overflow",
            argparse.Namespace(ne00=32, ne01=32, ne02=65536 * CUDA_CPY_BLOCK_NM + 1, ne=None, nb00=1, nb02=32),
            check_cpy_transpose,
            "grid.z.raw",
        ),
        (
            "get_rows_back grid.y overflow",
            argparse.Namespace(ne00=1, ne1=65536),
            check_get_rows_back,
            "grid.y.raw",
        ),
        (
            "fattn KQ mask int32 overflow",
            argparse.Namespace(iter_k=8388609, k_ne1=None, grid_x=1, grid_y=1, ncols1=1, s31=1, s33=1),
            check_fattn_mask,
            "KV_max_sj",
        ),
        (
            "quantize grid.y overflow",
            argparse.Namespace(ne0=32, ne1=65536, ne2=1, ne3=1),
            check_quantize_q8_1,
            "grid.y.raw",
        ),
        (
            "quantize grid.z overflow",
            argparse.Namespace(ne0=32, ne1=1, ne2=65536, ne3=1),
            check_quantize_q8_1,
            "grid.z.raw",
        ),
    ]

    failed = 0
    for label, ns, fn, expected_text in tests:
        check = fn(ns)
        print(f"self-test: {label}")
        print_result(check)
        if check.ok or not any(expected_text in failure or expected_text in key for failure in check.failures for key in check.values):
            print(f"  self-test failure: expected {expected_text} to be reported")
            failed += 1
        print()
    if failed:
        print(f"self-test summary: {failed} failed")
        return 1
    print("self-test summary: all expected overflow configurations were caught")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    raw = subparsers.add_parser("raw", help="validate explicit CUDA grid dimensions")
    raw.add_argument("--grid-x", type=int, required=True)
    raw.add_argument("--grid-y", type=int, required=True)
    raw.add_argument("--grid-z", type=int, required=True)
    raw.add_argument("--int32-expr", action="append", help="additional int32-sensitive NAME=VALUE expression to validate")
    raw.set_defaults(func=check_raw)

    cpy = subparsers.add_parser("cpy-transpose", help="derive the old cpy.cu transposed-copy launch grid")
    cpy.add_argument("--ne00", type=int, required=True)
    cpy.add_argument("--ne01", type=int, required=True)
    cpy.add_argument("--ne02", type=int, required=True)
    cpy.add_argument("--ne", type=int, help="total copied element count; defaults to ne00n*ne01n*ne02n")
    cpy.add_argument("--nb00", type=int, default=1, help="source stride selector matching cpy.cu's nb00 <= nb02 branch")
    cpy.add_argument("--nb02", type=int, default=1, help="source stride selector matching cpy.cu's nb00 <= nb02 branch")
    cpy.set_defaults(func=check_cpy_transpose)

    get_rows = subparsers.add_parser("get-rows-back", help="derive the old getrows.cu get_rows_back launch grid")
    get_rows.add_argument("--ne00", type=int, required=True)
    get_rows.add_argument("--ne1", type=int, required=True)
    get_rows.set_defaults(func=check_get_rows_back)

    quantize = subparsers.add_parser("quantize-q8-1", help="derive the old quantize.cu q8_1 launch grid")
    quantize.add_argument("--ne0", type=int, required=True)
    quantize.add_argument("--ne1", type=int, required=True)
    quantize.add_argument("--ne2", type=int, required=True)
    quantize.add_argument("--ne3", type=int, required=True)
    quantize.set_defaults(func=check_quantize_q8_1)

    fattn = subparsers.add_parser("fattn-mask", help="validate flash-attn KQ-mask grid and stride/index arithmetic")
    fattn.add_argument("--iter-k", type=int, help="K->ne[1] / FATTN_KQ_STRIDE")
    fattn.add_argument("--k-ne1", type=int, help="K->ne[1], used to derive iter_k when --iter-k is omitted")
    fattn.add_argument("--grid-x", type=int, required=True, help="ntiles_x")
    fattn.add_argument("--grid-y", type=int, required=True, help="Q->ne[3] sequences")
    fattn.add_argument("--ncols1", type=int, default=1, help="template ncols1 used in the mask pointer stride")
    fattn.add_argument("--s31", type=int, default=1, help="mask->nb[1] / sizeof(half2)")
    fattn.add_argument("--s33", type=int, default=1, help="mask->nb[3] / sizeof(half2)")
    fattn.set_defaults(func=check_fattn_mask)

    self_test = subparsers.add_parser("self-test", help="run repro cases for the fixed bug family")
    self_test.set_defaults(func=None)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.mode == "self-test":
            return run_self_test()
        check = args.func(args)
    except ValueError as exc:
        parser.error(str(exc))
    print_result(check)
    return 0 if check.ok else 1


if __name__ == "__main__":
    sys.exit(main())
