#!/usr/bin/env python3
#
# Source-level guard for unary op boundary semantics across ggml backends.
#
# This intentionally recognizes only the small set of source patterns used by
# the checked CPU, CUDA, and Vulkan implementations. If a backend changes one of
# these ops, update the pattern here after verifying the new boundary behavior.

import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SourceOp:
    backend: str
    op: str
    path: Path
    expr: str
    fn: Callable[[float], float]


@dataclass(frozen=True)
class Case:
    label: str
    value: float


CASES = [
    Case("+0", 0.0),
    Case("-0", -0.0),
    Case("+tiny", math.ldexp(1.0, -149)),
    Case("-tiny", -math.ldexp(1.0, -149)),
    Case("+inf", math.inf),
    Case("-inf", -math.inf),
    Case("nan", math.nan),
]

OPS = {
    "step": {
        "cases": CASES,
        "cpu": ROOT / "ggml/src/ggml-cpu/unary-ops.cpp",
        "cuda": ROOT / "ggml/src/ggml-cuda/unary.cu",
        "vulkan": ROOT / "ggml/src/ggml-vulkan/vulkan-shaders/step.comp",
    },
    "sgn": {
        "cases": CASES,
        "cpu": ROOT / "ggml/src/ggml-cpu/unary-ops.cpp",
        "cuda": ROOT / "ggml/src/ggml-cuda/unary.cu",
        "vulkan": ROOT / "ggml/src/ggml-vulkan/vulkan-shaders/sgn.comp",
    },
    "relu": {
        # NaN behavior for backend max/fmax intrinsics is not the zero-boundary
        # regression this check targets, so keep this op focused on the sign cut.
        "cases": [case for case in CASES if case.label != "nan"],
        "cpu": ROOT / "ggml/src/ggml-cpu/unary-ops.cpp",
        "cuda": ROOT / "ggml/src/ggml-cuda/unary.cu",
        "vulkan": ROOT / "ggml/src/ggml-vulkan/vulkan-shaders/relu.comp",
    },
}


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"failed to read {path}: {exc}") from exc


def compact(expr: str) -> str:
    expr = re.sub(r"//.*", "", expr)
    expr = re.sub(r"/\*.*?\*/", "", expr, flags=re.S)
    return re.sub(r"\s+", "", expr)


def extract_c_return(path: Path, op: str) -> str:
    source = read_text(path)
    match = re.search(
        rf"static\s+(?:inline\s+)?(?:__device__\s+__forceinline__\s+)?float\s+op_{op}\s*"
        r"\(\s*float\s+x\s*\)\s*\{\s*return\s+(?P<expr>.*?);\s*\}",
        source,
        flags=re.S,
    )
    if not match:
        raise RuntimeError(f"could not find op_{op} return expression in {path}")
    return match.group("expr").strip()


def extract_vulkan_assignment(path: Path) -> str:
    source = read_text(path)
    match = re.search(r"data_d\[[^\]]+\]\s*=\s*D_TYPE\((?P<expr>.*?)\)\s*;", source, flags=re.S)
    if not match:
        raise RuntimeError(f"could not find data_d assignment in {path}")
    return match.group("expr").strip()


def c_fmax(x: float, y: float) -> float:
    if math.isnan(x):
        return y
    if math.isnan(y):
        return x
    return x if x > y else y


def eval_step_gt_zero(x: float) -> float:
    return 1.0 if x > 0.0 else 0.0


def eval_step_ge_zero(x: float) -> float:
    return 1.0 if x >= 0.0 else 0.0


def eval_sgn_strict(x: float) -> float:
    return 1.0 if x > 0.0 else (-1.0 if x < 0.0 else 0.0)


def eval_relu_strict(x: float) -> float:
    return x if x > 0.0 else 0.0


def eval_relu_fmax_zero(x: float) -> float:
    return c_fmax(x, 0.0)


def fail_unrecognized(backend: str, op: str, path: Path, expr: str) -> None:
    rel = path.relative_to(ROOT)
    raise RuntimeError(f"unrecognized {backend} {op} expression in {rel}: {expr!r}")


def recognize(backend: str, op: str, path: Path, expr: str) -> SourceOp:
    normalized = compact(expr)

    if op == "step":
        if normalized in {
            "(x>0.f)?1.f:0.f",
            "x>0.f?1.f:0.f",
            "(x>0.0f)?1.0f:0.0f",
            "x>0.0f?1.0f:0.0f",
            "x>0.0f",
        }:
            return SourceOp(backend, op, path, expr, eval_step_gt_zero)
        if normalized in {
            "(x>=0.f)?1.f:0.f",
            "x>=0.f?1.f:0.f",
            "(x>=0.0f)?1.0f:0.0f",
            "x>=0.0f?1.0f:0.0f",
            "x>=0.0f",
        }:
            return SourceOp(backend, op, path, expr, eval_step_ge_zero)

    if op == "sgn":
        if normalized in {
            "(x>0.f)?1.f:((x<0.f)?-1.f:0.f)",
            "(x>0.f?1.f:((x<0.f?-1.f:0.f)))",
            "sign(float(data_a[i]))",
        }:
            return SourceOp(backend, op, path, expr, eval_sgn_strict)

    if op == "relu":
        if normalized in {
            "(x>0.f)?x:0.f",
            "(x>0.0f)?x:0.0f",
        }:
            return SourceOp(backend, op, path, expr, eval_relu_strict)
        if normalized in {
            "fmaxf(x,0)",
            "fmaxf(x,0.0f)",
            "max(float(data_a[i]),0)",
            "max(float(data_a[i]),0.0f)",
        }:
            return SourceOp(backend, op, path, expr, eval_relu_fmax_zero)

    fail_unrecognized(backend, op, path, expr)


def same_float(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or a == b


def load_op(backend: str, op: str, path: Path) -> SourceOp:
    expr = extract_vulkan_assignment(path) if backend == "vulkan" else extract_c_return(path, op)
    return recognize(backend, op, path, expr)


def check() -> list[str]:
    errors = []

    for op, config in OPS.items():
        cpu = load_op("cpu", op, config["cpu"])
        backends = [
            load_op("cuda", op, config["cuda"]),
            load_op("vulkan", op, config["vulkan"]),
        ]

        for backend in backends:
            for case in config["cases"]:
                expected = cpu.fn(case.value)
                actual = backend.fn(case.value)
                if same_float(expected, actual):
                    continue

                errors.append(
                    f"{op} {backend.backend} mismatch for {case.label}: "
                    f"cpu={expected!r} backend={actual!r}\n"
                    f"  cpu {cpu.path.relative_to(ROOT)}: {cpu.expr}\n"
                    f"  {backend.backend} {backend.path.relative_to(ROOT)}: {backend.expr}"
                )

    return errors


def main() -> int:
    try:
        errors = check()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if errors:
        print("Unary boundary source check failed:", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    checked = ", ".join(OPS)
    print(f"Unary boundary source check passed for: {checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
