#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TensorInfo:
    name: str
    dtype: str | None
    shape: list[int] | None


@dataclass
class Decision:
    name: str
    status: str
    base_name: str | None
    reason: str
    dtype: str | None
    shape: list[int] | None


def load_get_base_tensor_name() -> Callable[[str], str]:
    """Load the real converter helper without importing its heavy dependencies."""
    path = REPO_ROOT / "convert_lora_to_gguf.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "get_base_tensor_name":
            module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(module)
            namespace: dict[str, object] = {}
            exec(compile(module, str(path), "exec"), namespace)
            return namespace["get_base_tensor_name"]  # type: ignore[return-value]
    raise RuntimeError(f"get_base_tensor_name not found in {path}")


def resolve_safetensors_path(path: Path) -> Path:
    if path.is_dir():
        path = path / "adapter_model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"safetensors file not found: {path}")
    return path


def load_safetensors_header(path: Path) -> list[TensorInfo]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("install safetensors to inspect LoRA adapter files") from exc

    tensors: list[TensorInfo] = []
    with safe_open(path, framework="np") as handle:
        for name in handle.keys():
            tensor_slice = handle.get_slice(name)
            tensors.append(
                TensorInfo(
                    name=name,
                    dtype=tensor_slice.get_dtype(),
                    shape=tensor_slice.get_shape(),
                )
            )
    return tensors


def classify_tensor(
    tensor: TensorInfo,
    get_base_tensor_name: Callable[[str], str],
) -> Decision:
    name = tensor.name
    base_name = get_base_tensor_name(name)
    is_lora_a = ".lora_A.weight" in name or ".lora_embedding_A" in name
    is_lora_b = ".lora_B.weight" in name or ".lora_embedding_B" in name

    if is_lora_a:
        return Decision(name, "keep", base_name, "LoRA A tensor", tensor.dtype, tensor.shape)
    if is_lora_b:
        return Decision(name, "keep", base_name, "LoRA B tensor", tensor.dtype, tensor.shape)

    if ".base_layer.weight" in name:
        return Decision(name, "filter", None, "converter skips base layer weights", tensor.dtype, tensor.shape)
    if "_layernorm" in name or ".norm" in name:
        return Decision(name, "keep", base_name, "converter keeps adapter norm tensor", tensor.dtype, tensor.shape)

    reason = "unexpected name: not a LoRA A/B tensor"
    if ".embed_tokens.weight" in name or ".lm_head.weight" in name:
        reason += "; adapter appears to include embeddings or lm_head weights"
    return Decision(name, "reject", base_name, reason, tensor.dtype, tensor.shape)


def format_shape(shape: list[int] | None) -> str:
    return "?" if shape is None else "x".join(str(dim) for dim in shape)


def print_report(decisions: list[Decision]) -> int:
    print("status\tname\tbase_name\tdtype\tshape\treason")
    for decision in decisions:
        base_name = decision.base_name if decision.base_name is not None else "-"
        dtype = decision.dtype if decision.dtype is not None else "?"
        print(
            f"{decision.status}\t{decision.name}\t{base_name}\t"
            f"{dtype}\t{format_shape(decision.shape)}\t{decision.reason}"
        )

    parts: dict[str, set[str]] = defaultdict(set)
    for decision in decisions:
        if decision.status != "keep" or decision.base_name is None:
            continue
        if ".lora_A.weight" in decision.name or ".lora_embedding_A" in decision.name:
            parts[decision.base_name].add("A")
        if ".lora_B.weight" in decision.name or ".lora_embedding_B" in decision.name:
            parts[decision.base_name].add("B")

    missing = {base_name: {"A", "B"} - seen for base_name, seen in parts.items() if seen != {"A", "B"}}
    if missing:
        print()
        print("incomplete_lora_pairs")
        for base_name, missing_parts in sorted(missing.items()):
            print(f"{base_name}\tmissing {','.join(sorted(missing_parts))}")
        return 1

    return 1 if any(decision.status == "reject" for decision in decisions) else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run LoRA adapter safetensors names through convert_lora_to_gguf.py filtering."
    )
    parser.add_argument(
        "adapter",
        type=Path,
        help="adapter_model.safetensors file, or a directory containing adapter_model.safetensors",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        path = resolve_safetensors_path(args.adapter)
        get_base_tensor_name = load_get_base_tensor_name()
        tensors = load_safetensors_header(path)
    except (OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    decisions = [classify_tensor(tensor, get_base_tensor_name) for tensor in tensors]
    return print_report(decisions)


if __name__ == "__main__":
    raise SystemExit(main())
