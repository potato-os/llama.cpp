#!/usr/bin/env python3

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_get_base_tensor_name():
    tree = ast.parse((REPO_ROOT / "convert_lora_to_gguf.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "get_base_tensor_name":
            module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(module)
            namespace = {}
            exec(compile(module, "convert_lora_to_gguf.py", "exec"), namespace)
            return namespace["get_base_tensor_name"]
    raise AssertionError("get_base_tensor_name not found")


class TestLoraTensorNameMapping(unittest.TestCase):
    def test_gemma4_language_model_prefix(self):
        get_base_tensor_name = load_get_base_tensor_name()

        self.assertEqual(
            get_base_tensor_name(
                "base_model.model.model.language_model.layers.0.self_attn.k_proj.lora_A.weight"
            ),
            "model.layers.0.self_attn.k_proj.weight",
        )
        self.assertEqual(
            get_base_tensor_name(
                "base_model.model.model.language_model.layers.0.self_attn.k_proj.lora_B.weight"
            ),
            "model.layers.0.self_attn.k_proj.weight",
        )


if __name__ == "__main__":
    unittest.main()
