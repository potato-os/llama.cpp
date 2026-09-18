# Custom tensor types and the `general.tensor_types` convention

This fork adds quantization types beyond upstream's set. Their numeric
`ggml_type` ids are **local handles** that may change between forks and
versions; the permanent identity of a custom type is its **canonical name**.

## The metadata key

A GGUF that uses custom types carries a string-array metadata key:

```
general.tensor_types = ["54=Q1_sym_g32_sq4_sg8", "55=Q4_sym_g16_sq8_sg8"]
```

Each entry is `<id>=<name>`: `id` is the numeric type id as written in this
file's tensor headers (decimal), `name` is the canonical type name. Only
custom types are listed; standard types (`f32`, `f16`, `q8_0`, ...) are
governed by upstream and never appear here.

On load, the reader translates declared ids to its own internal enum by
name (see `gguf.cpp`). A declared id whose name this build does not know
fails only if a tensor actually uses it, with an error naming the type.
Files without the key load exactly as before. This keeps every shipped
file readable across renumbering, including a rebase onto an upstream
that has assigned those numeric ids to other types.

## Naming grammar

Hierarchical; each level states a size and how that level's scales are
stored. `fp16` is always the implied top-level scale format.

```
Q<b>       code bit-width (b bits per weight)
_sym       symmetric codes (levels centered on zero; no zero-point/min)
_g<N>      group size: N weights share one scale
_sq<m>     (only when group scales are quantized) scales stored in m bits
_sg<K>     (goes with _sq) K group-scales share one fp16 super-scale
```

No `_sq`/`_sg` fields means plain fp16 group scales with no sub-structure.
If a layout deviates from an implied default (for example a non-fp16 scale
format), its name must state that explicitly with an extra field. Encode
every axis that varies among shipped types; document implied constants
here.

## Registered names

Names are **append-only and frozen**: once a file has shipped with a name,
that name maps to exactly one byte layout forever. New layouts get new
names. MoE vs dense is not a type property (expert stacks are the same
bytes with a 3D shape) and is never encoded in a name.

| name | layout | block |
| --- | --- | --- |
| `Q1_sym_g32_sq4_sg8` | 1-bit symmetric codes; 32-weight groups; 8 x 4-bit unsigned sub-scales per fp16 super-scale | 256 weights / 38 bytes |
| `Q2_sym_g32_sq4_sg8` | 2-bit symmetric codes; 32-weight groups; 8 x 4-bit unsigned sub-scales per fp16 super-scale | 256 weights / 70 bytes |
| `Qt_sym_ms_g32_sq4_sg8` | ternary mask/sign codes; exactly 128 nonzeros per 256-weight block; 8 x 4-bit unsigned sub-scales per fp16 super-scale | 256 weights / 54 bytes |
| `Q4_sym_g16_f16` | 4-bit codes; 16-weight groups with one fp16 scale; value = (code - 7.5) × scale | 16 weights / 10 bytes |
| `Q4_sym_g16_sq8_sg8` | 4-bit symmetric codes; 16-weight groups; 8 x signed int8 sub-scales per fp16 super-scale | 128 weights / 74 bytes |

For the hierarchical `_sq`/`_sg` formats: fp16 super-scale; 8 sub-scales per
super-block. Sub-scale signedness is a per-name note (unsigned for the
`sq4` entries, signed for `sq8`) until a signed/unsigned pair ever
coexists at the same width.

## Compatibility rules

- Byte layouts and their names are frozen once shipped; the loader keeps a
  byte-exact decoder for every registered name.
- Numeric ids are free to differ per fork/version; files self-describe.
- A future reader may *remap* (same bytes, new decoder) freely; changing
  bytes is a re-export, never an in-place reinterpretation.
