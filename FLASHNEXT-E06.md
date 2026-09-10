# flashnext-e06: phase-memory prefill for host-resident MoE experts

Branch of the `flashnext-2x3090` setup (Qwen3.8-Flash-Next on 2x RTX 3090 with all expert layers pinned in host RAM
and an LRU expert cache in VRAM). This branch adds one thing: during a prompt, the expert cache and the decode buffers
are released from GPU memory and the prompt runs at a 2048-token micro-batch; before the first generated token the
cache and the decode buffers are restored and decode runs exactly as before.

What it buys on the 2x3090 / DDR4-2133 box (6 DIMMs, medians, fresh server per arm): prefill 99.9 -> 223.7 t/s on an
8k prompt (2.24x), 88.1 -> 212.6 t/s at ~37k context (2.41x), 81.3 -> 206.5 t/s at ~119k (2.54x); decode unchanged
(+2% / -1% / 0%). Fixed cost about 2.8 s per prompt for the release and restore. Write-ups with the full method are at
https://www.inovello.dev/writeups/ (parts 1 to 4).

## What is on this branch

`flashnext-2x3090` (llama.cpp master b96806d + PR #27861 expert cache + PR #28223 host buffer overrides under mmap +
PR #28243 MTP draft head + the batched-cache fixes + PR #28198 concurrent streams per split + the radix-select top-k of
PR #28671), plus:

- **E06 phase memory**: `llama_experimental_prefill_begin` / `_end` (`include/llama.h`, `src/llama-context.cpp`), the
  cache release and restore (`src/llama-moecache.cpp`), CUDA pool release (`ggml/src/ggml-cuda/ggml-cuda.cu`), and the
  server hook that wraps each prompt in a transaction (`tools/server/server-context.cpp`).
- **bypass-control diagnostics**: extra switches used while qualifying the change. Inert unless the environment
  variables below are set.

## How to enable it

Everything is off unless `LLAMA_PHASE_PREFILL_UBATCH` is set in the server's environment.

```
export LLAMA_PHASE_PREFILL_UBATCH=2048
export LLAMA_PHASE_PREFILL_MODE=transaction
```

`transaction` (the default when MODE is unset) makes the release and restore all-or-nothing: a transition that cannot
complete is an error, the server does not continue in a half-restored state.

The launch line these numbers were measured with:

```
export LLAMA_ATTN_ROT_DISABLE=1
export LLAMA_MMAP_PIN_HOST=1
export LLAMA_PHASE_PREFILL_UBATCH=2048
export LLAMA_PHASE_PREFILL_MODE=transaction

numactl --interleave=all build/bin/llama-server \
  -m /path/to/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -md /path/to/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf --spec-type draft-mtp -devd CUDA1 --spec-draft-n-max 3 \
  -ngl 99 -c 261888 --parallel 1 --flash-attn on \
  -ot "ffn_(gate|up|down)_exps\.weight=CUDA_Host,per_layer_token_embd\.weight=CPU" \
  -lzm off --numa distribute -t 16 -tb 44 -b 4096 -ub 512 -ctk f16 -ctv f16 \
  --moe-expert-cache 150 -lv 4
```

## What the code insists on

These are checked at startup or at the first transition and refuse otherwise. They are the configuration that was
qualified, not a claim that other values cannot work.

| requirement | where it is checked |
| --- | --- |
| `LLAMA_PHASE_PREFILL_UBATCH` is exactly `512` or `2048` | `tools/server/server-context.cpp`, `src/llama-context.cpp` (`llama_experimental_prefill_begin`) |
| server micro-batch `-ub 512`, batch `-b 4096`, `--parallel 1` | `tools/server/server-context.cpp` |
| expert cache exactly 150 slots (`--moe-expert-cache 150`) | `src/llama-context.cpp` (`llama_moe_cache_get_capacity() == 150`) |
| one speculative decoder, the MTP draft (`--spec-type draft-mtp`) | `tools/server/server-context.cpp` |
| CUDA backend | the release and restore paths are CUDA only |

## Diagnostic switches (bypass-control)

Off unless set. They exist to reproduce the qualification runs and are not needed to use the branch.

- `LLAMA_PHASE_PREFILL_MODE=bypass`: the control arm. Runs the same quiesce and settle at the same two boundaries as
  `transaction`, with cache lookup and admission disabled and the reusable graph results invalidated, but releases,
  reallocates and restores nothing; the micro-batch stays at 512. Requires `LLAMA_PHASE_PREFILL_UBATCH=512`. It exists
  so the transition itself can be measured against the memory swap.
- `LLAMA_PHASE_PREFILL_LOGIT_DUMP=<dir>`: writes the first sampled token's logits per request into `<dir>` (one file
  per slot and request) for A/B comparison between modes.
- `LLAMA_PHASE_PREFILL_VERIFY=1`: hash the output buffers in the invariant snapshot taken before and after each
  transition and log mismatches. Costs time; leave off in production.

## Measured on, and not measured on

Measured: one machine (2x RTX 3090, dual Broadwell Xeon, 6x32 GB DDR4-2133 ECC, CUDA 12.0), one model
(Qwen3.8-Flash-Next UD-Q4_K_XL with the shared Q8_0 MTP head), 8k to 119k context, greedy and production sampling.
Quality: 240 paired long-context questions over three seeds, 2 worse / 235 equal / 3 better against a same-day control.

Not measured: contexts above 119k, other models or quantizations, the no-MTP configuration, other GPUs or backends.
First-token logits differ from the untouched path by at most 1.51 (mean 0.22 over the 248k vocabulary) with the argmax
unchanged; changing the micro-batch from 512 to 2048 alone moves them by 1.81 / 0.27 on the same request.
