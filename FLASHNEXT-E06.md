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

## E06b / E06b2 additions (2026-09-10)

### `LLAMA_PHASE_PREFILL_MIN_TOKENS` (E06b)

New environment variable, parsed once at server startup alongside `LLAMA_PHASE_PREFILL_UBATCH`
and `LLAMA_PHASE_PREFILL_MODE`. Unset, or set to `0`, is the default and reproduces E06's
existing behavior exactly (no minimum). Set to a decimal integer from `1` to `1000000000`,
it gates the phase-prefill BEGIN transaction on the number of prompt tokens the current
request still has to process: `pending = total prompt tokens - cached prefix tokens at the
start of this prompt`. The transaction now begins only when `pending >= LLAMA_PHASE_PREFILL_MIN_TOKENS`
(in addition to the pre-existing `batch tokens > 8` check); a prompt whose pending count is
below the threshold runs the unmodified (non-phase-prefill) path for its entire prompt. The
decision is made once per prompt (not re-evaluated per batch) and is logged as
`phase_prefill DECISION slot=<n> task=<n> pending=<n> cached=<n> total=<n> min_tokens=<n> taken=<0|1>`.
Any malformed value refuses startup with `SRV_ERR`, matching the neighboring phase-prefill
variables.

Reason: the 6-DIMM depth window (report 77, `qc-e06-depth6`, cached-prefix follow-up turns,
`cache_n > 0`) measured that E06's transaction fires on every prompt batch over 8 tokens
regardless of how little of the prompt is actually new, so a short follow-up on an already-
cached prefix pays the full release/restore swap. Median prompt time for a ~170-200-token
follow-up: control 6.66 s (depth A) / 7.67 s (depth B), E06 10.90 s / 11.85 s -- a +4.2 s cost
per follow-up at both depths, larger than the ~2.8 s transition itself (2048 reserve + graph
rebuild on top of it). Against a prefill saving of only ~0.9 s on that short a prompt, the net
was about -3.3 s per short turn, eroding roughly 155-170 s per conversation against the
248 s (A) / 886 s (B) gained on the conversation's first (long, uncached) prefill.
`LLAMA_PHASE_PREFILL_MIN_TOKENS` lets a short cached follow-up skip the transaction while a
long fresh prompt still takes it.

### Expert-cache capacity unpin (E06b2)

`src/llama-context.cpp`'s three cache-consistency checks around the phase-prefill BEGIN/
bypass_end/restore transitions (`llama_experimental_prefill_begin`/`_end`) used to hardcode the
literal expert-cache capacity `150` (the value every build and comparison run in this campaign
has launched `llama-server` with, via `--moe-expert-cache 150`). BEGIN now only requires an
*active* cache with `capacity > 0`, and records that capacity into a transaction-local
`began_capacity` (a `static int32_t`, matching `llama_moe_cache_get_capacity()`'s return type,
declared next to the existing `released_generation`); the bypass_end and restore checks compare
the cache's capacity against `began_capacity` instead of the literal `150`. The invariant this
was already enforcing -- restore returns the cache to the exact capacity it had when the
transaction began -- is unchanged; only the magic number is gone, so `--moe-expert-cache 150`
still behaves identically, and any other capacity (verified at `120`) now works too. Three
error-message strings lost their embedded `150` in the same edit (`"begin requires ubatch512
pair, requested512/2048 and a ready active cache"`, `"cache bypass_end failed:
phase/capacity/active"`, `"cache restore failed: phase/capacity/active/generation"`); no other
`150` pin tied to the expert cache exists anywhere in `src/`, `tools/server/`, or `include/`.
See `e06b2.patch` (this change alone, ~10 lines against `source-e06b`), `e06b2-from-base.patch`
(both E06b changes, against `source-base`), `tests/e06b2/` (the new
`check-capacity-unpin.py` structural check plus retargeted copies of E06's and E06b's own
checks, all passing against `source-e06b2`), and `smoke/` (a 2-arm on-box smoke kit,
`--moe-expert-cache 150` and `120`, confirming the BEGIN/END/DECISION and
`moe-cache-phase ... capacity=<N>` evidence for each).

`LLAMA_PHASE_PREFILL_UBATCH=4096` is now also accepted (in addition to `512`/`2048`, whose
behavior is unchanged) as an experimental value, measured only in report 79 so far and needing
roughly 15 GiB of compute buffer per card.

Measured threshold: report 78 (crossover on this machine, 8k and 37k cached prefixes, 64 to 4096 new tokens): the
swap costs about 2.5 s on a short follow-up and first pays off at 2048 new tokens, so production runs with
`LLAMA_PHASE_PREFILL_MIN_TOKENS=2048`. 150 is the only expert-cache capacity measured for speed and quality; 120 was
smoke-tested for the mechanism only. `LLAMA_PHASE_PREFILL_UBATCH=4096` fails at the first swap on 24 GiB cards (the
reserve needs about 14 GiB on the draft card; report 79) and is not for production use.
