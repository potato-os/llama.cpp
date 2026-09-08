# Flash-Next top-k fallback: build and compare

This fork includes the locally tested `0005r1` CUDA top-k change: use existing radix selection for rows of at least 8192 columns when the CUB DeviceTopK path is unavailable. `GGML_CUDA_TOPK_ARGSORT=1` restores the old fallback in the same binary. This is an experimental combined Flash-Next branch, not upstream llama.cpp.

## Scope and credit

The base branch already combines the expert cache (#27861), pinned-host loader work (#28223), the historical MTP working diff (#28243), and local cache/duplicate-id changes. This update adds the preserved #28198 graph-opt guard change and the exact tested top-k source patch. Leave `GGML_CUDA_GRAPH_OPT` unset for this comparison. The parked runtime `/props` thread-setting experiment is not included.

The radix kernels already existed in llama.cpp. [Rhonstin's #28366](https://github.com/ggml-org/llama.cpp/pull/28366) proposes the same fallback approach, building on [#27466](https://github.com/ggml-org/llama.cpp/pull/27466). This local variant has an 8192-column threshold and an A/B switch; the numbers here are not measurements of the exact upstream PR head.

Our measured environment was CUDA toolkit 12.0, CUB 2.0.1, two RTX 3090s, two E5-2696 v4 CPUs and four 32 GB DDR4-2133 DIMMs. A compatible newer CCCL build has another optimized path, DeviceTopK, which has not been compared with this patch on this machine. This result is against the previous fallback, not against the latest CUDA stack. Installing a newer driver alone does not change the compiled path.

## Build on Linux

Use an isolated checkout and build directory. You need Git, CMake, a C++ compiler, a compatible CUDA toolkit, Python 3 and numactl for the example dual-socket launch below. Existing GGUF files can be reused.

```sh
git clone --branch flashnext-2x3090 https://github.com/Inovello/llama.cpp.git llama-flashnext-topk
cd llama-flashnext-topk
git rev-parse HEAD
nvcc --version
cmake -S . -B build-topk -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_BUILD_TESTS=ON
cmake --build build-topk --parallel 4 --target llama-server test-backend-ops
```

Record the commit and the actual compiler/toolkit used by CMake. For NVIDIA builds where `CUB_TOP_K_AVAILABLE` is defined, DeviceTopK takes precedence and the fallback switch has no effect. HIP follows its existing path and is not this A/B. Eligible wide rows are also required; a short prompt can legitimately show no difference.

On an otherwise idle GPU, run the existing operator checks in separate processes:

```sh
env -u GGML_CUDA_TOPK_ARGSORT ./build-topk/bin/test-backend-ops test -b CUDA0 -o TOP_K
GGML_CUDA_TOPK_ARGSORT=1 ./build-topk/bin/test-backend-ops test -b CUDA0 -o TOP_K
```

The benchmark records cited below came from the preserved production build. The publication checkout was checked against the preserved source hashes but has not undergone a new CUDA build or model run as part of publication. Building the same source locally does not guarantee matching compiler output or performance.

## Launch the two arms manually

Run one server at a time on an idle machine. Do not start these commands alongside an existing model server. Stop an arm yourself, wait for its process and host/GPU allocations to be released, then start the other. The client below never launches or stops a server.

Example settings for our two-GPU host-offload setup, using your existing model paths:

```sh
export MODEL=/absolute/path/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
export DRAFT=/absolute/path/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
unset GGML_CUDA_GRAPH_OPT
export GGML_CUDA_TOPK_ARGSORT=1

LLAMA_ATTN_ROT_DISABLE=1 numactl --interleave=all ./build-topk/bin/llama-server \
  -m "$MODEL" -md "$DRAFT" \
  --spec-type draft-mtp -devd CUDA1 --spec-draft-n-max 3 \
  -ngl 99 -c 261888 --parallel 1 -fa on \
  -ot 'ffn_(gate|up|down)_exps\.weight=CUDA_Host,per_layer_token_embd\.weight=CPU' \
  -lzm off --numa distribute -t 16 -tb 44 -b 4096 -ub 512 \
  -ctk f16 -ctv f16 --moe-expert-cache 150 -lv 4 \
  --host 127.0.0.1 --port 18080
```

That is the control. For the candidate, stop the control, wait for resources to drain, run `unset GGML_CUDA_TOPK_ARGSORT`, and repeat the identical launch. The flag is cached within the process: changing a shell variable while the server is running does not switch the server's path.

The example requires the main model and its compatible shared MTP head, sufficient host RAM and VRAM. Keep whatever hardware-specific settings you need identical between arms. The NUMA settings shown are for a dual-socket host, not universal defaults.

For stronger activation evidence, capture the actual long request with a compatible Nsight Systems version that records CUDA-graph kernels completely. In the eligible candidate path, expect `top_k_radix_init`, `top_k_radix_histogram`, `top_k_radix_select`, `top_k_radix_reset_counters` and `top_k_radix_gather`. Compare against the control's argsort kernels. An environment flag, a kernel present in the executable, or equal timings alone does not prove which path executed. Keep profiled runs separate from throughput measurements.

## Measure your own document

Create a UTF-8 `prompt.txt` containing a long document and a question that should elicit a substantial answer. Use identical bytes for both arms. The client calls `/tokenize` and reports the supplied token count. Stay within the server's context capacity, including the requested output. The original approximately 119k project-document corpus is not included; this reproduces the comparison method, not its exact workload or quality scores.

Once the server is healthy, from a second terminal:

```sh
python3 examples/flashnext-topk/bench.py --prompt prompt.txt --output control-seed1.json --seed 1 --tokens 512
```

After manually switching to the candidate server:

```sh
python3 examples/flashnext-topk/bench.py --prompt prompt.txt --output candidate-seed1.json --seed 1 --tokens 512
```

Use the same Python client and prompt for both arms. It sends explicit production-style sampling: temperature 0.7, top-p 0.8, top-k 20, min-p 0, seed supplied on the command line. It uses the raw completion endpoint; prepare your prompt accordingly. This is not the original chat-based quality harness and does not grade answers or automatically impose that harness's thinking format.

Repeat with seeds 2 and 3 and unique output filenames, reversing the arm order for seed 2. Use a fresh server process for each arm/seed if following the original launch design. For warmed-prefix decode, make an identical warmup request in each arm, preserve its output separately, then measure; do not mix first-request and warmed results. `cache_prompt` is enabled, but a fresh process has no prefix cached yet.

Each output preserves the prompt hash, supplied token count, request, full response, server timings and completion length. Files are created exclusively, never overwritten. A response shorter than the requested output is retained but labelled invalid for this fixed-length speed comparison. Do not silently pool a few-token early-EOS rate with a full 512-token continuation. Do not force an artificial continuation and call it a quality result either.

Compare decode throughput from server `predicted_n / (predicted_ms / 1000)`, and keep prefill time and total request latency separate. Use repeated runs, medians and ranges. Sampling can change the answer and MTP acceptance even with the same seed. Read both answers; throughput alone cannot establish preserved quality. Nothing in this helper establishes the original quality-screen result for a new machine or workload.

## Recorded results

Same deployed binary, old fallback versus radix, MTP-3 and production sampling, approximately 119k starting document depth. Each median is over 42 requests. Ranges below are request minima and maxima, not confidence intervals.

| Seed | Control median [min, max] t/s | Candidate median [min, max] t/s |
|---|---:|---:|
| 1 | 30.2 [27.3, 34.6] | 33.7 [30.3, 37.8] |
| 2 | 30.2 [27.1, 34.1] | 33.3 [30.3, 36.9] |
| 3 | 30.4 [27.3, 34.4] | 33.1 [29.0, 37.5] |

This is approximately 9-12% higher per-seed median decode throughput. It compares arm medians, not the median of question-paired speedups. It does not establish a prefill or no-MTP gain.

The frozen project-document quality screen covered 38 question/depth combinations at approximately 37k tokens and 42 at approximately 119k, repeated across three seeds and both arms: 480 requests, 240 paired comparisons. Control scored 235/240; candidate 238/240. The candidate was better on four pairs, worse on one and equal on 235. One twelve-item recall question was inconclusive: both arms missed one item on two seeds each. All outcomes remain in the totals. No consistent quality regression was detected in this bounded screen; it does not prove equivalent or improved quality.

## Source and measurement provenance

- Previous public branch: `9bd97fe54833d02fee8a56e8e476f0069bd94006`.
- All 24 file postimage hashes in the preserved production MTP working diff match that public branch.
- Preserved graph-opt change: upstream `0ba6499c3ba73ed408acac62ea650d3af613794a`, retained here with its original authorship. It remains opt-in and was unset in the quality comparison.
- Tested top-k source blob: `cf4122b27c68b410bfbed9a698479960135021be`; preceding blob: `c7a0c831788df8547ffd86c8a68e6d53aff61130`.
- Measured CUDA library MD5: `324165052b1d1199de869824e0fff4a7`; measured server library MD5: `8fe204dda5900636af1629436fbd3911`. These identify the recorded Linux artifacts, not expected hashes for a rebuild.
- Historical TOP_K operator suite: 525/525 cases passed on both paths, including 268 cases at or above the threshold and 88 tied cases among them. No claim of universal identical tied indices or model outputs.
- Source matching is scoped to the compared files; this publication does not claim a full independently verified bit-for-bit reconstruction of the original deployment.

Preparation of this fork update and benchmark helper was AI-assisted and based on preserved source and measurement records. No new upstream PR is being submitted by this publication.
