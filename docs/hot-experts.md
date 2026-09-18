# Hot experts

Hot-expert inference keeps a bounded set of routed experts resident per layer and reads selected cold experts from the GGUF as needed. The complete model remains on disk.

Build with the Metal backend. Command template (replace uppercase placeholders
with values chosen for your workload):

```sh
llama-cli -m model.gguf --hot-experts EXPERT_COUNT --hot-expert-bias ROUTING_BIAS -ngl all -np 1 -c CONTEXT_LENGTH -ub MICROBATCH_SIZE
```

- `--hot-experts N`: resident experts per layer, from 8 to 256. The default, 0, uses ordinary inference.
- `--hot-expert-bias N`: nonnegative additive routing-logit bias for resident experts; default 1.0. It changes expert selection. Mixture weights are computed from the original logits and normalized over the selected experts.

The current implementation supports the Qwen3.5 MoE architecture with 256 experts and eight selected per token, separate gate/up/down expert tensors, and full layer offload to one Metal device. It supports one sequence and one live context per model. LoRA and custom tensor placement are not supported. Microbatches must contain 1–512 tokens.

Selected experts move to the front of each layer's recency queue in routing order. Selected cold experts replace the least recent unselected entries and are used in the same step. Experts that remain resident keep their physical slots. During prefill, a reusable workspace holds the union of selected experts for one layer; the persistent cache is updated before proceeding to the next layer.

Expert weights use owned buffers rather than whole-file memory mapping. On supported platforms, the CPU input embedding is separately memory-mapped without prefetching. With memory locking enabled, it uses an owned allocation instead. This reduces resident weight allocations but adds storage reads and synchronization. Total memory also includes non-expert weights, conversation state, compute buffers, and the prefill workspace. The hot-expert count is not a total-memory limit.

If present, the GGUF `millie.expert_initial_ids` uint32 array supplies initial expert IDs, with equally sized rows for each layer. Otherwise, each layer uses a seeded permutation. Saved context and sequence states include the expert queue and slot mapping; restore them with the same model, hot-expert count, and routing bias.

In the C API, set `llama_model_params.hot_experts` and `llama_model_params.hot_expert_bias` before loading the model. Keep the model alive until its context is destroyed.

Normal warm-up retains top-8 routing in hot mode. It does not attempt to select all experts at once.
