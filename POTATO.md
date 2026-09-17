# llama.cpp for Potato-CODER

This branch (`potato-main`) is [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `master` plus two changes that the
Potato-CODER models need. Everything else is stock llama.cpp, see [README.md](README.md).

| change | what it does | needed by |
|---|---|---|
| qwen4exp NextN/MTP draft head | loads the `mtp-*.gguf` draft head of Qwen3.8-Flash-Next models (`--spec-type draft-mtp -md <head>`); squash of upstream PR #28243 by Daniel Han | Potato-CODER-48GB |
| `--mmproj-swap-draft` | keeps the vision projector off the GPU and, when an image arrives, encodes it on a temporary GPU projector while the MTP draft context releases its VRAM; falls back to the CPU projector when VRAM is still short | both, when the cards are full |

The build also compiles the `q8_0`/`q5_0` flash-attention kernel pair that Potato-CODER-24GB serves with.

## Binaries

Prebuilt zips are on the [releases page](../../releases):

| file | for |
|---|---|
| `llama-potato-bin-linux-cuda-12.8-x64.tar.gz` | Linux x64, NVIDIA driver 570+, RTX 20xx to 50xx |
| `llama-potato-bin-win-cuda-12.4-x64.zip` | Windows x64, NVIDIA driver 551+, RTX 20xx to 40xx |
| `llama-potato-bin-win-cuda-13.4-x64.zip` | Windows x64, NVIDIA driver 580+, adds RTX 50xx |

The Windows zips include the CUDA runtime DLLs. Unpack and run `llama-server` from the folder.

## Build from source

Linux:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DGGML_CUDA_FA_QUANTS="q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16;q8_0-q5_0"
cmake --build build -j --target llama-server
```

Windows (x64 Native Tools prompt of Visual Studio 2022, CUDA toolkit installed):

```bat
cmake -B build -G "Ninja Multi-Config" -DGGML_CUDA=ON -DLLAMA_BUILD_BORINGSSL=ON ^
  -DGGML_CUDA_FA_QUANTS="q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16;q8_0-q5_0"
cmake --build build --config Release --target llama-server
```

## Serving

Potato-CODER-48GB, two 24 GB cards, 262k context, vision and the MTP head:

```bash
llama-server -m Potato-CODER-48GB-IQ4_XS-00001-of-00002.gguf \
  -c 262144 -ngl 49 -sm layer -dev CUDA0,CUDA1 -mg 0 -ts 1.1,1 -fit off \
  -ctk f16 -ctv f16 -fa on -np 1 -ub 512 -b 4096 --load-mode none \
  --cache-prompt --cache-reuse 256 --ctx-checkpoints 8 --cache-ram 0 \
  --jinja --chat-template-kwargs '{"preserve_thinking": false}' \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --mmproj mmproj-Potato-CODER-48GB-Q8_0.gguf --no-mmproj-offload --mmproj-swap-draft --image-max-tokens 2048 \
  --spec-type draft-mtp --spec-draft-n-max 2 -md mtp-Potato-CODER-48GB-shared-Q3_K_M.gguf -devd CUDA1
```

Potato-CODER-24GB, one 24 GB card, 262k context, vision and MTP:

```bash
llama-server -m Potato-CODER-24GB-IQ4_XS.gguf \
  -c 262144 -ngl 99 -fa on -ctk q8_0 -ctv q5_0 -np 1 -ub 256 -b 2048 --no-context-shift \
  --jinja --chat-template-kwargs '{"preserve_thinking": false}' \
  --mmproj mmproj-Potato-CODER-24GB-Q8_0.gguf --no-mmproj-offload --mmproj-swap-draft \
  --spec-type draft-mtp --spec-draft-n-max 3
```

Both layouts fill the cards to within a few hundred MiB. The card that drives a display has less free memory; lower `-c`
if the server runs out of memory at start.

### Windows notes

- Turn off the driver's system-memory fallback for `llama-server.exe` (NVIDIA Control Panel, Manage 3D settings,
  "CUDA - Sysmem Fallback Policy": "Prefer No Sysmem Fallback"). With the fallback on, a full card spills into system RAM
  over PCIe and generation drops to a few tokens per second instead of failing visibly.
- Windows keeps a part of each card for the desktop. Expect to need a smaller `-c` than on a headless Linux box.
- In `cmd.exe` write the JSON argument as `"{\"preserve_thinking\": false}"`.

## Updating this branch

`master` of this fork mirrors upstream. `potato-main` is rebased onto it; release tags are `potato-v*`.
