# Building the Millie Runtime

For Millie 1.1, build from source using the commands below, then follow the
[Millie 1.1 quickstart](millie-1.1.md). Older CLI bundles do not contain the new
hot-expert and vision support.

To build from source, clone this fork. Use the corresponding release tag when
available; upstream binaries do not contain this fork's custom format support.

```sh
git clone https://github.com/llmsforall/millie-runtime.git
cd millie-runtime
```

Millie CLI is not required for a standalone runtime build. Clone its repository
as a sibling only if you also want to build the CLI or complete release bundles.

Run the following commands from the `millie-runtime` checkout. Install CMake, a C/C++
compiler and the development tools for the selected backend first.

## Apple Silicon / Metal

Use Xcode Command Line Tools and CMake:

```sh
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_SERVER=ON
cmake --build build-metal --target llama-cli llama-server -j 8
```

## Linux / Vulkan

Install the Vulkan headers and loader development library, the shaderc `glslc`
compiler, CMake and a C/C++ toolchain. The machine running inference also needs
a working Vulkan driver for its GPU.

```sh
cmake -S . -B build-vulkan -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_VULKAN=ON -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_SERVER=ON
cmake --build build-vulkan --target llama-cli llama-server -j 8
```

These commands build for the local development environment. For release compiler
versions, the portable Linux target and the bundled loader patches, see the
[release build guide](https://github.com/llmsforall/millie-cli/blob/main/docs/release-builds.md).
Other backend instructions are in the [upstream build guide](build.md).

## Connect it to Millie CLI

The executable is `build-metal/bin/llama-server` or
`build-vulkan/bin/llama-server`. Set `[llamacpp] server_bin` in Millie's config
to its absolute path. Install this checkout's `millie-native.jinja` beside the
`millie` executable, or set `MILLIE_LLAMACPP_CHAT_TEMPLATE` to its absolute path.
The CLI also needs its `system_prompt.md` and `millie-models.json` runtime files.
See the [CLI source-build instructions](https://github.com/llmsforall/millie-cli/blob/main/docs/install.md#build-from-source)
for the complete layout. Model weights are downloaded separately.
