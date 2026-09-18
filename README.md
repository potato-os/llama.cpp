# Millie Runtime

Millie Runtime is LLMs For All's inference runtime, built on
[llama.cpp](https://github.com/ggml-org/llama.cpp). It supports Millie's custom
quantization formats, text and image input, and low-memory inference.

## Get started

1. Download the archive for your platform from [Releases](https://github.com/llmsforall/millie-runtime/releases) and extract it.
2. Download `Millie-1.1-35B-A3B-Ternary.gguf` from [Hugging Face](https://huggingface.co/llmsforall/Millie-1.1-35B-A3B-Ternary).
3. Open a terminal in the extracted runtime folder and run:

```sh
./bin/llama-cli \
  -m /path/to/Millie-1.1-35B-A3B-Ternary.gguf \
  --jinja --reasoning off \
  --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 \
  --presence-penalty 1.5 --repeat-penalty 1
```

Replace `/path/to/` with your model's location, then type a message to chat.
For thinking, use `--reasoning on --temp 1.0 --top-p 0.95` instead.

For images, also download `Millie-1.1-35B-A3B-vision.gguf` from the same model
page and add these arguments:

```sh
--mmproj /path/to/Millie-1.1-35B-A3B-vision.gguf --image /path/to/photo.jpg
```

The runtime supports ordinary inference and a low-memory mode that keeps a
bounded number of experts resident per layer. Low-memory mode currently
requires Apple Metal. See [low-memory setup](docs/hot-experts.md).

Prefer to build it yourself? See [build instructions](docs/build-millie.md).

## More information

- [Custom tensor formats](docs/tensor-types.md)
- [Command-line usage](tools/cli/README.md)
- [Local server and API](tools/server/README.md)
- [Millie CLI](https://github.com/llmsforall/millie-cli), a separate application using this runtime
- [Upstream llama.cpp documentation](https://github.com/ggml-org/llama.cpp#readme)

Existing `llama_*` APIs and executable names such as `llama-cli` and
`llama-server` are retained for compatibility.

## License

MIT licensed. See [LICENSE](LICENSE) for the ggml authors' and LLMs For All's
copyright notices and license terms.
