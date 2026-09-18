# Millie 1.1 text and vision

Build `llama-cli` from this release of Millie Runtime using the
[Apple Silicon instructions](build-millie.md#apple-silicon--metal).
The example below uses ordinary inference. Optional low-memory mode is
described in [Hot experts](hot-experts.md).

Download `Millie-1.1-35B-A3B-Ternary.gguf`, and optionally
`Millie-1.1-35B-A3B-vision.gguf`, from the Millie 1.1 model repository.
Verify them against that repository's `SHA256SUMS`. Run the following from the
runtime checkout, replacing `/path/to/` with your local file locations.

## Text, non-thinking mode

```sh
./build-metal/bin/llama-cli \
  -m /path/to/Millie-1.1-35B-A3B-Ternary.gguf \
  --jinja --reasoning off --single-turn \
  --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 \
  --presence-penalty 1.5 --repeat-penalty 1 \
  -p "Give me three ideas for a quick dinner."
```

The model's embedded chat template is used automatically. `--reasoning off`
selects non-thinking generation. Choose context length, batching, KV precision,
and GPU placement for your workload and available memory.

## Image input

Use the same command with these additional arguments and an image-specific prompt:

```sh
--mmproj /path/to/Millie-1.1-35B-A3B-vision.gguf \
--image /path/to/ingredients.jpg
```

For example, replace the text prompt with
`"What can I cook with these ingredients? Give me steps."`
The vision encoder adds memory and processing time. Image preprocessing depends
on the image dimensions and caller configuration.

Optional hot-expert mode changes routing and reduces resident expert memory.
See [Hot experts](hot-experts.md) for its parameters and constraints, and the
model card for the configurations used in reported measurements.
