#!/usr/bin/env python3
"""Record one prompt-completion benchmark against an already running server."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import urllib.request


def post(url, endpoint, body):
    request = urllib.request.Request(
        url.rstrip("/") + endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=7200) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError(endpoint + " did not return a JSON object")
    return result


def positive_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New JSON evidence file; existing files are never overwritten")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=512)
    args = parser.parse_args()
    if args.tokens <= 0:
        parser.error("--tokens must be positive")

    prompt_bytes = args.prompt.read_bytes()
    prompt = prompt_bytes.decode("utf-8")
    # Reserve the evidence path before any network request.
    with args.output.open("x", encoding="utf-8") as output:
        record = {
            "url": args.url,
            "prompt_file": str(args.prompt),
            "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
            "requested_tokens": args.tokens,
            "status": "error",
        }
        exit_status = 1
        try:
            tokenized = post(args.url, "/tokenize", {"content": prompt, "add_special": True})
            if not isinstance(tokenized.get("tokens"), list) or not tokenized["tokens"]:
                raise ValueError("Missing or empty tokenize tokens")
            record["tokenized_prompt_count"] = len(tokenized["tokens"])
            request = {
                "prompt": prompt, "cache_prompt": True, "temperature": 0.7,
                "top_p": 0.8, "top_k": 20, "min_p": 0,
                "samplers": ["top_k", "top_p", "min_p", "temperature"],
                "seed": args.seed, "n_predict": args.tokens,
                "return_tokens": True, "stream": False,
            }
            record["request"] = request
            response = post(args.url, "/completion", request)
            record["response"] = response
            timings = response.get("timings")
            if not isinstance(timings, dict):
                raise ValueError("Missing timings object")
            record["timings"] = timings
            count, duration = timings.get("predicted_n"), timings.get("predicted_ms")
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise ValueError("Missing or invalid predicted_n")
            if not positive_number(duration):
                raise ValueError("Missing or invalid predicted_ms")
            record["predicted_tokens"] = count
            record["decode_tokens_per_second"] = count * 1000 / duration
            record["stop_type"] = response.get("stop_type")
            record["stopped_eos"] = response.get("stopped_eos")
            record["stopped_limit"] = response.get("stopped_limit")
            if count == args.tokens:
                record["status"] = "complete-length"
                exit_status = 0
            else:
                record["status"] = "screen-invalid-length"
                exit_status = 2
            print(f"{record['decode_tokens_per_second']:.3f} t/s; "
                  f"{count}/{args.tokens} tokens; {record['status']}")
            if count != args.tokens:
                print("Short or mismatched output is invalid for the fixed-length speed screen.")
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            print(record["error"], file=sys.stderr)
        finally:
            json.dump(record, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
    return exit_status


if __name__ == "__main__":
    sys.exit(main())
