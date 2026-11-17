# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated GLM DSA-CP four-arm serving benchmark (requires an idle TP16 node).

Run from the candidate source checkout with its PYTHONPATH selected. Each arm
owns one server process group and stops it in finally. Output includes exact
commands, logs, request-level measurements and deterministic output hashes.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import suppress
from pathlib import Path


def request(base, payload, first_token_event=None, capture_chunk_timing=False):
    req = urllib.request.Request(
        base + "/v1/completions", json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    begin, first, last, tokens, text = time.perf_counter(), None, None, 0, []
    chunk_intervals = []
    with urllib.request.urlopen(req, timeout=900) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            data = json.loads(line[6:])
            if "error" in data:
                raise RuntimeError(data["error"])
            now = time.perf_counter()
            for choice in data.get("choices", []):
                if choice.get("text"):
                    if capture_chunk_timing and last is not None:
                        chunk_intervals.append((now - last) * 1000)
                    if first is None:
                        first = now
                        if first_token_event is not None:
                            first_token_event.set()
                    last = now
                    text.append(choice["text"])
            if data.get("usage"):
                tokens = data["usage"]["completion_tokens"]
    if first is None or tokens == 0:
        raise RuntimeError("Server returned no output or usage")
    result = dict(
        ttft_ms=(first - begin) * 1000,
        latency_ms=(time.perf_counter() - begin) * 1000,
        tpot_ms=(last - first) * 1000 / max(tokens - 1, 1),
        output_tokens=tokens,
        sha256=hashlib.sha256("".join(text).encode()).hexdigest(),
    )
    if capture_chunk_timing:
        result["client_inter_chunk_ms"] = chunk_intervals
    return result


def profile_windows(base, out, seed_tokens):
    """Capture separate initial-prefill and steady-decode windows on all ranks."""

    def payload(length, output):
        return dict(
            model="glm53",
            prompt=(seed_tokens * (length // len(seed_tokens) + 1))[:length],
            max_tokens=output,
            temperature=0,
            seed=42,
            ignore_eos=True,
            stream=True,
            stream_options=dict(include_usage=True),
        )

    def control(action):
        req = urllib.request.Request(base + "/" + action, data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=900) as response:
            if response.status != 200:
                raise RuntimeError(f"{action} failed: {response.status}")

    def batch(length, concurrency, output):
        data = payload(length, output)
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            return list(executor.map(lambda _: request(base, data), range(concurrency)))

    batch(1024, 1, 8)
    batch(4096, 8, 8)
    windows = []
    for label, length, concurrency, steady in (
        ("1k-c1-initial", 1024, 1, False),
        ("4k-c8-initial", 4096, 8, False),
        ("4k-c8-decode", 4096, 8, True),
    ):
        before = set((out / "profiles").glob("*"))
        begin = time.perf_counter()
        if steady:
            events = [threading.Event() for _ in range(concurrency)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(request, base, payload(length, 128), event) for event in events]
                for event in events:
                    if not event.wait(timeout=600):
                        raise TimeoutError("Request did not enter decode before profiling")
                control("start_profile")
                try:
                    results = [future.result() for future in futures]
                finally:
                    control("stop_profile")
        else:
            control("start_profile")
            try:
                results = batch(length, concurrency, 128)
            finally:
                control("stop_profile")
        record = dict(
            label=label,
            duration_with_profiler_s=time.perf_counter() - begin,
            profile_dirs=[str(p.relative_to(out)) for p in sorted(set((out / "profiles").glob("*")) - before)],
            results=results,
        )
        windows.append(record)
        (out / "profile-windows.json").write_text(json.dumps(windows, indent=2) + "\n")
        print(json.dumps(dict(label=label, profile_dirs=len(record["profile_dirs"]))), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/GLM-5.3-w8a8")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["legacy", "async", "fused_only", "fused"],
        default=["legacy", "fused", "async", "fused_only"],
    )
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096, 8192])
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--outputs", type=int, nargs="+", default=[1, 256])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Capture separate all-rank prefill/decode traces instead of timing requests",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{args.port}"
    # Refuse to interfere with an existing server on the selected port.
    try:
        urllib.request.urlopen(base + "/health", timeout=2)
    except urllib.error.URLError:
        pass
    else:
        raise RuntimeError("Benchmark port is already occupied")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # Native runtimes may install their own handlers during import. Restore
    # Python interruption so the server process group is cleaned in finally.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    seed_tokens = tokenizer.encode(
        "Explain how distributed attention communicates key and value tensors. ", add_special_tokens=False
    )
    for variant in args.variants:
        out = args.output / variant
        out.mkdir(exist_ok=True)
        additional = dict(
            enable_dsa_cp=True,
            dsa_cp_comm_mode="async" if variant == "async" else "legacy" if variant == "legacy" else "fused",
            dsa_cp_comm_overlap=variant != "fused_only",
        )
        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            args.model,
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--served-model-name",
            "glm53",
            "--tensor-parallel-size",
            "16",
            "--enable-expert-parallel",
            "--distributed-executor-backend",
            "mp",
            "--quantization",
            "ascend",
            "--safetensors-load-strategy",
            "prefetch",
            "--max-model-len",
            "8704",
            "--max-num-seqs",
            "16",
            "--max-num-batched-tokens",
            "2048",
            "--gpu-memory-utilization",
            "0.90",
            "--no-enable-prefix-caching",
            "--compilation-config",
            json.dumps({"cudagraph_capture_sizes": [16]}),
            "--additional-config",
            json.dumps(additional),
        ]
        if args.eager:
            command.append("--enforce-eager")
        if args.profile_only:
            command += [
                "--profiler-config",
                json.dumps(
                    dict(
                        profiler="torch",
                        torch_profiler_dir=str(out / "profiles"),
                        torch_profiler_with_stack=False,
                        torch_profiler_with_memory=False,
                        ignore_frontend=True,
                        max_iterations=6,
                    )
                ),
            ]
        (out / "command.json").write_text(json.dumps(command, indent=2))
        with (out / "server.log").open("wb") as log:
            env = os.environ.copy()
            env.update(
                VLLM_USE_V2_MODEL_RUNNER="0",
                HCCL_OP_EXPANSION_MODE="AIV",
                HCCL_BUFFSIZE="400",
                OMP_NUM_THREADS="1",
                VLLM_NO_USAGE_STATS="1",
                VLLM_DO_NOT_TRACK="1",
            )
            server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (out / "pid").write_text(str(server.pid))
        try:
            deadline = time.monotonic() + 1800
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"{variant} server exited: see {out / 'server.log'}")
                try:
                    urllib.request.urlopen(base + "/health", timeout=2)
                    break
                except urllib.error.URLError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Server startup exceeded 30 minutes")
                    time.sleep(2)
            log_text = (out / "server.log").read_text(errors="replace")
            if "Native DSA-CP communication mode=" not in log_text:
                raise RuntimeError("DSA-CP implementation did not initialize; refusing an inactive-path benchmark")
            if args.profile_only:
                profile_windows(base, out, seed_tokens)
                continue
            with (out / "requests.jsonl").open("w") as results:
                for length in args.lengths:
                    prompt = (seed_tokens * (length // len(seed_tokens) + 1))[:length]
                    for concurrency in args.concurrencies:
                        for output in args.outputs:
                            payload = dict(
                                model="glm53",
                                prompt=prompt,
                                max_tokens=output,
                                temperature=0,
                                seed=42,
                                ignore_eos=True,
                                stream=True,
                                stream_options=dict(include_usage=True),
                            )
                            for iteration in range(-args.warmup, args.rounds):
                                begin = time.perf_counter()
                                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                                    batch = list(
                                        executor.map(
                                            lambda _, payload=payload: request(base, payload), range(concurrency)
                                        )
                                    )
                                elapsed = time.perf_counter() - begin
                                if iteration >= 0:
                                    record = dict(
                                        variant=variant,
                                        input_tokens=length,
                                        concurrency=concurrency,
                                        requested_output=output,
                                        iteration=iteration,
                                        tokens_per_second=sum(x["output_tokens"] for x in batch) / elapsed,
                                        requests=batch,
                                    )
                                    results.write(json.dumps(record) + "\n")
                                    results.flush()
                                    print(json.dumps(record), flush=True)
        finally:
            with suppress(ProcessLookupError):
                os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=30)
    if not args.profile_only:
        subprocess.run(
            [sys.executable, str(Path(__file__).with_name("analyze_dsa_cp_e2e.py")), str(args.output)], check=True
        )


if __name__ == "__main__":
    main()
