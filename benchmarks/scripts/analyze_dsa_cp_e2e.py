# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize the four-arm benchmark without treating missing runs as success."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import regex as re


def evaluate_acceptance(summaries, mode):
    expected = {
        (mode, length, concurrency, output)
        for length in (1024, 4096, 8192)
        for concurrency in (1, 8)
        for output in (1, 256)
    }
    complete = all(key in summaries and summaries[key]["rounds"] >= 3 for key in expected)
    cells = [summaries[key] for key in sorted(expected & summaries.keys())]
    outputs_match = (
        all(row.get("matches_baseline_outputs", False) and row["output_length_ok"] for row in cells) if cells else None
    )
    regressions, improvements = [], []
    for key in sorted(expected & summaries.keys()):
        row = summaries[key]
        values = [row.get("ttft_improvement_percent", 0)]
        if key[3] > 1:
            values.append(row.get("tpot_improvement_percent", 0))
        if min(values) < -2 - 1e-9:
            regressions.append(list(key[1:]))
        if max(values) >= 5 - 1e-9:
            improvements.append(list(key[1:]))
    return dict(
        passed=complete and outputs_match is True and bool(improvements) and not regressions,
        complete=complete,
        outputs_match=outputs_match,
        improved_at_least_5_percent=improvements,
        regressed_more_than_2_percent=regressions,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    groups = defaultdict(list)
    for path in args.directory.glob("*/requests.jsonl"):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            key = (record["variant"], record["input_tokens"], record["concurrency"], record["requested_output"])
            groups[key].append(record)
    summaries = {}
    for key, batches in groups.items():
        requests = [r for batch in batches for r in batch["requests"]]
        summaries[key] = dict(
            variant=key[0],
            input_tokens=key[1],
            concurrency=key[2],
            output_tokens=key[3],
            rounds=len(batches),
            ttft_ms=statistics.median(r["ttft_ms"] for r in requests),
            tpot_ms=statistics.median(r["tpot_ms"] for r in requests),
            tokens_per_second=statistics.median(b["tokens_per_second"] for b in batches),
            hashes=sorted({r["sha256"] for r in requests}),
            output_length_ok=all(r["output_tokens"] == key[3] for r in requests),
        )
    for key, summary in summaries.items():
        baseline = summaries.get(("legacy", *key[1:]))
        if baseline:
            summary["matches_baseline_outputs"] = summary["hashes"] == baseline["hashes"]
            summary["ttft_improvement_percent"] = (1 - summary["ttft_ms"] / baseline["ttft_ms"]) * 100
            if key[3] > 1:
                summary["tpot_improvement_percent"] = (1 - summary["tpot_ms"] / baseline["tpot_ms"]) * 100
    expected = {
        (v, length, c, out)
        for v in ("legacy", "async", "fused_only", "fused")
        for length in (1024, 4096, 8192)
        for c in (1, 8)
        for out in (1, 256)
    }
    capacity = {}
    for path in args.directory.glob("*/server.log"):
        selected = [
            line
            for line in path.read_text(errors="replace").splitlines()
            if any(marker in line.lower() for marker in ("kv cache size", "maximum concurrency", "preempt"))
        ]
        capacity[path.parent.name] = selected
        match = re.search(r"KV cache size: ([\d,]+) tokens", "\n".join(selected))
        if match:
            tokens = int(match[1].replace(",", ""))
            for key, summary in summaries.items():
                if key[0] == path.parent.name:
                    summary["kv_capacity_tokens"] = tokens
                    summary["requested_kv_tokens_exceed_capacity"] = key[2] * (key[1] + key[3]) > tokens
    report = dict(
        server_capacity_and_preemption=capacity,
        complete=all(k in summaries and summaries[k]["rounds"] >= 3 for k in expected),
        missing=[list(k) for k in sorted(expected - summaries.keys())],
        insufficient_rounds=[list(k) for k in sorted(expected & summaries.keys()) if summaries[k]["rounds"] < 3],
        scenarios=[summaries[k] for k in sorted(summaries)],
        default_mode="legacy",
        acceptance_by_variant={mode: evaluate_acceptance(summaries, mode) for mode in ("async", "fused_only", "fused")},
    )
    target = args.directory / "summary.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    columns = [
        "variant",
        "input_tokens",
        "concurrency",
        "output_tokens",
        "rounds",
        "ttft_ms",
        "tpot_ms",
        "tokens_per_second",
        "ttft_improvement_percent",
        "tpot_improvement_percent",
        "matches_baseline_outputs",
        "output_length_ok",
        "kv_capacity_tokens",
        "requested_kv_tokens_exceed_capacity",
    ]
    with (args.directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, columns, extrasaction="ignore")
        writer.writeheader()
        for row in report["scenarios"]:
            writer.writerow({key: round(value, 3) if isinstance(value, float) else value for key, value in row.items()})
    print(json.dumps(dict(complete=report["complete"], scenarios=len(summaries), summary=str(target))))


if __name__ == "__main__":
    main()
