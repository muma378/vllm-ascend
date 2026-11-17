# Native DSA-CP communication

Native SFA DSA-CP can combine main KV and indexer K into one TP AllGather,
including optional indexer quantization scales. This is communication fusion
using existing PyTorch/HCCL operators; it adds no custom compute kernel.
The default remains `legacy`: the measured candidates have not passed the
end-to-end performance gate.

## Configuration

Use `--additional-config` alongside `enable_dsa_cp`:

| Field | Default | Behavior |
| --- | --- | --- |
| `dsa_cp_comm_mode` | `"legacy"` | Existing pipeline; `"async"` stages separate gathers; `"fused"` combines them. |
| `dsa_cp_comm_overlap` | `true` | Submit independent Q work before waiting for staged communication. |

These per-engine experiment controls use validated additional configuration
rather than environment variables. Invalid mode values are rejected.

For example:

```json
{"enable_dsa_cp": true, "dsa_cp_comm_mode": "fused", "dsa_cp_comm_overlap": true}
```

New scheduling applies to native DSA-CP with PCP=1, DCP=1 and unpacked main
KV caches. Combined communication also requires a runtime indexer supporting
staged updates. Other layouts retain the existing pipeline.

## Data flow and lifetime

Without indexer quantization scales, a runtime indexer layer changes from:

```text
legacy: AllGather(cat(k_pe, k_nope)), then separate AllGather(k_li)
fused:  AllGather(cat(k_pe, k_nope, k_li))
```

- K preparation runs once. The model-facing indexer wrapper forwards prepared
  K, gathered cache data and optional prepared Q to compatible backends.
- With overlap enabled, main Q and indexer Q run between collective submission
  and the cache-consumer dependency. Layers sharing top-k without their own
  indexer can overlap their main KV gather with Q, without combining fields.
- Homogeneous fields keep their native dtype. Mixed fields travel as aligned
  bytes without numerical casts. Rank-major token order is preserved.
- Per-layer workspaces use power-of-two capacity buckets. Captured graphs have
  stable storage; pending contexts retain input storage until communication
  finishes. Forwards for each layer must remain serialized.
- Main and indexer cache writes precede the existing cache-written notification.
  Skipping top-k still updates the indexer cache.
- PCP/DCP and packed main-cache paths keep their existing collective order.
  This does not add fused preprocessing or Model Runner V2 support.

## Graph configuration and benchmark

For the tested TP16 setup, retain the aligned bucket and allow it through the
sequence limit:

```text
--max-num-seqs 16 --compilation-config '{"cudagraph_capture_sizes":[16]}'
```

Client concurrency can still be 1 or 8. With a sequence limit of 8, the
16-token FULL decode key can be filtered out. Verify actual replay rather
than inferring it from configuration. Large prefill chunks exceed this bucket
and do not replay an ACL graph; compilation can remain enabled.

Transport correctness and microbenchmark, on an idle TP16 NPU node:

```bash
HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=400 OMP_NUM_THREADS=1 \
  torchrun --nproc-per-node=16 benchmarks/scripts/benchmark_dsa_cp_comm.py
```

This checks rank ordering, mixed dtypes, cache scatter and graph replay with
changed inputs. Its representative GEMM does not predict full-model TTFT/TPOT.

The serving benchmark starts and stops its own server on localhost port 18083:

```bash
PYTHONPATH="$PWD" python benchmarks/scripts/benchmark_dsa_cp_e2e.py \
  --model /models/GLM-5.3-w8a8 --output /tmp/dsa-cp-e2e
python benchmarks/scripts/analyze_dsa_cp_e2e.py /tmp/dsa-cp-e2e
```

It covers 1K/4K/8K inputs, concurrency 1/8, output length 1/256 and three
measured rounds, with prefix cache and MTP disabled. Commands, request timings,
output hashes and server logs are saved. Optional profiling runs separately
from timing. The acceptance gate requires at least one 5% TTFT/TPOT gain and
rejects any measured scenario with more than 2% regression.

## Measured FULL decode comparison

The 2026-11-16 GLM-5.3 W8A8 experiment used TP16/EP16, batch-token limit 2048,
maximum sequence count 16, graph bucket 16 and equal KV capacity of 37,248
tokens. Four independent server/capture arms were run: legacy before,
fusion without overlap, fusion with overlap, and legacy after.
All 648 measured requests had matching output hashes, lengths and completion.
A separate 16-rank audit verified FULL decode replay; audit instrumentation
was removed for timing.

The baseline is the mean of the two legacy arm medians. Each arm metric is
the median of the three round request medians. TPOT below uses 256 output tokens:

| Input / concurrency | Legacy TPOT (ms) | Fusion only gain | Fusion + overlap gain |
| --- | ---: | ---: | ---: |
| 1K / 1 | 51.79 | 1.13% | 5.23% |
| 1K / 8 | 62.29 | 2.84% | 3.64% |
| 4K / 1 | 54.27 | 1.08% | 4.98% |
| 4K / 8 | 73.51 | 0.34% | 3.08% |
| 8K / 1 | 54.40 | 1.05% | 4.92% |
| 8K / 8 | 77.72 | 1.35% | 2.98% |

Both candidates failed the gate because of TTFT regressions. For fusion with
overlap, examples are 1K/C1/output256: 487.88 to 510.02 ms (+4.54%), and
8K/C1/output1: 1909.25 to 2007.00 ms (+5.12%). TTFT includes scheduling and
chunked prefill; legacy before/after TTFT drifted 2.49%-9.31%, and some 1K/C8
cells had more than 10% round spread. The 8K/C8 footprint exceeds KV capacity,
so it includes queueing. These are configuration-specific experimental results.

Legacy prefill already overlaps main KV with Q. Fusion moves indexer K before
the main gather and moves indexer Q before the O-proj weight gather; the changed
dependencies may reduce existing overlap. Disabling the new overlap also waits
before main Q, so this is not a strict fusion-only ablation for prefill.
Device-side attribution remains pending. Prefill-legacy/decode-fused dispatch
is a proposed follow-up and is not implemented here.

The hardware implementation was based on Ascend commit
`9f9da5562089280a7f01c3d03e855b258e372807` and vLLM commit
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`. This fork PR packages the patch
on its preserved local development baseline
`2777efea7b63cb3de375fabf8fe9cac41e779f67`; results do not establish
performance on later main-branch refactors.

## Unit tests

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q \
  tests/ut/attention/test_dsa_cp_comm.py \
  tests/ut/attention/test_indexer.py tests/ut/attention/test_sfa_cp.py \
  tests/ut/attention/test_sfa_v1.py tests/ut/test_ascend_config.py \
  tests/ut/ops/test_mla.py tests/ut/models/test_glm5next_indexer_backend.py \
  benchmarks/tests/test_dsa_cp_e2e_analysis.py
```

Transport tests cover empty and padded shapes, noncontiguous inputs, native
and mixed dtypes, delayed waits and workspace reuse. Integration tests cover
staged wrapper forwarding, skipped-top-k cache updates and layout fallbacks.
