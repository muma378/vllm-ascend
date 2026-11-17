# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run with torchrun --nproc-per-node=16; emits rank-max latency as JSONL.

Checks real HCCL rank ordering, mixed dtype transport, cache scatter and
ACL graph capture/replay. This is a communication microbenchmark, not an
end-to-end GLM performance claim.
"""

import argparse
import json
import os
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch_npu

from vllm_ascend.attention.dsa_cp_comm import DSACPCommBuffers, IndexerKVComm


def launch(source, group, output=None, async_op=True):
    if output is None:
        output = torch.empty(
            (source.shape[0] * group.world_size, *source.shape[1:]), dtype=source.dtype, device=source.device
        )
    work = dist.all_gather_into_tensor(output, source, async_op=async_op)
    return output, work


def run_case(rows, quantized, variant, graph_mode, args, group, rank, keepalive):
    buffers = DSACPCommBuffers()
    main = (
        torch.full((rows, 64), rank / 16, dtype=torch.bfloat16, device="npu"),
        torch.full((rows, 512), rank / 16 + 1, dtype=torch.bfloat16, device="npu"),
    )
    k = torch.full((rows, 1, 128), rank, dtype=torch.int8 if quantized else torch.bfloat16, device="npu")
    scale = torch.full((rows, 1, 1), rank + 0.5, dtype=torch.float32, device="npu") if quantized else None
    prepared = (k, scale, torch.empty(rows, 32, dtype=torch.bfloat16, device="npu"))
    # Independent local GEMM represents Q work that can overlap communication.
    q_input = torch.ones(rows, 256, dtype=torch.bfloat16, device="npu")
    q_weight = torch.ones(256, 256, dtype=torch.bfloat16, device="npu")
    cache = torch.zeros(rows * group.world_size, 128, dtype=k.dtype, device="npu")
    slots = torch.arange(rows * group.world_size, dtype=torch.int32, device="npu").view(-1, 1)

    def step():
        if variant == "legacy":
            main_out, work = launch(torch.cat(main, dim=-1), group, async_op=False)
            q = q_input @ q_weight
            k_out, kh = launch(k, group)
            if scale is not None:
                scale_out, sh = launch(scale, group)
            kh.wait()
            if scale is not None:
                sh.wait()
        else:
            mode = "async" if variant == "async" else "fused"
            pending = IndexerKVComm.start(main, prepared, group, mode, buffers, launch)
            if variant != "fused_only":
                q = q_input @ q_weight
            main_out = pending.finish()
            if variant == "fused_only":
                q = q_input @ q_weight
            k_out, scale_out = pending.indexer_cache
        torch_npu.npu_scatter_nd_update_(cache, slots, k_out.view(-1, 128))
        return main_out, cache, scale_out if scale is not None else None, q

    for _ in range(args.warmup):
        result = step()
    torch.npu.synchronize()
    graph = None
    if graph_mode:
        dist.barrier()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = step()
        dist.barrier()
        torch.npu.synchronize()
        graph.replay()
    torch.npu.synchronize()
    if graph is not None:
        # Match a serving runner: retain every captured graph and its input
        # storage until all graph buckets have finished using the communicator.
        keepalive.append((graph, result, main, prepared, q_input, q_weight, cache, slots, buffers))
    for tensor, expected in (
        (result[0][:, :64], torch.arange(group.world_size).repeat_interleave(rows)[:, None] / 16),
        (result[0][:, 64:], torch.arange(group.world_size).repeat_interleave(rows)[:, None] / 16 + 1),
        (result[1], torch.arange(group.world_size).repeat_interleave(rows)[:, None]),
    ):
        torch.testing.assert_close(tensor.cpu().float(), expected.expand(tensor.shape).float(), rtol=0, atol=0)
    if scale is not None:
        expected = torch.arange(group.world_size).repeat_interleave(rows).view(-1, 1, 1) + 0.5
        torch.testing.assert_close(result[2].cpu(), expected, rtol=0, atol=0)
    # Change graph inputs and verify replay reads current data and updates cache.
    if graph is not None:
        k.add_(1)
        dist.barrier()
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        expected = torch.arange(group.world_size).repeat_interleave(rows)[:, None] + 1
        torch.testing.assert_close(result[1].cpu().float(), expected.expand(result[1].shape).float(), rtol=0, atol=0)
    dist.barrier()
    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    begin.record()
    for _ in range(args.iterations):
        if graph is None:
            step()
        else:
            graph.replay()
    end.record()
    torch.npu.synchronize()
    elapsed = torch.tensor(begin.elapsed_time(end) / args.iterations, device="npu")
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(
            json.dumps(
                dict(
                    rows_per_rank=rows,
                    quantized=quantized,
                    variant=variant,
                    graph=graph_mode,
                    rank_max_ms=elapsed.item(),
                )
            ),
            flush=True,
        )
    dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 128])
    parser.add_argument("--variants", nargs="+", default=["legacy", "async", "fused_only", "fused"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    group, rank = SimpleNamespace(world_size=dist.get_world_size()), dist.get_rank()
    keepalive = []
    try:
        for rows in args.rows:
            for quantized in (False, True):
                for graph_mode in (False, True):
                    for variant in args.variants:
                        run_case(rows, quantized, variant, graph_mode, args, group, rank, keepalive)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
