# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.attention.dsa_cp_comm import DSACPCommBuffers, IndexerKVComm


class DelayedGather:
    """Delay writes until wait, and emulate rank-major HCCL output."""

    def __init__(self):
        self.calls = []
        self.waits = 0

    def __call__(self, source, group, output, async_op):
        assert async_op
        self.calls.append(source)

        def wait():
            self.waits += 1
            output.copy_(torch.cat([source] * group.world_size, dim=0))

        return output, SimpleNamespace(wait=wait)


@pytest.mark.parametrize("rows", [0, 1, 3, 16])
@pytest.mark.parametrize("world_size", [1, 2, 16])
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("mode", ["async", "fused"])
def test_roundtrip_and_wait(rows, world_size, quantized, mode):
    main = (torch.arange(rows * 12, dtype=torch.bfloat16).reshape(rows, 12)[:, ::2],)
    k = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 1, 3)
    k = k.to(torch.int8 if quantized else torch.bfloat16)
    scale = torch.full((rows,), 0.125) if quantized else None
    launch, buffers = DelayedGather(), DSACPCommBuffers()
    pending = IndexerKVComm.start(
        main, (k, scale, torch.empty(rows, 2)), SimpleNamespace(world_size=world_size), mode, buffers, launch
    )
    assert launch.waits == 0
    assert len(launch.calls) == (1 if mode == "fused" else 2 + int(quantized))
    torch.testing.assert_close(pending.finish(), torch.cat(list(main) * world_size))
    torch.testing.assert_close(pending.indexer_cache[0], torch.cat([k] * world_size))
    if scale is not None:
        torch.testing.assert_close(pending.indexer_cache[1], torch.cat([scale] * world_size))
    else:
        assert pending.indexer_cache[1] is None


@pytest.mark.parametrize("fused", [False, True])
def test_bucket_buffers_are_stable_and_refresh_on_replay(fused):
    buffers, launch, group = DSACPCommBuffers(), DelayedGather(), SimpleNamespace(world_size=2)
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)[:, ::2]
    first = buffers.start((x,), group, launch, fused).wait()[0]
    ptr = first.data_ptr()
    x.add_(10)
    second = buffers.start((x,), group, launch, fused).wait()[0]
    assert second.data_ptr() == ptr
    torch.testing.assert_close(second, torch.cat([x, x]))
    adjacent_shape = buffers.start((torch.ones(4, 2),), group, launch, fused).wait()[0]
    assert adjacent_shape.data_ptr() == ptr
    assert len(buffers.buffers) == 1
    assert ptr == buffers.start((x,), group, launch, fused).wait()[0].data_ptr()


def test_mixed_dtype_roundtrip_is_bit_exact():
    floats = torch.tensor([[0x7FC12345, -2147483648]], dtype=torch.int32).view(torch.float32)
    fields = (torch.tensor([[1, -2, 3]], dtype=torch.int8), floats, torch.ones(1, 3, dtype=torch.bfloat16))
    pending = DSACPCommBuffers().start(fields, SimpleNamespace(world_size=2), DelayedGather(), True)
    for original, actual in zip(fields, pending.wait()):
        expected = torch.cat([original, original]).contiguous().view(torch.uint8)
        assert torch.equal(actual.contiguous().view(torch.uint8), expected)


@pytest.mark.parametrize("fields", [(), (torch.tensor(1),), (torch.zeros(2, 3), torch.zeros(3, 1))])
def test_invalid_layout_rejected_before_collective(fields):
    launch = DelayedGather()
    with pytest.raises(ValueError):
        DSACPCommBuffers().start(fields, SimpleNamespace(world_size=2), launch, True)
    assert not launch.calls


@pytest.mark.parametrize("mode", ["async", "fused"])
def test_capture_keeps_asynchronous_submission(monkeypatch, mode):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: True), raising=False)
    launch = DelayedGather()
    k = torch.ones(2, 1, 4)
    pending = IndexerKVComm.start(
        (torch.zeros(2, 8),), (k, None, None), SimpleNamespace(world_size=2), mode, DSACPCommBuffers(), launch
    )
    assert pending.overlap
    assert launch.waits == 0
    pending.finish()
    torch.testing.assert_close(pending.indexer_cache[0], torch.cat([k, k]))


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("mixed_main", [False, True])
def test_main_kv_prefix_preserves_values_without_an_extra_copy(quantized, mixed_main):
    main = (
        torch.arange(12, dtype=torch.float32 if mixed_main else torch.bfloat16).reshape(3, 4),
        torch.arange(18, dtype=torch.bfloat16).reshape(3, 6),
    )
    k = torch.arange(9, dtype=torch.int8 if quantized else torch.bfloat16).reshape(3, 1, 3)
    scale = torch.ones(3, 1, dtype=torch.float32) if quantized else None
    pending = IndexerKVComm.start(
        main, (k, scale, None), SimpleNamespace(world_size=2), "fused", DSACPCommBuffers(), DelayedGather()
    )
    result = pending.finish()
    expected = torch.cat([torch.cat(main, dim=1)] * 2)
    torch.testing.assert_close(result, expected)
    if not mixed_main:
        assert result.untyped_storage().data_ptr() == pending.gathered.tensors[0].untyped_storage().data_ptr()
        assert result.stride(0) > result.shape[1]
    torch.testing.assert_close(pending.indexer_cache[0], torch.cat([k, k]))


def test_separate_gathers_do_not_copy_contiguous_inputs():
    fields = (torch.ones(2, 4), torch.ones(2, 1, 3))
    launch = DelayedGather()
    pending = DSACPCommBuffers().start(fields, SimpleNamespace(world_size=2), launch, fused=False)
    assert all(actual is original for actual, original in zip(launch.calls, fields))
    for actual, original in zip(pending.wait(), fields):
        torch.testing.assert_close(actual, torch.cat([original, original]))
