# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reusable buffers for native DSA-CP KV/indexer communication.

Buffers belong to one attention layer, and are keyed by the padded graph
shape. The model runner must serialize forwards for that layer, as it does
for cache writes. No mutable state is stored in shared attention metadata.
"""

from collections.abc import Callable
from dataclasses import dataclass
from math import prod

import torch


@dataclass
class GatheredFields:
    tensors: tuple[torch.Tensor, ...]
    handles: tuple
    # Keep input storage alive until the collective has finished.
    inputs: tuple[torch.Tensor, ...]

    def wait(self) -> tuple[torch.Tensor, ...]:
        for handle in self.handles:
            if handle is not None:
                handle.wait()
        return self.tensors


class DSACPCommBuffers:
    def __init__(self):
        self.buffers: dict[tuple, tuple[torch.Tensor | None, torch.Tensor]] = {}

    def start(self, fields: tuple[torch.Tensor, ...], group, launch: Callable, fused: bool) -> GatheredFields:
        if not fields or any(x.ndim < 1 for x in fields):
            raise ValueError("DSA-CP fields require a token dimension")
        rows, device = fields[0].shape[0], fields[0].device
        if any(x.shape[0] != rows or x.device != device for x in fields):
            raise ValueError("DSA-CP fields must have matching token counts and devices")
        # Dtypes, field shapes and rank count are static for each graph bucket.
        capacity = 1 << max(rows - 1, 0).bit_length()
        key = (fused, group.world_size, capacity, tuple((tuple(x.shape[1:]), x.dtype, x.device) for x in fields))
        homogeneous = all(x.dtype == fields[0].dtype for x in fields)
        dtype = fields[0].dtype if homogeneous else torch.uint8
        unit = fields[0].element_size() if homogeneous else 1
        alignment = max(x.element_size() for x in fields) // unit
        offsets, widths = [], []
        width = 0
        for x in fields:
            width = (width + alignment - 1) // alignment * alignment
            offsets.append(width)
            field_width = prod(x.shape[1:]) * x.element_size() // unit
            widths.append(field_width)
            width += field_width
        width = (width + alignment - 1) // alignment * alignment
        if fused and key not in self.buffers:
            self.buffers[key] = (
                torch.empty((capacity, width), dtype=dtype, device=device),
                torch.empty((capacity * group.world_size, width), dtype=dtype, device=device),
            )
        if fused:
            packed, output = self.buffers[key]
            assert packed is not None
            if rows != capacity:
                packed, output = packed[:rows], output[: rows * group.world_size]
            sources = []
            for x in fields:
                source = x if x.ndim == 2 else x.reshape(rows, prod(x.shape[1:]))
                if not source.is_contiguous():
                    source = source.contiguous()
                if not homogeneous:
                    # Empty slices can report contiguous while retaining a
                    # non-unit final stride; dtype views still require one.
                    if source.stride(-1) != 1:
                        source = source.clone(memory_format=torch.contiguous_format)
                    source = source.view(torch.uint8)
                sources.append(source)
            if width == sum(widths):
                torch.cat(sources, dim=1, out=packed)
            else:
                # Clear alignment gaps so transport never reads uninitialized bytes.
                packed.zero_()
                for source, offset, count in zip(sources, offsets, widths):
                    packed[:, offset : offset + count].copy_(source)
            gathered, handle = launch(packed, group, output=output, async_op=True)
            tensors = []
            for x, offset, count in zip(fields, offsets, widths):
                view = gathered.narrow(1, offset, count)
                if not homogeneous:
                    view = view.view(x.dtype)
                if x.ndim != 2:
                    view = view.reshape(rows * group.world_size, *x.shape[1:])
                tensors.append(view)
            return GatheredFields(tuple(tensors), (handle,), (packed,))

        # The unfused ablation reuses existing contiguous inputs, matching the
        # legacy collective. Only outputs require persistent bucket storage.
        tensors, handles, inputs = [], [], []
        for index, x in enumerate(fields):
            field_key = (key, index)
            if field_key not in self.buffers:
                self.buffers[field_key] = (
                    None,
                    torch.empty((capacity * group.world_size, *x.shape[1:]), dtype=x.dtype, device=device),
                )
            _, target = self.buffers[field_key]
            if rows != capacity:
                target = target[: rows * group.world_size]
            source = x if x.is_contiguous() else x.contiguous()
            gathered, handle = launch(source, group, output=target, async_op=True)
            tensors.append(gathered)
            handles.append(handle)
            inputs.append(source)
        return GatheredFields(tuple(tensors), tuple(handles), tuple(inputs))


@dataclass
class IndexerKVComm:
    prepared_k: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]
    gathered: GatheredFields
    num_main_fields: int
    indexer_cache: tuple[torch.Tensor, torch.Tensor | None] | None = None
    overlap: bool = True

    @classmethod
    def start(cls, main_parts, prepared_k, group, mode, buffers, launch, overlap=True):
        k_li, scale, _ = prepared_k
        # Main KV fields already have their native storage dtype. Keep every
        # field's bits, including optional int8 K and float32 scales.
        main = tuple(x if x.ndim == 2 else x.reshape(x.shape[0], prod(x.shape[1:])) for x in main_parts)
        # Preserve the original one-collective main KV pack in async mode.
        if mode == "async":
            main = (torch.cat(main, dim=-1),)
        fields = main + (k_li,)
        if scale is not None:
            fields += (scale,)
        gathered = buffers.start(fields, group, launch, fused=mode == "fused")
        return cls(prepared_k, gathered, len(main), overlap=overlap)

    def finish(self):
        fields = self.gathered.wait()
        k_li = fields[self.num_main_fields]
        scale = fields[self.num_main_fields + 1] if self.prepared_k[1] is not None else None
        self.indexer_cache = (k_li, scale)
        # The existing main-cache writer consumes a concatenated main KV row.
        if self.num_main_fields == 1:
            return fields[0]
        first = fields[0]
        width = first.shape[1]
        for field in fields[1 : self.num_main_fields]:
            if field.dtype != first.dtype or field.storage_offset() != first.storage_offset() + width:
                return torch.cat(fields[: self.num_main_fields], dim=-1)
            width += field.shape[1]
        # The main fields are adjacent in the fused row. Preserve its row
        # stride for the existing cache writer instead of copying a new KV
        # tensor merely to split it back into the same fields.
        return first.as_strided((first.shape[0], width), first.stride())
