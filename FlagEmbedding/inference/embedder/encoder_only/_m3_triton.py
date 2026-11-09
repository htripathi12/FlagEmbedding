"""Experimental CUDA kernel for corresponding-pair ColBERT MaxSim."""

import torch
import triton
import triton.language as tl


@triton.jit
def _maximum(left, right):
    return tl.maximum(left, right, propagate_nan=tl.PropagateNan.ALL)


@triton.jit(do_not_specialize=["query_length", "passage_length"])
def _colbert_token_maxima_kernel(
    queries, passages, maxima,
    query_length, passage_length, width: tl.constexpr,
    query_tile: tl.constexpr, passage_tile: tl.constexpr, feature_tile: tl.constexpr,
):
    pair = tl.program_id(0)
    query_tokens = tl.program_id(1) * query_tile + tl.arange(0, query_tile)
    passage_offsets = tl.arange(0, passage_tile)
    feature_offsets = tl.arange(0, feature_tile)
    query_base = queries + pair.to(tl.int64) * query_length * width
    passage_base = passages + pair.to(tl.int64) * passage_length * width
    running_max = tl.full((query_tile,), float("-inf"), tl.float32)

    for passage_start in range(0, passage_length, passage_tile):
        passage_tokens = passage_start + passage_offsets
        similarities = tl.full((query_tile, passage_tile), 0, tl.float32)
        for feature_start in range(0, width, feature_tile):
            features = feature_start + feature_offsets
            query_block = tl.load(
                query_base + query_tokens[:, None] * width + features[None, :],
                (query_tokens[:, None] < query_length) & (features[None, :] < width),
                other=0,
            )
            passage_block = tl.load(
                passage_base + passage_tokens[None, :] * width + features[:, None],
                (passage_tokens[None, :] < passage_length) & (features[:, None] < width),
                other=0,
            )
            similarities = tl.dot(query_block, passage_block, similarities, input_precision="ieee")

        # Match bmm's output rounding and include the batch's zero-padded tokens.
        # Only lanes past the allocated passage length are excluded.
        similarities = similarities.to(maxima.dtype.element_ty).to(tl.float32)
        similarities = tl.where(passage_tokens[None, :] < passage_length, similarities, float("-inf"))
        tile_max = tl.reduce(similarities, axis=1, combine_fn=_maximum)
        running_max = _maximum(running_max, tile_max)

    tl.store(
        maxima + pair.to(tl.int64) * query_length + query_tokens,
        running_max,
        query_tokens < query_length,
    )


def colbert_token_maxima(queries: torch.Tensor, passages: torch.Tensor) -> torch.Tensor:
    """Return one passage-token maximum per query token for matched CUDA pairs."""
    batch_size, query_length, width = queries.shape
    passage_length = passages.size(1)
    maxima = queries.new_empty((batch_size, query_length))
    query_tile = 16
    with torch.cuda.device(queries.device):
        _colbert_token_maxima_kernel[(batch_size, triton.cdiv(query_length, query_tile))](
            queries.contiguous(), passages.contiguous(), maxima,
            query_length, passage_length, width,
            query_tile=query_tile, passage_tile=32, feature_tile=32,
            num_warps=4,
        )
    return maxima
