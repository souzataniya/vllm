# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.multimodal.video_prune.evs import recompute_mrope_positions

TEXT, TS, VISION_START, VISION_END, VIDEO, IMAGE = 1, 2, 777, 778, 888, 999
NUM_TIMESTAMP_TOKENS = 6
NUM_FRAMES, NUM_KEPT_TOKENS = 4, 10


def _frame_block(num_video_tokens: int) -> list[int]:
    """One Qwen3-VL video frame: timestamp tokens, vision_start, video
    tokens, vision_end."""
    return (
        [TS] * NUM_TIMESTAMP_TOKENS
        + [VISION_START]
        + [VIDEO] * num_video_tokens
        + [VISION_END]
    )


def _make_mm_pos() -> torch.Tensor:
    # Model-side layout: EVS keeps all retained tokens on frame 0; the model
    # still spreads positions across all frames.
    model_side = sum((_frame_block(n) for n in (4, 2, 2, 2)), [])
    ids = torch.tensor(model_side)
    mm_pos = torch.zeros(5, len(model_side), dtype=torch.long)
    i = torch.arange(len(model_side))
    mm_pos[0], mm_pos[1], mm_pos[2] = 2 * i, 3 * i + 1, 5 * i + 2
    mm_pos[3] = ids.eq(VISION_START).long()
    mm_pos[4] = ids.eq(VIDEO).long()
    return mm_pos


@pytest.mark.parametrize("prefix_len", [3, 20])
@pytest.mark.parametrize("keep_outer_vision_markers", [True, False])
def test_recompute_mrope_positions_with_timestamp_tokens(
    prefix_len: int, keep_outer_vision_markers: bool
):
    """Regression test for a transformers >= 5.10 Qwen3-VL prompt layout
    where an outer <|vision_start|>/<|vision_end|> pair wraps the whole
    video (in addition to the per-frame inner <|vision_start|>/
    <|vision_end|> pair that already exists around each frame's video
    tokens).

    Before the fix, `recompute_mrope_positions` located a new video block by
    taking the first vision_start token at/after the cursor. With the outer
    marker present, that is the *outer* vision_start -- which precedes the
    timestamp tokens -- instead of the inner one that immediately precedes
    the video tokens. Subtracting `num_timestamp_tokens` from the outer
    marker's index then placed the block `num_timestamp_tokens + 1` tokens
    too early, which either raised a RuntimeError (short prefix) or silently
    overwrote the positions of preceding text tokens (longer prefix).
    """
    placeholder = _frame_block(NUM_KEPT_TOKENS) + sum(
        (_frame_block(0) for _ in range(NUM_FRAMES - 1)), []
    )
    mm_pos = _make_mm_pos()

    prompt = (
        [TEXT] * prefix_len
        + ([VISION_START] if keep_outer_vision_markers else [])
        + placeholder
        + ([VISION_END] if keep_outer_vision_markers else [])
        + [TEXT] * 3
    )
    video_start = prefix_len + int(keep_outer_vision_markers)

    positions = torch.arange(len(prompt)).repeat(3, 1)
    out, _ = recompute_mrope_positions(
        torch.tensor(prompt),
        [mm_pos],
        positions.clone(),
        0,
        VISION_START,
        IMAGE,
        VIDEO,
    )

    expected_video_positions = mm_pos[0:3] + positions[-1, video_start - 1] + 1
    actual_video_positions = out[:, video_start : video_start + mm_pos.shape[1]]
    assert torch.equal(actual_video_positions, expected_video_positions)

    # Positions for every token before the video block must be untouched.
    assert torch.equal(out[:, :video_start], positions[:, :video_start])
