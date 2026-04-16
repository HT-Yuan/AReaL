# SPDX-License-Identifier: Apache-2.0

"""Routing Replay (R3) for MoE models.

Implements the R3 technique: during RL training, force the MoE router to use
the same expert routing decisions recorded during rollout (inference) instead
of recomputing them. This ensures consistency between rollout and training
logprobs, which is critical for PPO/GRPO importance sampling ratios.

Reference: Slime's routing_replay.py (arXiv:2510.11370)

Usage:
    1. During rollout, SGLang returns `routed_experts` per token per layer.
    2. Before training forward, call `routing_replay.load(routed_experts)` to
       populate the replay buffer.
    3. The patched router's `forward()` will pop indices from the buffer
       instead of computing topk.
    4. After backward, call `routing_replay.clear()` to reset.
"""

from __future__ import annotations

import torch

from areal.utils import logging

logger = logging.getLogger("RoutingReplay")


class RoutingReplay:
    """Per-layer routing replay buffer.

    Stores pre-recorded expert routing indices and serves them sequentially
    during forward and backward passes (since AC may re-run forward during
    backward, we maintain separate forward/backward cursors).
    """

    def __init__(self) -> None:
        self._layers: list[torch.Tensor] = []  # per-layer routing indices
        self._forward_idx: int = 0
        self._backward_idx: int = 0

    def record(self, top_indices: torch.Tensor) -> None:
        """Record routing indices for one layer.

        Args:
            top_indices: Expert indices, shape (num_tokens, top_k).
                         Will be copied to CPU pinned memory.
        """
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        buf.copy_(top_indices)
        self._layers.append(buf)

    def pop_forward(self, device: torch.device | str = "cuda") -> torch.Tensor:
        """Pop the next layer's routing indices for forward pass."""
        indices = self._layers[self._forward_idx].to(device, non_blocking=True)
        self._forward_idx += 1
        return indices

    def pop_backward(self, device: torch.device | str = "cuda") -> torch.Tensor:
        """Pop the next layer's routing indices for backward pass."""
        indices = self._layers[self._backward_idx].to(device, non_blocking=True)
        self._backward_idx += 1
        return indices

    def clear(self) -> None:
        """Reset all state."""
        self._layers.clear()
        self._forward_idx = 0
        self._backward_idx = 0

    def reset_cursors(self) -> None:
        """Reset cursors without clearing data (for re-forward in AC)."""
        self._forward_idx = 0
        self._backward_idx = 0

    @property
    def num_layers(self) -> int:
        return len(self._layers)

    @property
    def is_loaded(self) -> bool:
        return len(self._layers) > 0


# Module-level singleton — set by ArchonEngine before forward pass
_active_replay: RoutingReplay | None = None
_replay_enabled: bool = False
_replay_stage: str = "off"  # "off", "replay_forward", "replay_backward"


def set_replay_state(
    replay: RoutingReplay | None,
    enabled: bool = False,
    stage: str = "off",
) -> None:
    """Set the global routing replay state.

    Args:
        replay: The RoutingReplay instance, or None to disable.
        enabled: Whether replay is active.
        stage: One of "off", "replay_forward", "replay_backward".
    """
    global _active_replay, _replay_enabled, _replay_stage
    _active_replay = replay
    _replay_enabled = enabled
    _replay_stage = stage


def get_replay_state() -> tuple[RoutingReplay | None, bool, str]:
    """Get the current global replay state."""
    return _active_replay, _replay_enabled, _replay_stage


def load_routed_experts_into_replay(
    replay: RoutingReplay,
    routed_experts: torch.Tensor,
    num_moe_layers: int,
    top_k: int,
) -> None:
    """Load rollout routing decisions into a RoutingReplay buffer.

    Args:
        replay: Target RoutingReplay instance.
        routed_experts: Flat array from SGLang, shape (num_tokens, num_moe_layers * top_k).
            Each row contains [layer0_expert0, layer0_expert1, ..., layerN_expertK].
        num_moe_layers: Number of MoE layers in the model.
        top_k: Number of experts per token.
    """
    replay.clear()
    num_tokens = routed_experts.shape[0]
    expected_cols = num_moe_layers * top_k
    assert routed_experts.shape[1] == expected_cols, (
        f"routed_experts has {routed_experts.shape[1]} columns, "
        f"expected {expected_cols} (num_moe_layers={num_moe_layers}, top_k={top_k})"
    )

    # Reshape to (num_tokens, num_moe_layers, top_k) then split by layer
    reshaped = routed_experts.view(num_tokens, num_moe_layers, top_k)
    for layer_idx in range(num_moe_layers):
        layer_indices = reshaped[:, layer_idx, :].contiguous()  # (num_tokens, top_k)
        replay.record(layer_indices)


__all__ = [
    "RoutingReplay",
    "set_replay_state",
    "get_replay_state",
    "load_routed_experts_into_replay",
]
