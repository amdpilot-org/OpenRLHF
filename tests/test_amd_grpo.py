# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Integration tests for the AMD FSDP-AMD GRPO path (Stage 1).

Verifies that:
  * FSDPStrategy / MXFP4 loader / AMDGRPOTrainer are importable,
  * a GRPO step executes end-to-end on ROCm,
  * reward strictly increases over 2 steps (the GRPO update is learning),
  * a ``grpo_step_time`` metric is produced.

Run:  pytest tests/test_amd_grpo.py -v
  or:  python tests/test_amd_grpo.py
"""

import os
import sys

try:
    import pytest
except ImportError:  # standalone runner without pytest installed
    class _Mark:
        def skipif(self, *a, **k):
            def deco(f):
                return f
            return deco
        def skip(self, *a, **k):
            def deco(f):
                return f
            return deco
    class pytest:  # type: ignore
        mark = _Mark()

# Single-process distributed env for the smoke test.
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29514")

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_fsdp_strategy_importable():
    from openrlhf.utils.fsdp import FSDPStrategy

    s = FSDPStrategy(seed=0, max_norm=1.0, micro_train_batch_size=1, train_batch_size=8)
    assert s.stage == 3
    assert s.param_dtype == "bf16"
    assert s.is_rank_0()


def test_mxfp4_loader_importable():
    from openrlhf.utils.mxfp4_loader import has_mxfp4_support, load_mxfp4_model

    # has_mxfp4_support must be callable and return a bool (False here, aiter absent).
    assert isinstance(has_mxfp4_support(), bool)
    assert callable(load_mxfp4_model)


def test_amd_grpo_trainer_importable():
    from openrlhf.trainer.amd_grpo_trainer import (
        AMDGRPOTrainer,
        GRPOConfig,
        group_norm_advantages,
        run_minimal_grpo_step,
    )

    # group_norm: zero variance -> zero advantage (no NaN).
    r = torch.tensor([1.0, 1.0, 1.0, 1.0])
    adv = group_norm_advantages(r, group_size=4)
    assert torch.allclose(adv, torch.zeros_like(adv))

    # group_norm: known values.
    r = torch.tensor([0.0, 2.0])
    adv = group_norm_advantages(r, group_size=2)
    assert adv[1] > 0 and adv[0] < 0
    assert callable(run_minimal_grpo_step)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires ROCm GPU")
def test_grpo_reward_strictly_increases():
    """The defining Stage-1 assertion: reward goes up over 2 GRPO steps."""
    from openrlhf.trainer.amd_grpo_trainer import run_minimal_grpo_step

    metrics, avg_step_time = run_minimal_grpo_step(
        num_steps=2,
        group_size=16,
        num_prompts=4,
        response_length=16,
        prompt_length=4,
        lr=5e-3,
        vocab_size=64,
        verbose=False,
    )

    assert len(metrics) == 2
    # A real step time was produced.
    assert avg_step_time > 0
    for m in metrics:
        assert "step_time" in m and m["step_time"] > 0
        assert "reward_mean" in m

    # Reward strictly increases (GRPO is learning).
    r0, r1 = metrics[0]["reward_mean"], metrics[1]["reward_mean"]
    assert r1 > r0, f"reward did not increase: {r0} -> {r1}"

    print(f"\ngrpo_step_time: {avg_step_time:.6f}")
    print(f"reward trajectory: {r0:.4f} -> {r1:.4f}")


if __name__ == "__main__":
    # Standalone runner (no pytest required).
    test_fsdp_strategy_importable()
    print("PASS test_fsdp_strategy_importable")
    test_mxfp4_loader_importable()
    print("PASS test_mxfp4_loader_importable")
    test_amd_grpo_trainer_importable()
    print("PASS test_amd_grpo_trainer_importable")
    if torch.cuda.is_available():
        test_grpo_reward_strictly_increases()
        print("PASS test_grpo_reward_strictly_increases")
    else:
        print("SKIP test_grpo_reward_strictly_increases (no GPU)")
    print("\nAll tests passed.")
