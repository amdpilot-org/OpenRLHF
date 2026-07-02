# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""AMD GRPO trainer (FSDP + ROCm).

Group Relative Policy Optimization (GRPO) on AMD MI300/MI325/MI355X GPUs,
using ``torch.distributed.fsdp`` instead of DeepSpeed.

GRPO = PPO with **group-normalised advantages** and **no critic**:
for each prompt, sample a group of G responses, score them with a
rule-based verifier, then set

    advantage_i = (reward_i - mean(rewards)) / (std(rewards) + eps)

The policy is updated with the standard PPO clipped surrogate plus an
optional KL penalty against a frozen reference model.

This module is the AMD-native counterpart of ``PPOTrainer``: it consumes
an :class:`openrlhf.utils.fsdp.FSDPStrategy` and exposes a
``run_minimal_grpo_step`` entrypoint used by the Stage-1 harness and the
integration test.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------- #
# GRPO advantage estimator (group_norm)
# --------------------------------------------------------------------- #
def group_norm_advantages(
    rewards: torch.Tensor, group_size: int, eps: float = 1e-6
) -> torch.Tensor:
    """Normalise rewards within each prompt group of size ``group_size``.

    ``rewards`` has shape ``[num_groups * group_size]`` (flattened, group-major).
    Returns advantages of the same shape.
    """
    rewards = rewards.float()
    rewards = rewards.view(-1, group_size)
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    adv = (rewards - mean) / (std + eps)
    return adv.view(-1)


# --------------------------------------------------------------------- #
# Reward verifiers
# --------------------------------------------------------------------- #
def make_token_count_reward(target_token_id: int) -> Callable:
    """Reward = number of ``target_token_id`` occurrences in the response."""

    def _reward(responses: torch.Tensor, prompt_len: int) -> torch.Tensor:
        resp = responses[:, prompt_len:]
        return (resp == target_token_id).float().sum(dim=1)

    return _reward


# --------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------- #
@dataclass
class GRPOConfig:
    group_size: int = 8
    response_length: int = 8
    prompt_length: int = 4
    clip_ratio: float = 0.2
    kl_beta: float = 0.0
    temperature: float = 1.0
    eps: float = 1e-6
    seed: int = 42


class AMDGRPOTrainer:
    """GRPO trainer over an FSDP-wrapped policy + frozen reference model."""

    def __init__(
        self,
        strategy,
        policy: nn.Module,
        reference: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        config: Optional[GRPOConfig] = None,
    ) -> None:
        self.strategy = strategy
        self.policy = policy
        self.reference = reference
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or GRPOConfig()
        self._step = 0

    # ----------------------------------------------------------------- #
    # Generation
    # ----------------------------------------------------------------- #
    @torch.no_grad()
    def _sample_responses(
        self, prompt_ids: torch.Tensor, group_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample ``group_size`` responses per prompt.

        Returns
        -------
        full_ids: ``[P*G, prompt_len + resp_len]``
        old_logprobs: ``[P*G, resp_len]``  (log-prob of each sampled token)
        """
        cfg = self.config
        P = prompt_ids.size(0)
        G = group_size
        device = next(self.policy.parameters()).device

        # Expand prompts: [P, L] -> [P*G, L]
        prompts = prompt_ids.unsqueeze(1).expand(P, G, -1).reshape(P * G, -1).to(device)
        cur = prompts
        sampled = []
        token_logprobs = []

        for _ in range(cfg.response_length):
            out = self.policy(cur)
            logits = out.logits[:, -1, :] / max(cfg.temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            logp = torch.log_softmax(logits, dim=-1).gather(-1, next_token.unsqueeze(-1)).squeeze(-1)
            sampled.append(next_token)
            token_logprobs.append(logp)
            cur = torch.cat([cur, next_token.unsqueeze(-1)], dim=-1)

        responses = torch.stack(sampled, dim=1)  # [P*G, resp_len]
        old_logprobs = torch.stack(token_logprobs, dim=1)  # [P*G, resp_len]
        full_ids = cur  # [P*G, prompt_len + resp_len]
        return full_ids, old_logprobs

    # ----------------------------------------------------------------- #
    # Log-prob recomputation (teacher-forced)
    # ----------------------------------------------------------------- #
    def _response_logprobs(self, model: nn.Module, full_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
        """Recompute log-probs of the response tokens under ``model``."""
        out = model(full_ids)
        logits = out.logits[:, :-1, :]  # predict token t+1 from token t
        targets = full_ids[:, prompt_len:]  # response tokens
        # align: logits at positions [prompt_len-1 .. end-1] predict response tokens
        resp_logits = logits[:, prompt_len - 1 :, :]
        logp = torch.log_softmax(resp_logits, dim=-1)
        token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return token_logp  # [P*G, resp_len]

    # ----------------------------------------------------------------- #
    # One GRPO update
    # ----------------------------------------------------------------- #
    def grpo_step(
        self,
        prompt_ids: torch.Tensor,
        reward_fn: Callable,
    ) -> Dict[str, float]:
        """Run a single GRPO update and return metrics."""
        cfg = self.config
        G = cfg.group_size
        prompt_len = cfg.prompt_length

        # 1. Sample a group of responses with the current policy.
        full_ids, old_logprobs = self._sample_responses(prompt_ids, G)
        old_logprobs = old_logprobs.detach()

        # 2. Score with the verifier.
        with torch.no_grad():
            rewards = reward_fn(full_ids, prompt_len).float()  # [P*G]
            advantages = group_norm_advantages(rewards, G, cfg.eps)  # [P*G]

        # 3. Recompute log-probs under the (trainable) policy.
        new_logprobs = self._response_logprobs(self.policy, full_ids, prompt_len)  # [P*G, resp_len]
        # Per-token ratio; sum over response for the sequence-level ratio.
        log_ratio = new_logprobs - old_logprobs
        ratio = log_ratio.exp()

        # 4. PPO clipped surrogate loss (token-level mean).
        adv = advantages.unsqueeze(-1)  # [P*G, 1]
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        # 5. Optional KL penalty against the reference model.
        kl_loss = torch.tensor(0.0, device=policy_loss.device)
        if self.reference is not None and cfg.kl_beta > 0:
            with torch.no_grad():
                ref_logprobs = self._response_logprobs(self.reference, full_ids, prompt_len)
            # Schulman estimator: KL ≈ mean( exp(ref-new) - 1 - (ref-new) )
            log_diff = ref_logprobs - new_logprobs
            kl_loss = (log_diff.exp() - 1 - log_diff).mean()
            kl_loss = kl_loss.clamp(min=0.0)

        loss = policy_loss + cfg.kl_beta * kl_loss

        # 6. Backward + optimizer step (FSDP-aware).
        self.strategy.backward(loss, self.policy, self.optimizer)
        grad_norm = self.strategy.get_grad_norm(self.policy)
        self.strategy.optimizer_step(self.optimizer, self.policy, self.scheduler)

        self._step += 1
        return {
            "loss": float(loss.detach().item()),
            "policy_loss": float(policy_loss.detach().item()),
            "kl": float(kl_loss.item()),
            "grad_norm": float(grad_norm),
            "reward_mean": float(rewards.mean().item()),
            "reward_std": float(rewards.std().item()),
            "adv_mean": float(advantages.mean().item()),
            "step": self._step,
        }


# --------------------------------------------------------------------- #
# Minimal end-to-end GRPO step (benchmark / smoke-test entrypoint)
# --------------------------------------------------------------------- #
def _build_tiny_llama(vocab_size: int = 64, device: torch.device = torch.device("cuda")) -> nn.Module:
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(config)
    return model


def run_minimal_grpo_step(
    num_steps: int = 1,
    world_size: int = 1,
    group_size: int = 16,
    num_prompts: int = 4,
    response_length: int = 16,
    prompt_length: int = 4,
    lr: float = 5e-3,
    kl_beta: float = 0.0,
    seed: int = 42,
    use_fsdp: bool = True,
    verbose: bool = True,
    vocab_size: int = 64,
) -> Tuple[List[Dict[str, float]], float]:
    """Run ``num_steps`` GRPO updates on a tiny Llama and return per-step metrics.

    This is the Stage-1 "enable required path" entrypoint: it exercises the
    full FSDP-AMD GRPO code path (FSDPStrategy.setup_distributed → prepare →
    sample → group-norm advantage → PPO-clipped loss → backward → step) on a
    small model so the path is verifiably executable and produces a
    ``grpo_step_time`` metric.

    Returns
    -------
    metrics: list of per-step metric dicts
    avg_step_time: mean wall-clock seconds per step (excluding warmup)
    """
    # Single-process distributed init (gloo, world_size=1) for the smoke test.
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29512")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    from openrlhf.utils.fsdp import FSDPStrategy

    strategy = FSDPStrategy(
        seed=seed,
        max_norm=1.0,
        micro_train_batch_size=1,
        train_batch_size=num_prompts * group_size,
    )
    strategy.setup_distributed(timeout=300)
    device = strategy.device

    # Tiny policy + frozen reference.
    policy = _build_tiny_llama(vocab_size=vocab_size, device=device)
    reference = _build_tiny_llama(vocab_size=vocab_size, device=device)
    reference.load_state_dict(policy.state_dict())
    for p in reference.parameters():
        p.requires_grad_(False)
    reference.eval()

    if use_fsdp:
        policy, optimizer, scheduler = strategy.prepare(
            (policy, {"optim": "adam", "adam": {"lr": lr}, "lr_scheduler": "cosine",
                      "scheduler_steps": 100, "lr_warmup_ratio": 0.0})
        )
        reference = strategy.prepare(reference)
    else:
        policy = policy.to(device)
        reference = reference.to(device)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
        scheduler = None

    config = GRPOConfig(
        group_size=group_size,
        response_length=response_length,
        prompt_length=prompt_length,
        kl_beta=kl_beta,
        seed=seed,
    )
    trainer = AMDGRPOTrainer(strategy, policy, reference, optimizer, scheduler, config)

    # Random prompts + a token-count reward (target token = 1).
    target_token = 1
    reward_fn = make_token_count_reward(target_token)
    prompt_ids = torch.randint(2, vocab_size, (num_prompts, prompt_length))

    # Warmup (not timed): one forward to trigger FSDP unshard / kernel JIT.
    _ = trainer.grpo_step(prompt_ids, reward_fn)

    metrics: List[Dict[str, float]] = []
    step_times: List[float] = []
    for _ in range(num_steps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        m = trainer.grpo_step(prompt_ids, reward_fn)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        m["step_time"] = dt
        metrics.append(m)
        step_times.append(dt)
        if verbose:
            strategy.print(
                f"[GRPO step {m['step']}] loss={m['loss']:.4f} "
                f"reward_mean={m['reward_mean']:.4f} grad_norm={m['grad_norm']:.4f} "
                f"step_time={dt*1000:.2f}ms"
            )

    avg_step_time = sum(step_times) / max(len(step_times), 1)
    return metrics, avg_step_time


if __name__ == "__main__":
    metrics, avg = run_minimal_grpo_step(num_steps=2, verbose=True)
    print(f"\ngrpo_step_time: {avg:.6f}")
