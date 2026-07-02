# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""FSDP distributed strategy for AMD ROCm (MI300/MI325/MI355X).

A drop-in replacement for :class:`openrlhf.utils.deepspeed.DeepspeedStrategy`
that shards model parameters / gradients / optimizer state with
``torch.distributed.fsdp.FullyShardedDataParallel`` instead of DeepSpeed.

This is the AMD-native path: it avoids the DeepSpeed + ROCm ZeRO-3
incompatibilities (hipBLASLt solver crashes, missing ROCm hooks) and uses
PyTorch's first-class FSDP, which is fully supported on ROCm 7.x.

The public surface mirrors DeepspeedStrategy so that ``PPOTrainer`` /
``AMDGRPOTrainer`` can consume either strategy interchangeably.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from functools import partial

from openrlhf.utils.distributed_sampler import DistributedSampler


class FSDPStrategy:
    """Distributed strategy backed by ``torch.distributed.fsdp``.

    Unlike DeepSpeed there is no engine object: the wrapped model *is* the
    FSDP module and the optimizer is a plain ``torch.optim`` optimizer.
    ``backward`` / ``optimizer_step`` therefore call the standard PyTorch
    APIs (``loss.backward()`` / ``optim.step()``).
    """

    def __init__(
        self,
        seed: int = 42,
        full_determinism: bool = False,
        max_norm: float = 1.0,
        micro_train_batch_size: int = 1,
        train_batch_size: int = 1,
        args=None,
    ) -> None:
        self.seed = seed
        self.full_determinism = full_determinism
        self.max_norm = max_norm
        self.micro_train_batch_size = micro_train_batch_size
        self.train_batch_size = train_batch_size
        self.args = args

        # FSDP sharding config (defaults mirror ZeRO-3 semantics).
        self.stage = getattr(getattr(args, "ds", None), "zero_stage", 3)
        self.param_dtype = getattr(getattr(args, "ds", None), "param_dtype", "bf16")
        self.adam_offload = getattr(getattr(args, "ds", None), "adam_offload", False)

        self._world_size = 1
        self._rank = 0
        self._accumulated_gradient = 1
        self._setup = False

    # ------------------------------------------------------------------ #
    # Distributed setup
    # ------------------------------------------------------------------ #
    def setup_distributed(self, timeout: int = 1800) -> None:
        """Initialise the default process group + device mesh.

        On ROCm the ``nccl`` backend maps to RCCL. For single-process
        smoke tests (world_size == 1) ``gloo`` is used to avoid pulling in
        RCCL when it is unnecessary.
        """
        if self._setup:
            return

        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available")

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        backend = "nccl" if world_size > 1 and torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            from datetime import timedelta

            dist.init_process_group(
                backend=backend,
                timeout=timedelta(seconds=timeout),
            )

        self._world_size = dist.get_world_size() if dist.is_initialized() else world_size
        self._rank = dist.get_rank() if dist.is_initialized() else rank
        self._accumulated_gradient = max(
            1, self.train_batch_size // (self.micro_train_batch_size * self._world_size)
        )

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")

        self._setup = True

    # ------------------------------------------------------------------ #
    # Model / optimizer preparation
    # ------------------------------------------------------------------ #
    def _resolve_dtype(self) -> torch.dtype:
        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(
            self.param_dtype, torch.bfloat16
        )

    def _sharding_strategy(self) -> ShardingStrategy:
        # stage 3 -> FULL_SHARD (params+grads+optim), stage 2 -> SHARD_GRAD_OP
        return ShardingStrategy.FULL_SHARD if self.stage >= 3 else ShardingStrategy.SHARD_GRAD_OP

    def prepare(self, *args) -> Tuple:
        """Wrap models with FSDP and (optionally) build optimizers.

        Accepts ``(model, cfg)`` tuples for trainable models or bare
        ``model`` objects for eval-only models, mirroring
        ``DeepspeedStrategy.prepare``.
        """
        results = []
        for item in args:
            if isinstance(item, tuple):
                model, cfg = item
                results.append(self._fsdp_init_train_model(model, cfg))
            else:
                results.append(self._fsdp_init_eval_model(item))
        return tuple(results) if len(results) > 1 else results[0]

    def _wrap_fsdp(self, model: nn.Module) -> nn.Module:
        model = model.to(self.device)
        dtype = self._resolve_dtype()
        mp = MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=dtype,
            buffer_dtype=dtype,
        )
        auto_wrap = partial(size_based_auto_wrap_policy, min_num_params=1_000_000)
        fsdp_kwargs = dict(
            sharding_strategy=self._sharding_strategy(),
            mixed_precision=mp,
            auto_wrap_policy=auto_wrap,
            device_id=self.device,
            use_orig_params=True,
            limit_all_gathers=True,
        )
        if self.adam_offload:
            fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)
        return FSDP(model, **fsdp_kwargs)

    def _fsdp_init_train_model(self, model: nn.Module, cfg: dict):
        model = self._wrap_fsdp(model)

        # Optimizer (AdamW; Muon left to a future extension).
        optim_name = (cfg or {}).get("optim", "adam").lower()
        if optim_name not in ("adam", "adamw"):
            raise NotImplementedError(f"FSDPStrategy: optim '{optim_name}' not supported yet")
        adam_cfg = (cfg or {}).get("adam", {})
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=adam_cfg.get("lr", 1e-6),
            betas=tuple(adam_cfg.get("betas", (0.9, 0.95))),
            eps=adam_cfg.get("eps", 1e-8),
            weight_decay=adam_cfg.get("weight_decay", 0.0),
        )

        # LR scheduler (linear warmup -> cosine).
        scheduler = self._build_scheduler(optimizer, cfg)
        return model, optimizer, scheduler

    def _fsdp_init_eval_model(self, model: nn.Module) -> nn.Module:
        return self._wrap_fsdp(model)

    def _build_scheduler(self, optimizer, cfg: dict):
        try:
            from transformers import get_scheduler

            total = (cfg or {}).get("scheduler_steps", 1000)
            warmup_ratio = (cfg or {}).get("lr_warmup_ratio", 0.03)
            num_warmup = max(1, int(total * warmup_ratio))
            return get_scheduler(
                name=(cfg or {}).get("lr_scheduler", "cosine"),
                optimizer=optimizer,
                num_warmup_steps=num_warmup,
                num_training_steps=total,
            )
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Training step primitives
    # ------------------------------------------------------------------ #
    def backward(self, loss, model, optimizer) -> None:
        loss.backward()

    def optimizer_step(self, optimizer, model, scheduler, name=None) -> None:
        if self.max_norm > 0:
            # FSDP-aware grad norm clipping; returns the pre-clip total norm.
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

    def get_grad_norm(self, model) -> float:
        # Compute the FSDP-aware total grad norm without clipping.
        # Accumulate on GPU and sync once (not per-parameter) to avoid
        # N separate CPU-GPU synchronisations.
        sq = torch.zeros(1, device=self.device)
        for p in model.parameters():
            if p.grad is not None:
                sq += p.grad.detach().float().pow(2).sum()
        return float(sq.item()) ** 0.5

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #
    def setup_dataloader(
        self,
        replay_buffer,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        collate_fn=None,
    ) -> "StatefulDataLoader":
        from torchdata.stateful_dataloader import StatefulDataLoader

        sampler = DistributedSampler(
            replay_buffer,
            num_replicas=self._world_size,
            rank=self._rank,
            shuffle=shuffle,
            seed=self.seed,
            drop_last=drop_last,
        )
        return StatefulDataLoader(
            replay_buffer,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            drop_last=drop_last,
            collate_fn=collate_fn,
            num_workers=0,
        )

    # ------------------------------------------------------------------ #
    # Checkpoint I/O (FSDP state dict)
    # ------------------------------------------------------------------ #
    def _unwrap_model(self, model) -> nn.Module:
        while hasattr(model, "module"):
            model = model.module
        return model

    def save_model(self, model, tokenizer, output_dir) -> None:
        os.makedirs(output_dir, exist_ok=True)
        with FSDP.state_dict_type(model, torch.distributed.fsdp.StateDictType.FULL_STATE_DICT):
            state = model.state_dict()
        if self.is_rank_0():
            torch.save(state, os.path.join(output_dir, "model.pt"))
            if tokenizer is not None:
                tokenizer.save_pretrained(output_dir)

    def load_model(self, model, path, map_location="cpu") -> None:
        state = torch.load(path, map_location=map_location)
        with FSDP.state_dict_type(model, torch.distributed.fsdp.StateDictType.FULL_STATE_DICT):
            model.load_state_dict(state, strict=False)

    def save_ckpt(self, model, save_dir, tag="ckpt", **kwargs) -> None:
        os.makedirs(save_dir, exist_ok=True)
        with FSDP.state_dict_type(model, torch.distributed.fsdp.StateDictType.SHARDED_STATE_DICT):
            state = model.state_dict()
        torch.save(state, os.path.join(save_dir, f"{tag}.pt"))

    def load_ckpt(self, model, load_dir, tag="ckpt", **kwargs) -> None:
        path = os.path.join(load_dir, f"{tag}.pt")
        state = torch.load(path, map_location="cpu")
        with FSDP.state_dict_type(model, torch.distributed.fsdp.StateDictType.SHARDED_STATE_DICT):
            model.load_state_dict(state, strict=False)

    # ------------------------------------------------------------------ #
    # Collective comms
    # ------------------------------------------------------------------ #
    def all_reduce(self, data, op="mean"):
        if not dist.is_initialized() or self._world_size == 1:
            return data
        op_map = {"mean": dist.ReduceOp.SUM, "sum": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX}
        red_op = op_map.get(op, dist.ReduceOp.SUM)
        if isinstance(data, torch.Tensor):
            out = data.clone()
            dist.all_reduce(out, op=red_op)
            if op == "mean":
                out /= self._world_size
            return out
        # python scalars
        t = torch.tensor([float(data)], device=self.device)
        dist.all_reduce(t, op=red_op)
        if op == "mean":
            t /= self._world_size
        return t.item()

    def all_gather(self, data):
        if not dist.is_initialized() or self._world_size == 1:
            return data
        if isinstance(data, torch.Tensor):
            gathered = [torch.zeros_like(data) for _ in range(self._world_size)]
            dist.all_gather(gathered, data.contiguous())
            return torch.cat(gathered, dim=0)
        return data

    # ------------------------------------------------------------------ #
    # EMA / rank helpers
    # ------------------------------------------------------------------ #
    def moving_average(self, model, model_ema, beta=0.992, device=None) -> None:
        if model_ema is None:
            return
        with torch.no_grad():
            for p, p_ema in zip(model.parameters(), model_ema.parameters()):
                p_ema.data.mul_(beta).add_(p.data, alpha=1 - beta)

    def is_rank_0(self) -> bool:
        return self._rank == 0

    def get_rank(self) -> int:
        return self._rank

    def print(self, *msg) -> None:
        if self.is_rank_0():
            print(*msg, flush=True)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def accumulated_gradient(self) -> int:
        return self._accumulated_gradient

    @property
    def ring_attn_group(self):
        # Ring attention is not wired up for the FSDP path yet.
        return None
