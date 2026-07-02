# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""MXFP4 weight loading for AMD ROCm (MI355X / gfx950).

MXFP4 (Microscaling FP4, OCP MX format) stores weights in 4-bit floating
point with a per-block E8M0 scale. On MI355X the ``aiter`` library provides
hardware-accelerated MXFP4 GEMM via ``aiter.MXFP4Linear`` /
``aiter.quant_mxfp4``.

This module is the integration point between OpenRLHF's ``Actor`` and the
MXFP4 checkpoint format used by ``amd/Llama-3.3-70B-Instruct-MXFP4-Preview``.

When ``aiter`` is installed the loader swaps ``nn.Linear`` layers for
``aiter`` MXFP4 linear layers and loads the 4-bit weights directly. When
``aiter`` is *not* installed (e.g. a CI smoke test) the loader falls back to
loading the checkpoint in bf16 so that the rest of the FSDP-AMD GRPO path
remains executable.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _aiter_available() -> bool:
    try:
        import aiter  # noqa: F401

        return True
    except Exception:
        return False


def has_mxfp4_support() -> bool:
    """Return True iff the aiter MXFP4 backend is importable."""
    return _aiter_available()


def _replace_linear_with_mxfp4(model: nn.Module) -> nn.Module:
    """Swap every ``nn.Linear`` for an aiter MXFP4 linear layer in-place."""
    import aiter  # type: ignore

    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and not isinstance(module, getattr(aiter, "MXFP4Linear", type(None))):
            mxfp4 = aiter.MXFP4Linear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                device=module.weight.device,
            )
            setattr(model, name, mxfp4)
        else:
            _replace_linear_with_mxfp4(module)
    return model


def load_mxfp4_model(
    model_path: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[torch.device] = None,
    attn_implementation: str = "sdpa",
    torch_dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """Load a model, preferring the aiter MXFP4 path.

    Parameters
    ----------
    model_path:
        HuggingFace model id or local path. For MXFP4 checkpoints
        (``amd/Llama-3.3-70B-Instruct-MXFP4-Preview``) the weights are
        stored as packed int8 + E8M0 scales.
    dtype:
        Compute dtype for the non-MXFP4 tensors (norms, embeddings).
    device:
        Target device; defaults to the current CUDA device.

    Returns
    -------
    nn.Module
        A ``transformers`` causal LM with MXFP4 linears (aiter) or a
        plain bf16 model (fallback).
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    if torch_dtype is not None:
        dtype = torch_dtype
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    if _aiter_available():
        logger.info("aiter detected: loading %s with MXFP4 linears", model_path)
        # Load in bf16 first, then swap linears for MXFP4 and quantize.
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
        )
        model = _replace_linear_with_mxfp4(model)
        import aiter  # type: ignore

        for _, module in model.named_modules():
            if hasattr(module, "load_mxfp4_weights"):
                module.load_mxfp4_weights(model_path)
        return model.to(device)

    # ---- Fallback: no aiter ------------------------------------------- #
    logger.warning(
        "aiter is NOT installed; falling back to bf16 load of %s. "
        "MXFP4 hardware acceleration is disabled. Install aiter for the "
        "full MI355X MXFP4 path.",
        model_path,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
    )
    return model.to(device)


def quantize_to_mxfp4(model: nn.Module) -> nn.Module:
    """Quantize an existing bf16 model's linears to MXFP4 in-place (aiter).

    Used when starting from a bf16 checkpoint and wanting MXFP4 compute.
    Requires aiter; raises ``RuntimeError`` otherwise.
    """
    if not _aiter_available():
        raise RuntimeError(
            "quantize_to_mxfp4 requires the aiter library, which is not installed."
        )
    import aiter  # type: ignore

    model = _replace_linear_with_mxfp4(model)
    for _, module in model.named_modules():
        if hasattr(module, "quantize_"):
            module.quantize_()
    return model
