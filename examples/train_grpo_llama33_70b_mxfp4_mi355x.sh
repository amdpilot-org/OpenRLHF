#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# GRPO training of Llama-3.3-70B (MXFP4) on 8x MI355X (gfx950) with the
# FSDP-AMD path (no DeepSpeed).
#
# This recipe uses:
#   * openrlhf.trainer.amd_grpo_trainer.AMDGRPOTrainer  (FSDP-native GRPO)
#   * openrlhf.utils.fsdp.FSDPStrategy                  (replaces DeepSpeed)
#   * openrlhf.utils.mxfp4_loader.load_mxfp4_model      (aiter MXFP4 weights)
#
# Usage:
#   bash examples/train_grpo_llama33_70b_mxfp4_mi355x.sh
#
# Requires: 8x MI355X, ROCm 7.x, aiter installed for MXFP4 acceleration.

set -euo pipefail

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export NNODES=${NNODES:-1}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# ROCm / aiter env
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTORCH_TUNABLEOP_ENABLED=${PYTORCH_TUNABLEOP_ENABLED:-1}
export PYTORCH_TUNABLEOP_VERBOSE=${PYTORCH_TUNABLEOP_VERBOSE:-0}
# aiter MXFP4 path
export AITER_MXFP4=1

MODEL_PATH=${MODEL_PATH:-amd/Llama-3.3-70B-Instruct-MXFP4-Preview}
DATASET_PATH=${DATASET_PATH:-data/gsm8k.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/llama33_70b_grpo_mi355x}

# GRPO hyperparameters
GROUP_SIZE=${GROUP_SIZE:-8}
ROLLOUT_BATCH=${ROLLOUT_BATCH:-64}
MICRO_BATCH=${MICRO_BATCH:-2}
RESPONSE_LEN=${RESPONSE_LEN:-1024}
LR=${LR:-1e-6}
KL_BETA=${KL_BETA:-0.001}
SAVE_STEPS=${SAVE_STEPS:-50}

mkdir -p "${OUTPUT_DIR}"

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m openrlhf.cli.train_grpo \
  --strategy fsdp \
  --pretrain "${MODEL_PATH}" \
  --mxfp4 \
  --dataset "${DATASET_PATH}" \
  --save_path "${OUTPUT_DIR}" \
  --micro_rollout_batch_size "${ROLLOUT_BATCH}" \
  --train_batch_size "${ROLLOUT_BATCH}" \
  --micro_train_batch_size "${MICRO_BATCH}" \
  --max_samples 100000 \
  --max_len 4096 \
  --generate_max_len "${RESPONSE_LEN}" \
  --zero_stage 3 \
  --bf16 \
  --adam_offload \
  --actor_learning_rate "${LR}" \
  --advantage_estimator group_norm \
  --kl_coef "${KL_BETA}" \
  --group_size "${GROUP_SIZE}" \
  --save_steps "${SAVE_STEPS}" \
  --max_ckpt_num 4 \
  --max_steps 1000 \
  --temperature 1.0 \
  --prompt_key prompt \
  --apply_chat_template \
  --rm_type gsm8k \
  --gradient_checkpointing

echo "GRPO training complete. Output: ${OUTPUT_DIR}"
