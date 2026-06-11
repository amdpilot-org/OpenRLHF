#!/bin/bash
# AMD GRPO training recipe for Llama-3.3-70B-Instruct-MXFP4-Preview on 8xMI355X (gfx950)
#
# Uses DeepSpeed ZeRO-3 (FSDP-AMD equivalent) with BF16 and flash_attention_2.
# GRPO advantage estimator is group_norm (no critic model).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Model paths (cached on node)
POLICY_MODEL="amd/Llama-3.3-70B-Instruct-MXFP4-Preview"
REWARD_MODEL="Qwen/Qwen3-32B"

# Training hyperparameters
NUM_STEPS=20
MICRO_BATCH=2
MAX_LEN=2048
N_SAMPLES_PER_PROMPT=2
BATCH_SIZE=16
ROLLOUT_BATCH_SIZE=32

# DeepSpeed / AMD settings
ZERO_STAGE=3
PARAM_DTYPE="bf16"
ATTN_IMPL="flash_attention_2"

python3 -m openrlhf.cli.train_ppo_ray \
    --actor.num_nodes 1 \
    --actor.num_gpus_per_node 8 \
    --ref.num_nodes 1 \
    --ref.num_gpus_per_node 8 \
    --reward.num_nodes 1 \
    --reward.num_gpus_per_node 8 \
    --vllm.num_engines 4 \
    --vllm.tensor_parallel_size 2 \
    --vllm.gpu_memory_utilization 0.85 \
    --vllm.sync_backend nccl \
    --train.colocate_all \
    --vllm.enforce_eager \
    --actor.model_name_or_path "${POLICY_MODEL}" \
    --reward.model_name_or_path "${REWARD_MODEL}" \
    --ckpt.output_dir "${WORK_DIR}/ckpt/grpo_llama33_70b_mxfp4" \
    --ckpt.path "${WORK_DIR}/ckpt/grpo_llama33_70b_mxfp4/checkpoints" \
    --ckpt.save_hf \
    --train.batch_size ${BATCH_SIZE} \
    --rollout.batch_size ${ROLLOUT_BATCH_SIZE} \
    --rollout.n_samples_per_prompt ${N_SAMPLES_PER_PROMPT} \
    --rollout.micro_batch_size ${MICRO_BATCH} \
    --train.micro_batch_size ${MICRO_BATCH} \
    --train.max_epochs 1 \
    --train.num_episodes 1 \
    --data.max_len ${MAX_LEN} \
    --data.max_samples 100000 \
    --ds.zero_stage ${ZERO_STAGE} \
    --ds.param_dtype ${PARAM_DTYPE} \
    --ds.attn_implementation ${ATTN_IMPL} \
    --actor.gradient_checkpointing_enable \
    --ds.packing_samples \
    --train.dynamic_batch_enable \
    --train.max_tokens_per_gpu 16192 \
    --actor.adam.lr 1e-6 \
    --algo.advantage.estimator group_norm \
    --algo.kl.init_coef 0.01 \
    --algo.kl.use_loss \
    --algo.kl.estimator k3 \
    --reward.normalize_enable \
    --data.prompt_dataset OpenRLHF/prompt-collection-v0.1 \
    --data.input_key context_messages \
    --data.apply_chat_template \
    --logger.logging_steps 1 \
    --eval.steps -1 \
    --ckpt.save_steps -1
