"""AMD GRPO Trainer for single-node multi-GPU training on MI300/MI350 series.

This trainer implements Group Relative Policy Optimization (GRPO) without a
critic model, using DeepSpeed ZeRO for distributed training on AMD GPUs.
"""

from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)


class AMDGRPOTrainer:
    """GRPO trainer optimized for AMD MI300/MI350 GPUs using DeepSpeed.

    GRPO does not use a value network; instead it estimates advantages by
    normalizing rewards within each group of responses generated from the
    same prompt (group_norm).
    """

    def __init__(
        self,
        strategy,
        actor,
        actor_optim: torch.optim.Optimizer,
        actor_scheduler,
        tokenizer,
        micro_train_batch_size: int = 1,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        kl_coef: float = 0.01,
        entropy_coef: Optional[float] = None,
        **kwargs,
    ) -> None:
        from openrlhf.models import PolicyLoss
        from openrlhf.trainer.ppo_utils.replay_buffer import NaiveReplayBuffer

        self.strategy = strategy
        self.args = strategy.args
        self.actor = actor
        self.actor_optim = actor_optim
        self.actor_scheduler = actor_scheduler
        self.tokenizer = tokenizer
        self.micro_train_batch_size = micro_train_batch_size
        self.kl_coef = kl_coef
        self.entropy_coef = entropy_coef
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")

        self.actor_loss_fn = PolicyLoss(
            clip_eps_low=eps_clip,
            clip_eps_high=eps_clip,
            policy_loss_type="ppo",
        )

        self.replay_buffer = NaiveReplayBuffer(
            micro_train_batch_size,
            buffer_limit,
            buffer_cpu_offload,
            getattr(self.args.ds, "packing_samples", False),
            getattr(self.args.train, "dynamic_batch_enable", False),
        )

    def append_experience(self, experience) -> None:
        """Add experience to the replay buffer."""
        self.replay_buffer.append(experience)

    def compute_grpo_advantages(
        self, experiences, n_samples_per_prompt: int = 2
    ):
        """Compute group-normalized advantages for GRPO.

        For each group of responses generated from the same prompt, subtract
        the mean and divide by the standard deviation of rewards within the group.
        """
        if n_samples_per_prompt <= 1:
            raise ValueError("GRPO requires n_samples_per_prompt > 1")

        all_rewards = torch.cat([exp.rewards for exp in experiences])
        group_size = n_samples_per_prompt
        num_groups = len(all_rewards) // group_size

        advantages_list = []
        for i in range(num_groups):
            group_rewards = all_rewards[i * group_size : (i + 1) * group_size]
            mean_reward = group_rewards.mean()
            std_reward = group_rewards.std() + 1e-9
            normalized = (group_rewards - mean_reward) / std_reward
            advantages_list.append(normalized)

        advantages = torch.cat(advantages_list)

        # Per-token advantages: broadcast the scalar advantage across all action tokens
        for exp, adv in zip(experiences, advantages):
            action_len = exp.action_mask.sum().item()
            exp.advantages = torch.full((action_len,), adv.item(), device=exp.action_mask.device)
            exp.returns = exp.advantages.clone()

        return experiences

    @torch.no_grad()
    def generate_samples(self, prompts, max_length: int = 64, temperature: float = 1.0):
        """Generate synthetic experience for testing purposes.

        In a real workflow this would call vLLM or the actor model for generation.
        """
        experiences = []
        for prompt in prompts:
            # Tokenize prompt (simplified)
            tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            prompt_len = tokens.shape[1]

            # Fake generation: just append some tokens
            response_len = min(max_length - prompt_len, 16)
            if response_len <= 0:
                response_len = 4
            generated = torch.randint(0, self.tokenizer.vocab_size, (1, response_len), device=self.device)
            sequences = torch.cat([tokens, generated], dim=1)

            total_len = sequences.shape[1]
            action_mask = torch.zeros(total_len, dtype=torch.bool, device=self.device)
            action_mask[prompt_len:] = True
            attention_mask = torch.ones(total_len, dtype=torch.long, device=self.device)

            # Compute log probs (placeholder)
            with torch.cuda.amp.autocast(enabled=False):
                action_log_probs = self.actor(sequences, action_mask, attention_mask=attention_mask)

            # Fake reference log probs
            base_action_log_probs = action_log_probs.detach().clone()

            from openrlhf.trainer.ppo_utils.experience import Experience
            exp = Experience(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                action_log_probs=action_log_probs,
                base_action_log_probs=base_action_log_probs,
                rollout_log_probs=action_log_probs.detach().clone(),
                advantages=torch.zeros_like(action_log_probs),
                returns=torch.zeros_like(action_log_probs),
                rewards=torch.tensor([0.0], device=self.device),
                response_length=torch.tensor([response_len], device=self.device),
                truncated=torch.tensor([False], device=self.device),
                total_length=torch.tensor([total_len], device=self.device),
                info={},
            )
            experiences.append(exp)
        return experiences

    def training_step(self, experience, step: int) -> Dict[str, float]:
        """Execute one GRPO training step on a batch of experience."""
        self.actor.train()

        sequences = experience.sequences
        action_mask = experience.action_mask
        attention_mask = experience.attention_mask
        old_action_log_probs = experience.action_log_probs
        advantages = experience.advantages
        base_action_log_probs = experience.base_action_log_probs

        from openrlhf.utils.loss_utils import get_loss_batch_info
        loss_batch_info = get_loss_batch_info(
            self.strategy,
            action_mask,
            replay_buffer=self.replay_buffer,
            step=step,
            dynamic_batch=getattr(self.args.train, "dynamic_batch_enable", False),
        )

        action_log_probs, output = self.actor(
            sequences,
            action_mask,
            attention_mask=attention_mask,
            return_output=True,
            return_entropy=self.entropy_coef is not None,
        )

        actor_loss, clip_ratio, ppo_kl, _ = self.actor_loss_fn(
            action_log_probs,
            old_action_log_probs,
            advantages,
            action_mask=action_mask,
            rollout_log_probs=experience.rollout_log_probs,
            **loss_batch_info,
        )

        # KL penalty as loss
        if self.kl_coef > 0:
            from openrlhf.models.utils import masked_mean
            kl = (action_log_probs - base_action_log_probs).clamp(min=-20.0, max=20.0).exp()
            kl = (kl - 1 - (action_log_probs - base_action_log_probs)).clamp(min=0)
            kl_loss = masked_mean(kl, action_mask)
            actor_loss = actor_loss + self.kl_coef * kl_loss

        # Entropy bonus
        entropy_loss = 0.0
        if self.entropy_coef is not None and output.get("entropy") is not None:
            entropy_loss = -masked_mean(output["entropy"], action_mask)
            actor_loss = actor_loss + self.entropy_coef * entropy_loss

        self.strategy.backward(actor_loss, self.actor, self.actor_optim)
        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler)
        self.actor_optim.zero_grad()

        reward = experience.rewards.mean().item() if experience.rewards is not None else 0.0
        return {
            "policy_loss": actor_loss.item(),
            "reward": reward,
            "kl": ppo_kl.item() if isinstance(ppo_kl, torch.Tensor) else ppo_kl,
            "clip_ratio": clip_ratio.item() if isinstance(clip_ratio, torch.Tensor) else clip_ratio,
            "entropy_loss": entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else entropy_loss,
        }

    def grpo_train(self, num_epochs: int = 1) -> Dict[str, float]:
        """Run GRPO training on the replay buffer."""
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self.replay_buffer.collate_fn,
        )

        status_list = []
        for epoch in range(num_epochs):
            pbar = tqdm(dataloader, desc=f"GRPO epoch [{epoch + 1}/{num_epochs}]")
            for step, experience in enumerate(pbar):
                experience.to_device(self.device)
                status = self.training_step(experience, step)
                status_list.append(status)
                pbar.set_postfix(status)

        if status_list:
            return {k: sum(s[k] for s in status_list) / len(status_list) for k in status_list[0]}
        return {}

    def fit(
        self,
        prompts_dataloader,
        reward_fn,
        num_steps: int = 20,
        n_samples_per_prompt: int = 2,
        max_length: int = 2048,
        temperature: float = 1.0,
    ) -> Dict[str, List[float]]:
        """Main GRPO training loop.

        Args:
            prompts_dataloader: Iterable of prompt batches.
            reward_fn: Callable that takes (list of strings) -> tensor of rewards.
            num_steps: Number of GRPO update steps.
            n_samples_per_prompt: Number of responses per prompt (group size).
            max_length: Max sequence length.
            temperature: Sampling temperature.

        Returns:
            Dict with per-step metrics history.
        """
        history = {"reward": [], "policy_loss": [], "kl": []}
        global_step = 0

        for batch_idx, batch in enumerate(prompts_dataloader):
            if global_step >= num_steps:
                break

            prompts = batch if isinstance(batch, list) else batch["prompt"]

            # Generate multiple responses per prompt
            all_experiences = []
            for _ in range(n_samples_per_prompt):
                experiences = self.generate_samples(prompts, max_length=max_length, temperature=temperature)
                all_experiences.extend(experiences)

            # Compute rewards
            texts = [self.tokenizer.decode(exp.sequences[0], skip_special_tokens=True) for exp in all_experiences]
            rewards = reward_fn(texts)
            for exp, r in zip(all_experiences, rewards):
                exp.rewards = torch.tensor([r], device=self.device)

            # Compute GRPO advantages
            all_experiences = self.compute_grpo_advantages(all_experiences, n_samples_per_prompt)

            # Add to replay buffer
            for exp in all_experiences:
                self.append_experience(exp)

            # Train
            status = self.grpo_train(num_epochs=1)
            global_step += 1

            for k in history:
                if k in status:
                    history[k].append(status[k])

            logger.info(f"GRPO step {global_step}: {status}")

        return history
