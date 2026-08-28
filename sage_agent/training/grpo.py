"""GRPO (Group Relative Policy Optimization) training for SAGE-Agent.

This module implements GRPO training with certainty-weighted rewards from
Section 6.2 of the paper "Structured Uncertainty Guided Clarification for LLM Agents".

Key insight from the paper:
    The agent should be rewarded for taking actions that align with its certainty:
    - Execute tool calls when confident (high max probability)
    - Ask clarifying questions when uncertain (low max probability)

Certainty-weighted reward (Equation from Section 6.2):
    Cert(a_t) = max_c π_c(t)           if a_t is a tool call
    Cert(a_t) = 1 - max_c π_c(t)       if a_t is a clarification question
    R(a_t) = Cert(a_t) * r_base(a_t)   final weighted reward

This ensures the policy learns to calibrate its actions with its uncertainty,
asking questions when needed rather than making uncertain tool calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Iterable, List, Optional, Protocol, Sequence, Tuple

import torch


class ActionType(Enum):
    """Type of action taken by the agent.

    Used to determine how to compute certainty weight for the reward.
    """
    TOOL_CALL = auto()       # Agent executed a tool call
    CLARIFICATION = auto()   # Agent asked a clarifying question
    OTHER = auto()           # Other action (e.g., final response)


@dataclass
class CertaintyWeightedReward:
    """Certainty-weighted reward computation per Section 6.2 of the paper.

    The key insight is that rewards should be weighted by how well the action
    aligns with the agent's uncertainty state:

    For tool calls:
        Cert(a_t) = max_c π_c(t)
        High certainty → high weight → reward tool calls when confident

    For clarification questions:
        Cert(a_t) = 1 - max_c π_c(t)
        Low certainty → high weight → reward questions when uncertain

    Final reward:
        R(a_t) = Cert(a_t) * r_base(a_t)

    Attributes:
        min_certainty: Minimum certainty value to avoid zero rewards (default: 0.01)
    """
    min_certainty: float = 0.01

    def compute_certainty(
        self,
        action_type: ActionType,
        max_candidate_probability: float,
    ) -> float:
        """Compute certainty weight for an action.

        Args:
            action_type: Type of action (TOOL_CALL, CLARIFICATION, OTHER)
            max_candidate_probability: Maximum probability among candidates, max_c π_c(t)

        Returns:
            Certainty weight in range [min_certainty, 1.0]
        """
        if action_type == ActionType.TOOL_CALL:
            # Tool calls should be rewarded when confident
            certainty = max_candidate_probability
        elif action_type == ActionType.CLARIFICATION:
            # Questions should be rewarded when uncertain
            certainty = 1.0 - max_candidate_probability
        else:
            # Other actions: neutral certainty
            certainty = 0.5

        return max(self.min_certainty, certainty)

    def weighted_reward(
        self,
        base_reward: float,
        action_type: ActionType,
        max_candidate_probability: float,
    ) -> float:
        """Compute certainty-weighted reward.

        R(a_t) = Cert(a_t) * r_base(a_t)

        Args:
            base_reward: Base reward from environment/oracle
            action_type: Type of action taken
            max_candidate_probability: Maximum probability among candidates

        Returns:
            Certainty-weighted reward
        """
        certainty = self.compute_certainty(action_type, max_candidate_probability)
        return certainty * base_reward


class PolicyModel(Protocol):
    def sample(self, prompts: Sequence[str], num_samples: int) -> List[List[str]]:
        ...

    def log_probs(self, prompts: Sequence[str], responses: Sequence[Sequence[str]]) -> torch.Tensor:
        """Return log-probabilities shaped [batch, num_samples]."""
        ...


class ReferenceModel(Protocol):
    def log_probs(self, prompts: Sequence[str], responses: Sequence[Sequence[str]]) -> torch.Tensor:
        ...


RewardFn = Callable[[str, str], float]


@dataclass
class GRPOConfig:
    num_samples: int = 4
    kl_coef: float = 0.02
    max_grad_norm: float = 1.0
    device: str = "cpu"


@dataclass
class GRPOStepResult:
    loss: float
    mean_reward: float
    mean_kl: float


class GRPOTrainer:
    def __init__(
        self,
        policy: PolicyModel,
        reference: ReferenceModel,
        reward_fn: RewardFn,
        config: GRPOConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.policy = policy
        self.reference = reference
        self.reward_fn = reward_fn
        self.config = config
        self.optimizer = optimizer

    def train_epoch(self, prompts: Sequence[str]) -> List[GRPOStepResult]:
        results: List[GRPOStepResult] = []
        for prompt in prompts:
            result = self._train_step([prompt])
            results.append(result)
        return results

    def _train_step(self, prompts: Sequence[str]) -> GRPOStepResult:
        responses = self.policy.sample(prompts, self.config.num_samples)
        rewards = self._compute_rewards(prompts, responses)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.config.device)

        policy_log_probs = self.policy.log_probs(prompts, responses)
        ref_log_probs = self.reference.log_probs(prompts, responses)
        kl = policy_log_probs - ref_log_probs

        advantages = _group_normalize(rewards_tensor)
        loss = -(advantages * policy_log_probs).mean() + self.config.kl_coef * kl.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.optimizer.param_groups[0]["params"], self.config.max_grad_norm
        )
        self.optimizer.step()

        return GRPOStepResult(
            loss=float(loss.detach().cpu().item()),
            mean_reward=float(rewards_tensor.mean().cpu().item()),
            mean_kl=float(kl.mean().detach().cpu().item()),
        )

    def _compute_rewards(
        self, prompts: Sequence[str], responses: Sequence[Sequence[str]]
    ) -> List[List[float]]:
        reward_matrix: List[List[float]] = []
        for prompt, response_group in zip(prompts, responses):
            row = [self.reward_fn(prompt, response) for response in response_group]
            reward_matrix.append(row)
        return reward_matrix


def _group_normalize(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if rewards.numel() == 0:
        return rewards
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True) + eps
    return (rewards - mean) / std


# =============================================================================
# SAGE-Agent Specific GRPO Training (Section 6.2)
# =============================================================================


@dataclass
class SAGEAction:
    """An action taken by the SAGE agent with associated metadata.

    Attributes:
        response: The generated response text
        action_type: Type of action (TOOL_CALL, CLARIFICATION, OTHER)
        max_candidate_probability: Maximum probability among candidates at decision time
    """
    response: str
    action_type: ActionType
    max_candidate_probability: float


@dataclass
class SAGEGRPOConfig:
    """Configuration for SAGE-Agent GRPO training.

    Extends base GRPO config with certainty-weighting parameters.

    Attributes:
        num_samples: Number of response samples per prompt
        kl_coef: KL divergence coefficient (β in GRPO)
        max_grad_norm: Maximum gradient norm for clipping
        device: PyTorch device for tensors
        min_certainty: Minimum certainty value for reward weighting
        use_certainty_weighting: Whether to apply certainty weighting
    """
    num_samples: int = 4
    kl_coef: float = 0.02
    max_grad_norm: float = 1.0
    device: str = "cpu"
    # SAGE-specific
    min_certainty: float = 0.01
    use_certainty_weighting: bool = True


# Type alias for SAGE reward function that returns action metadata
SAGERewardFn = Callable[[str, str], Tuple[float, ActionType, float]]


class SAGEGRPOTrainer:
    """GRPO trainer with certainty-weighted rewards for SAGE-Agent.

    This trainer implements the training procedure from Section 6.2 of the paper,
    where rewards are weighted by how well actions align with uncertainty:

    - Tool calls are rewarded more when the agent is confident
    - Clarification questions are rewarded more when the agent is uncertain

    This encourages the policy to learn proper calibration between
    taking action and seeking clarification.
    """

    def __init__(
        self,
        policy: PolicyModel,
        reference: ReferenceModel,
        reward_fn: SAGERewardFn,
        config: SAGEGRPOConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Initialize SAGE GRPO trainer.

        Args:
            policy: Policy model to train
            reference: Reference model for KL constraint
            reward_fn: Function returning (base_reward, action_type, max_prob)
            config: Training configuration
            optimizer: PyTorch optimizer for policy parameters
        """
        self.policy = policy
        self.reference = reference
        self.reward_fn = reward_fn
        self.config = config
        self.optimizer = optimizer
        self.certainty_reward = CertaintyWeightedReward(
            min_certainty=config.min_certainty
        )

    def train_epoch(self, prompts: Sequence[str]) -> List[GRPOStepResult]:
        """Train one epoch over the given prompts.

        Args:
            prompts: Sequence of training prompts

        Returns:
            List of step results, one per prompt
        """
        results: List[GRPOStepResult] = []
        for prompt in prompts:
            result = self._train_step([prompt])
            results.append(result)
        return results

    def _train_step(self, prompts: Sequence[str]) -> GRPOStepResult:
        """Perform one training step on a batch of prompts."""
        responses = self.policy.sample(prompts, self.config.num_samples)
        rewards = self._compute_rewards(prompts, responses)
        rewards_tensor = torch.tensor(
            rewards, dtype=torch.float32, device=self.config.device
        )

        policy_log_probs = self.policy.log_probs(prompts, responses)
        ref_log_probs = self.reference.log_probs(prompts, responses)
        kl = policy_log_probs - ref_log_probs

        advantages = _group_normalize(rewards_tensor)
        loss = -(advantages * policy_log_probs).mean() + self.config.kl_coef * kl.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.optimizer.param_groups[0]["params"], self.config.max_grad_norm
        )
        self.optimizer.step()

        return GRPOStepResult(
            loss=float(loss.detach().cpu().item()),
            mean_reward=float(rewards_tensor.mean().cpu().item()),
            mean_kl=float(kl.mean().detach().cpu().item()),
        )

    def _compute_rewards(
        self, prompts: Sequence[str], responses: Sequence[Sequence[str]]
    ) -> List[List[float]]:
        """Compute certainty-weighted rewards for responses.

        For each response, the reward function returns:
        - base_reward: The task-specific reward
        - action_type: Whether it was a tool call or clarification
        - max_prob: Maximum candidate probability at decision time

        The final reward is: R(a_t) = Cert(a_t) * r_base(a_t)
        """
        reward_matrix: List[List[float]] = []
        for prompt, response_group in zip(prompts, responses):
            row: List[float] = []
            for response in response_group:
                base_reward, action_type, max_prob = self.reward_fn(prompt, response)

                if self.config.use_certainty_weighting:
                    weighted = self.certainty_reward.weighted_reward(
                        base_reward, action_type, max_prob
                    )
                else:
                    weighted = base_reward

                row.append(weighted)
            reward_matrix.append(row)
        return reward_matrix
