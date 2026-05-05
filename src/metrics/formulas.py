"""
Core metric formulas for LLM-MAS coordination measurement.

Implements the mathematical framework from the research paper:
- E_norm: Coordination efficiency (Section 3.2.1)
- C_eff: Semantic cost (Section 3.2.2)
- A_role: Role alignment (Section 3.2.4)
- Q_coord: Integrated coordination quality (Section 3.3)
"""

from typing import Dict
from dataclasses import dataclass

from src.config.constants import ALPHA, LAMBDA, PHI_MAX, Q_WEIGHTS


@dataclass
class CoordinationMetrics:
    """
    Complete set of coordination metrics.
    
    Attributes:
        E_norm: Coordination efficiency (0-1, higher is better)
        C_eff_norm: Normalized semantic cost (0-1, lower is better)
        rho: Semantic density - productive actions per token
        A_role: Role alignment score (0-1, higher is better)
        R_avg: Robustness score (default 1.0 for baseline)
        Q_coord: Integrated coordination quality (0-1, higher is better)
        w_norm: Normalized work distribution per agent
    """
    E_norm: float
    C_eff_norm: float
    rho: float
    A_role: float
    R_avg: float
    Q_coord: float
    w_norm: Dict[str, float]
    
    def to_dict(self) -> dict:
        """Export metrics as dictionary."""
        return {
            "E_norm": round(self.E_norm, 3),
            "C_eff_norm": round(self.C_eff_norm, 3),
            "rho": round(self.rho, 4),
            "A_role": round(self.A_role, 3),
            "R_avg": round(self.R_avg, 3),
            "Q_coord": round(self.Q_coord, 3),
            "w_norm": {k: round(v, 2) for k, v in self.w_norm.items()},
        }


def calculate_E_norm(
    total_reward: float,
    steps_taken: int,
    max_steps: int,
    max_possible_reward: float = 8.5,
    is_done: bool = True,
    alpha: float = ALPHA
) -> float:
    """
    Calculate coordination efficiency (Section 3.2.1).
    
    E_norm = α·SR + (1-α)·(1 - T_actual/T_max)
    
    Where:
    - SR = Success Rate = total_reward / max_possible_reward
    - T_actual = steps_taken
    - T_max = max_steps
    
    Args:
        total_reward: Cumulative reward earned
        steps_taken: Number of coordination steps
        max_steps: Maximum allowed steps
        max_possible_reward: Theoretical maximum reward (default 8.5)
        is_done: Whether task was completed successfully
        alpha: Weight for SR vs time penalty (default 0.7)
    
    Returns:
        E_norm value in [0, 1]
    """
    if not is_done:
        return 0.0
    
    # Success Rate
    SR = min(1.0, total_reward / max_possible_reward)
    
    # Time penalty: reward faster completion
    time_penalty = 1 - (steps_taken / max_steps)
    time_penalty = max(0.0, time_penalty)
    
    return alpha * SR + (1 - alpha) * time_penalty


def calculate_C_eff(
    total_tokens: int,
    productive_actions: int,
    phi_max: int = PHI_MAX,
    lambda_: float = LAMBDA
) -> tuple[float, float]:
    """
    Calculate semantic cost and density (Section 3.2.2).
    
    ρ = (productive_actions × K) / total_tokens
    
    Where K = total_tokens / productive_actions (balancing coefficient)
    
    This ensures ρ ∈ [0, 1] where:
    - ρ = 1.0 means every token leads to productive action
    - ρ = 0.5 means half of tokens lead to productive action
    - ρ → 0 means very few productive actions
    
    C_eff_norm = (Φ_actual / Φ_max) · (1 + λ·(1 - ρ))
    
    Args:
        total_tokens: Total tokens consumed
        productive_actions: Number of actions with positive reward
        phi_max: Maximum token budget for normalization
        lambda_: Weight for semantic density
    
    Returns:
        Tuple of (C_eff_norm, rho)
    """
    # Balancing coefficient: tokens per productive action
    K = total_tokens / max(1, productive_actions)
    
    # Normalized semantic density (now in [0, 1] range)
    rho = min(1.0, productive_actions / max(1, total_tokens) * K)
    
    # Normalized semantic cost
    raw_ceff = (total_tokens / phi_max) * (1.0 + lambda_ * (1.0 - rho))
    C_eff_norm = min(1.0, max(0.0, raw_ceff))
    
    return C_eff_norm, rho


def calculate_A_role(
    subtask_contributions: Dict[str, int],
    competence_vector: Dict[str, float]
) -> tuple[float, Dict[str, float]]:
    """
    Calculate role alignment (Section 3.2.4).
    
    A_role = 1 - (1/n)·Σ|w_i - c_i|
    
    Where:
    - w_i = normalized work distribution for agent i
    - c_i = competence vector value for agent i
    - n = number of agents
    
    Perfect alignment (A_role = 1) occurs when work distribution
    exactly matches the competence vector.
    
    Args:
        subtask_contributions: Dict mapping agent_id -> productive action count
        competence_vector: Expected work distribution based on competence
    
    Returns:
        Tuple of (A_role, w_norm)
    """
    total_subtasks = sum(subtask_contributions.values())
    
    if total_subtasks == 0:
        return 0.0, {}
    
    # Normalize work distribution
    w_norm = {k: v / total_subtasks for k, v in subtask_contributions.items()}
    
    # Calculate alignment
    n = len(competence_vector)
    if n == 0:
        return 0.0, w_norm
    
    # Sum of absolute differences
    sum_abs_diff = sum(
        abs(w_norm.get(k, 0.0) - competence_vector[k])
        for k in competence_vector.keys()
    )
    
    A_role = 1.0 - (1.0 / n) * sum_abs_diff
    A_role = max(0.0, min(1.0, A_role))
    
    return A_role, w_norm


def calculate_Q_coord(
    E_norm: float,
    C_eff_norm: float,
    A_role: float,
    R_avg: float = 1.0,
    weights: Dict[str, float] = None
) -> float:
    """
    Calculate integrated coordination quality (Section 3.3).
    
    Q_coord = w_E·E_norm + w_C·(1 - C_eff_norm) + w_R·R_avg + w_A·A_role
    
    Note: C_eff is inverted because lower cost is better.
    
    Args:
        E_norm: Coordination efficiency
        C_eff_norm: Normalized semantic cost
        A_role: Role alignment score
        R_avg: Robustness score (default 1.0 for baseline)
        weights: Custom weights (default: {"E_norm": 0.4, "C_eff": 0.2, "R_avg": 0.2, "A_role": 0.2})
    
    Returns:
        Q_coord value in [0, 1]
    """
    if weights is None:
        weights = Q_WEIGHTS
    
    return (
        weights["E_norm"] * E_norm +
        weights["C_eff"] * (1 - C_eff_norm) +
        weights["R_avg"] * R_avg +
        weights["A_role"] * A_role
    )
