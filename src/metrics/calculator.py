"""
High-level metrics calculator for easy integration.
"""

from typing import Dict, Optional
from dataclasses import asdict

from src.core.agent_state import AgentState
from src.config.models import CompetenceVector
from src.metrics.formulas import (
    CoordinationMetrics,
    calculate_E_norm,
    calculate_C_eff,
    calculate_A_role,
    calculate_Q_coord,
)


class MetricsCalculator:
    """
    Main calculator class for LLM-MAS coordination metrics.
    
    Usage:
        # Initialize with competence vector
        calculator = MetricsCalculator(competence_vector)
        
        # Calculate metrics from agent state
        metrics = calculator.calculate(state, num_orders=3)
        
        # Or calculate from raw values
        metrics = calculator.calculate_from_raw(
            total_reward=8.5,
            steps_taken=15,
            total_tokens=5000,
            productive_actions=9,
            subtask_contributions={"agent1": 3, "agent2": 4, "agent3": 2}
        )
    """
    
    def __init__(
        self,
        competence_vector: CompetenceVector,
        max_steps: int = 20,
        max_possible_reward: float = 8.5,
    ):
        """
        Initialize metrics calculator.
        
        Args:
            competence_vector: CompetenceVector instance
            max_steps: Maximum allowed steps for E_norm calculation
            max_possible_reward: Theoretical maximum reward
        """
        self.competence_vector = competence_vector
        self.max_steps = max_steps
        self.max_possible_reward = max_possible_reward
    
    def calculate(
        self,
        state: AgentState,
        num_orders: int = 3,
        is_done: bool = None
    ) -> CoordinationMetrics:
        """
        Calculate all metrics from AgentState.
        
        Args:
            state: AgentState object with coordination data
            num_orders: Number of orders (for max_reward calculation)
            is_done: Override is_done flag
        
        Returns:
            CoordinationMetrics with all calculated values
        """
        if is_done is None:
            is_done = state.is_done
        
        # Calculate maximum possible reward based on num_orders
        max_reward = num_orders * (0.5 + 0.5 + 2.0) + 1.0  # Allocate + Negotiate + Plan_Route + completion bonus
        
        # E_norm
        E_norm = calculate_E_norm(
            total_reward=state.total_reward,
            steps_taken=state.steps_taken,
            max_steps=self.max_steps,
            max_possible_reward=max_reward,
            is_done=is_done
        )
        
        # C_eff and rho
        C_eff_norm, rho = calculate_C_eff(
            total_tokens=state.total_tokens,
            productive_actions=state.productive_actions
        )
        
        # A_role
        A_role, w_norm = calculate_A_role(
            subtask_contributions=state.subtask_contributions,
            competence_vector=self.competence_vector.vector
        )
        
        # Q_coord
        Q_coord = calculate_Q_coord(
            E_norm=E_norm,
            C_eff_norm=C_eff_norm,
            A_role=A_role
        )
        
        return CoordinationMetrics(
            E_norm=E_norm,
            C_eff_norm=C_eff_norm,
            rho=rho,
            A_role=A_role,
            R_avg=1.0,
            Q_coord=Q_coord,
            w_norm=w_norm
        )
    
    def calculate_from_raw(
        self,
        total_reward: float,
        steps_taken: int,
        total_tokens: int,
        productive_actions: int,
        subtask_contributions: Dict[str, int],
        is_done: bool = True,
        num_orders: int = 3
    ) -> CoordinationMetrics:
        """
        Calculate metrics from raw values (without AgentState).
        
        Args:
            total_reward: Cumulative reward
            steps_taken: Number of steps
            total_tokens: Total tokens consumed
            productive_actions: Number of productive actions
            subtask_contributions: Dict of agent_id -> contribution count
            is_done: Whether task completed
            num_orders: Number of orders
        
        Returns:
            CoordinationMetrics with all calculated values
        """
        state = AgentState(
            total_reward=total_reward,
            steps_taken=steps_taken,
            total_tokens=total_tokens,
            productive_actions=productive_actions,
            subtask_contributions=subtask_contributions,
            is_done=is_done
        )
        
        return self.calculate(state, num_orders=num_orders, is_done=is_done)
