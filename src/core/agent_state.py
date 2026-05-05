"""
Agent state definition for LangGraph integration.
"""

from typing import Annotated, List, Dict, Optional
import operator
from dataclasses import dataclass


@dataclass
class AgentState:
    """
    State object for multi-agent coordination tracking.
    
    This class tracks all metrics needed for Q_coord calculation:
    - subtask_contributions: For A_role calculation
    - total_tokens: For C_eff calculation  
    - productive_actions: For rho (semantic density)
    - total_reward, steps_taken: For E_norm calculation
    
    Attributes:
        messages: List of messages from agents (LangGraph compatibility)
        subtask_contributions: Dict mapping agent_id -> number of productive actions
        total_tokens: Total tokens consumed by all LLM calls
        productive_actions: Count of actions that produced positive reward
        env_observation: Current environment state as text
        is_done: Whether episode is complete
        total_reward: Cumulative reward
        steps_taken: Number of coordination steps
        routing_decision: Optional routing decision for debugging
    """
    messages: Annotated[List, operator.add] = None
    subtask_contributions: Dict[str, int] = None
    total_tokens: int = 0
    productive_actions: int = 0
    env_observation: str = ""
    is_done: bool = False
    total_reward: float = 0.0
    steps_taken: int = 0
    routing_decision: Optional[str] = None
    
    def __post_init__(self):
        if self.messages is None:
            self.messages = []
        if self.subtask_contributions is None:
            self.subtask_contributions = {}
    
    def to_dict(self) -> dict:
        """Export state as dictionary for serialization."""
        return {
            "messages": self.messages,
            "subtask_contributions": self.subtask_contributions,
            "total_tokens": self.total_tokens,
            "productive_actions": self.productive_actions,
            "env_observation": self.env_observation,
            "is_done": self.is_done,
            "total_reward": self.total_reward,
            "steps_taken": self.steps_taken,
            "routing_decision": self.routing_decision,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AgentState":
        """Create AgentState from dictionary."""
        return cls(**data)
