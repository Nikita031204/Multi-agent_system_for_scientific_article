"""
Model configuration and competence vector calculation.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelProfile:
    """
    Profile of an LLM model in the multi-agent system.
    
    Attributes:
        name: Display name of the model
        role: Assigned role (e.g., "Planner", "Executor")
        mmlu_pro: MMLU-Pro benchmark score (0-1)
        agentbench: AgentBench benchmark score (0-1)
        api_model_name: Model identifier for API calls
    """
    name: str
    role: str
    mmlu_pro: float
    agentbench: float
    api_model_name: str
    
    def __post_init__(self):
        if not 0 <= self.mmlu_pro <= 1:
            raise ValueError(f"mmlu_pro must be in [0, 1], got {self.mmlu_pro}")
        if not 0 <= self.agentbench <= 1:
            raise ValueError(f"agentbench must be in [0, 1], got {self.agentbench}")


class CompetenceVector:
    """
    Calculates and manages competence vectors for agent coordination.
    
    The competence vector C represents each agent's relative capability
    in the multi-agent system, normalized so that Σc_i = 1.
    
    Formula: c_i = (β·mmlu_pro + (1-β)·agentbench) / Σ(all raw scores)
    """
    
    def __init__(self, models_config: Dict[str, ModelProfile], beta: float = 0.6):
        """
        Initialize competence vector from model configurations.
        
        Args:
            models_config: Dictionary mapping agent_id -> ModelProfile
            beta: Weight for MMLU-Pro vs AgentBench (default 0.6)
        """
        self.models_config = models_config
        self.beta = beta
        self._vector = self._calculate()
    
    def _calculate(self) -> Dict[str, float]:
        """Calculate normalized competence vector."""
        raw_scores = {
            k: self.beta * v.mmlu_pro + (1 - self.beta) * v.agentbench
            for k, v in self.models_config.items()
        }
        total = sum(raw_scores.values())
        return {k: round(v / total, 4) for k, v in raw_scores.items()}
    
    @property
    def vector(self) -> Dict[str, float]:
        """Returns the competence vector."""
        return self._vector
    
    def __getitem__(self, agent_id: str) -> float:
        """Get competence value for a specific agent."""
        return self._vector.get(agent_id, 0.0)
    
    def __repr__(self) -> str:
        return f"CompetenceVector({self._vector})"
    
    def to_dict(self) -> Dict[str, float]:
        """Export as dictionary."""
        return self._vector.copy()
