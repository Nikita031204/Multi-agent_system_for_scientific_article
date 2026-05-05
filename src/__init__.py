"""
LLM-MAS Coordination Metrics Framework

A framework for measuring coordination efficiency in multi-agent LLM systems.
Based on the Q_coord metric and related formulas from the research paper.
"""

from src.config import ModelProfile, CompetenceVector
from src.core.agent_state import AgentState
from src.metrics.calculator import CoordinationMetrics, MetricsCalculator
from src.metrics.formulas import calculate_E_norm, calculate_C_eff, calculate_A_role, calculate_Q_coord

__version__ = "1.0.0"
__author__ = "Your Name"
__all__ = [
    "ModelProfile",
    "CompetenceVector", 
    "AgentState",
    "CoordinationMetrics",
    "MetricsCalculator",
    "calculate_E_norm",
    "calculate_C_eff",
    "calculate_A_role",
    "calculate_Q_coord",
]
