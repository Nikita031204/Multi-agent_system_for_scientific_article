"""
Framework constants and hyperparameters.

These values are used in the Q_coord calculation and can be customized.
"""

# Weight for Success Rate in E_norm calculation (Section 3.2.1)
ALPHA: float = 0.7

# Weight for semantic density in C_eff calculation (Section 3.2.2)
LAMBDA: float = 1.0

# Maximum token budget for normalization (Section 3.2.2)
# Adjust based on your LLM provider limits
PHI_MAX: int = 2_000_000

# Weight for MMLU-Pro vs AgentBench in competence vector
BETA: float = 0.6

# Q_coord weights (Section 3.3)
# These weights determine the importance of each component
Q_WEIGHTS = {
    "E_norm": 0.4,      # Coordination efficiency
    "C_eff": 0.2,       # Semantic cost (inverted)
    "R_avg": 0.2,       # Robustness
    "A_role": 0.2,      # Role alignment
}
