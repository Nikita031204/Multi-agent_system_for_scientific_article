"""
Topology templates for multi-agent coordination.

This module provides reference implementations of coordination topologies
that can be used to compare coordination metrics.

Topology Types:
- Flat Democracy: All agents vote on actions
- Naive Hierarchy: Manager delegates to workers
- Competency-Aligned: Work assigned by competence
"""

from src.core import AgentState

# Placeholder for topology implementations
# Full implementations depend on LangGraph integration

__all__ = [
    "TopologyBase",
    "build_flat_democracy",
    "build_naive_hierarchy",
]


class TopologyBase:
    """
    Base class for coordination topologies.
    
    Subclasses should implement the build() method to create
    a LangGraph workflow that coordinates agents.
    """
    
    def __init__(self, env, competence_vector):
        """
        Initialize topology.
        
        Args:
            env: Environment instance
            competence_vector: CompetenceVector for agent capabilities
        """
        self.env = env
        self.competence_vector = competence_vector
    
    def build(self):
        """
        Build the coordination graph.
        
        Returns:
            Compiled LangGraph workflow
        """
        raise NotImplementedError("Subclasses must implement build()")


def build_flat_democracy(env, competence_vector=None, llm_configs=None):
    """
    Build Flat Democracy topology.
    
    In this topology:
    1. All agents propose actions
    2. All agents vote on proposals
    3. Winner executes
    
    Args:
        env: Environment instance
        competence_vector: Optional competence vector
        llm_configs: Optional LLM configurations
    
    Returns:
        LangGraph workflow
    """
    # Placeholder - full implementation in topologies/
    pass


def build_naive_hierarchy(env, competence_vector=None, llm_configs=None):
    """
    Build Naive Hierarchy topology.
    
    In this topology:
    1. Manager agent plans and delegates
    2. Worker agents execute
    
    Args:
        env: Environment instance
        competence_vector: Optional competence vector
        llm_configs: Optional LLM configurations
    
    Returns:
        LangGraph workflow
    """
    # Placeholder - full implementation in topologies/
    pass
