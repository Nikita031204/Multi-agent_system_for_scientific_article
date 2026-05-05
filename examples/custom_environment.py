"""
Example: Using the framework with the Semantic Supply Chain environment.

This demonstrates a complete workflow:
1. Set up environment and models
2. Run coordination experiment
3. Calculate and interpret metrics
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from src.config import ModelProfile, CompetenceVector
from src.environment import SemanticSupplyChainEnv
from src.core import AgentState
from src.metrics import MetricsCalculator


async def run_simple_coordination():
    """Run a simple coordination scenario."""
    
    # Configure models
    models = {
        "agent-a": ModelProfile("Model-A", "Planner", 0.9, 0.85, "model-a"),
        "agent-b": ModelProfile("Model-B", "Executor", 0.7, 0.65, "model-b"),
    }
    
    competence = CompetenceVector(models)
    print(f"Competence Vector: {competence.vector}")
    
    # Create environment
    env = SemanticSupplyChainEnv(grid_size=3, max_steps=20, seed=42)
    
    # Initialize state
    agent_ids = list(models.keys())
    initial_obs = env.reset(agent_ids)
    
    state = AgentState(
        subtask_contributions={aid: 0 for aid in agent_ids},
        total_tokens=0,
        productive_actions=0,
        env_observation=initial_obs,
        steps_taken=0,
        total_reward=0.0,
        is_done=False
    )
    
    print("\nInitial observation:")
    print(initial_obs)
    
    # Simulate some actions
    actions = [
        ("agent-a", {"type": "Check_Inventory", "direction": "right"}),
        ("agent-b", {"type": "Check_Inventory", "direction": "down"}),
    ]
    
    for agent_id, action in actions:
        obs, reward, done, productive = env.step(agent_id, action)
        state.total_reward += reward
        if productive:
            state.productive_actions += 1
            state.subtask_contributions[agent_id] += 1
        state.total_tokens += 100  # Simulated token usage
        state.steps_taken += 1
        state.is_done = done
        print(f"\n{agent_id} executed {action['type']}: reward={reward}")
    
    # Calculate metrics
    calculator = MetricsCalculator(competence)
    metrics = calculator.calculate(state, num_orders=3)
    
    print("\n" + "=" * 40)
    print("METRICS:")
    print(f"  Q_coord: {metrics.Q_coord:.3f}")
    print(f"  E_norm:  {metrics.E_norm:.3f}")
    print(f"  A_role:  {metrics.A_role:.3f}")
    print("=" * 40)
    
    return metrics


if __name__ == "__main__":
    asyncio.run(run_simple_coordination())
