"""
Quick start example for LLM-MAS Coordination Metrics Framework.

This example demonstrates how to:
1. Configure model profiles and competence vectors
2. Use the MetricsCalculator to measure coordination
3. Interpret the results
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ModelProfile, CompetenceVector
from src.core import AgentState
from src.metrics import MetricsCalculator, calculate_Q_coord


def main():
    print("=" * 60)
    print("LLM-MAS COORDINATION METRICS - Quick Start")
    print("=" * 60)
    
    # ========================================
    # 1. Configure your models
    # ========================================
    models_config = {
        "planner": ModelProfile(
            name="GPT-4",
            role="Planner",
            mmlu_pro=0.92,
            agentbench=0.95,
            api_model_name="gpt-4"
        ),
        "tactician": ModelProfile(
            name="Claude-3",
            role="Tactician",
            mmlu_pro=0.88,
            agentbench=0.90,
            api_model_name="claude-3-opus"
        ),
        "executor": ModelProfile(
            name="Mistral",
            role="Executor",
            mmlu_pro=0.70,
            agentbench=0.65,
            api_model_name="mistral-large"
        ),
    }
    
    # ========================================
    # 2. Calculate competence vector
    # ========================================
    competence_vector = CompetenceVector(models_config)
    print("\n[1] Competence Vector:")
    for agent_id, value in competence_vector.vector.items():
        print(f"    {agent_id}: {value}")
    
    # ========================================
    # 3. Create metrics calculator
    # ========================================
    calculator = MetricsCalculator(
        competence_vector=competence_vector,
        max_steps=20,
        max_possible_reward=8.5
    )
    
    # ========================================
    # 4. Simulate coordination results
    # ========================================
    print("\n[2] Simulating coordination scenario...")
    
    # Example: Agents completed a task with the following stats
    state = AgentState(
        messages=[],
        subtask_contributions={
            "planner": 4,      # Planner made 4 productive decisions
            "tactician": 3,    # Tactician made 3
            "executor": 2,     # Executor made 2
        },
        total_tokens=5000,
        productive_actions=9,
        env_observation="All orders delivered",
        is_done=True,
        total_reward=8.0,
        steps_taken=12,
    )
    
    # ========================================
    # 5. Calculate metrics
    # ========================================
    metrics = calculator.calculate(state, num_orders=3)
    
    print("\n[3] Coordination Metrics:")
    print("-" * 40)
    print(f"    E_norm (Efficiency):      {metrics.E_norm:.3f}")
    print(f"    C_eff_norm (Cost):        {metrics.C_eff_norm:.3f}")
    print(f"    rho (Semantic Density):   {metrics.rho:.4f}")
    print(f"    A_role (Role Alignment):  {metrics.A_role:.3f}")
    print(f"    R_avg (Robustness):       {metrics.R_avg:.3f}")
    print("-" * 40)
    print(f"    Q_coord (Overall Score):  {metrics.Q_coord:.3f}")
    
    # ========================================
    # 6. Work distribution analysis
    # ========================================
    print("\n[4] Work Distribution:")
    print(f"    Actual:  {metrics.w_norm}")
    print(f"    Expected: {competence_vector.vector}")
    
    # Calculate alignment deviation
    print("\n[5] Interpretation:")
    if metrics.Q_coord > 0.7:
        print("    [+] Excellent coordination quality")
    elif metrics.Q_coord > 0.5:
        print("    [~] Moderate coordination quality")
    else:
        print("    [-] Poor coordination quality")
    
    if metrics.A_role > 0.8:
        print("    [+] Agents worked according to their competence")
    elif metrics.A_role > 0.5:
        print("    [~] Some role misalignment detected")
    else:
        print("    [-] Significant role misalignment")
    
    # ========================================
    # 7. Using individual formulas
    # ========================================
    print("\n[6] Using individual formulas:")
    
    # You can also use formulas directly
    from src.metrics.formulas import calculate_E_norm as calc_e
    
    E_norm = calc_e(
        total_reward=8.0,
        steps_taken=12,
        max_steps=20,
        is_done=True
    )
    
    print(f"    Direct E_norm calculation: {E_norm:.3f}")
    
    print("\n" + "=" * 60)
    print("Quick start complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
