"""
Unit tests for coordination metrics.
"""

import pytest
from src.config import ModelProfile, CompetenceVector
from src.core import AgentState
from src.metrics import (
    MetricsCalculator,
    calculate_E_norm,
    calculate_C_eff,
    calculate_A_role,
    calculate_Q_coord,
)


class TestModelProfile:
    """Tests for ModelProfile."""
    
    def test_valid_profile(self):
        profile = ModelProfile("Test", "Planner", 0.9, 0.8, "test-model")
        assert profile.name == "Test"
        assert profile.mmlu_pro == 0.9
    
    def test_invalid_mmlu_pro(self):
        with pytest.raises(ValueError):
            ModelProfile("Test", "Planner", 1.5, 0.8, "test-model")
    
    def test_invalid_agentbench(self):
        with pytest.raises(ValueError):
            ModelProfile("Test", "Planner", 0.9, -0.1, "test-model")


class TestCompetenceVector:
    """Tests for CompetenceVector."""
    
    @pytest.fixture
    def models(self):
        return {
            "agent1": ModelProfile("A1", "Planner", 0.9, 0.8, "a1"),
            "agent2": ModelProfile("A2", "Worker", 0.7, 0.6, "a2"),
        }
    
    def test_vector_normalization(self, models):
        cv = CompetenceVector(models)
        assert abs(sum(cv.vector.values()) - 1.0) < 0.001
    
    def test_higher_competence_higher_value(self, models):
        cv = CompetenceVector(models)
        assert cv["agent1"] > cv["agent2"]
    
    def test_custom_beta(self, models):
        cv_default = CompetenceVector(models, beta=0.6)
        cv_custom = CompetenceVector(models, beta=0.8)
        # Higher beta = more weight on MMLU-Pro
        assert cv_default.vector != cv_custom.vector


class TestE_Norm:
    """Tests for E_norm calculation."""
    
    def test_perfect_score(self):
        E = calculate_E_norm(
            total_reward=8.5,
            steps_taken=10,
            max_steps=20,
            is_done=True
        )
        assert E > 0.8
    
    def test_incomplete_task(self):
        E = calculate_E_norm(
            total_reward=5.0,
            steps_taken=15,
            max_steps=20,
            is_done=False
        )
        assert E == 0.0  # Not done = 0 efficiency
    
    def test_slow_completion(self):
        E_fast = calculate_E_norm(total_reward=8.5, steps_taken=5, max_steps=20, is_done=True)
        E_slow = calculate_E_norm(total_reward=8.5, steps_taken=18, max_steps=20, is_done=True)
        assert E_fast > E_slow


class TestC_Eff:
    """Tests for C_eff calculation."""
    
    def test_high_density(self):
        C_eff, rho = calculate_C_eff(total_tokens=1000, productive_actions=100)
        assert rho == pytest.approx(0.1, rel=0.01)
        assert C_eff < 0.01
    
    def test_low_density(self):
        C_eff, rho = calculate_C_eff(total_tokens=10000, productive_actions=1)
        assert rho == pytest.approx(0.0001, rel=0.01)
        assert C_eff > 0


class TestA_Role:
    """Tests for A_role calculation."""
    
    def test_perfect_alignment(self):
        contributions = {"a": 60, "b": 40}
        competence = {"a": 0.6, "b": 0.4}
        A_role, w_norm = calculate_A_role(contributions, competence)
        assert A_role == pytest.approx(1.0, rel=0.01)
    
    def test_no_alignment(self):
        contributions = {"a": 100, "b": 0}
        competence = {"a": 0.0, "b": 1.0}  # Inverted
        A_role, w_norm = calculate_A_role(contributions, competence)
        assert A_role < 0.5
    
    def test_empty_contributions(self):
        A_role, w_norm = calculate_A_role({}, {"a": 0.5, "b": 0.5})
        assert A_role == 0.0


class TestQ_Coord:
    """Tests for Q_coord calculation."""
    
    def test_high_coordination(self):
        Q = calculate_Q_coord(
            E_norm=0.9,
            C_eff_norm=0.1,
            A_role=0.9
        )
        assert Q > 0.7
    
    def test_low_coordination(self):
        Q = calculate_Q_coord(
            E_norm=0.2,
            C_eff_norm=0.8,
            A_role=0.2
        )
        assert Q < 0.5
    
    def test_custom_weights(self):
        weights = {"E_norm": 0.5, "C_eff": 0.5, "R_avg": 0.0, "A_role": 0.0}
        Q = calculate_Q_coord(E_norm=1.0, C_eff_norm=0.0, A_role=0.0, weights=weights)
        assert Q == pytest.approx(0.5, rel=0.01)


class TestMetricsCalculator:
    """Tests for MetricsCalculator."""
    
    @pytest.fixture
    def calculator(self):
        models = {
            "planner": ModelProfile("P", "Planner", 0.9, 0.8, "p"),
            "worker": ModelProfile("W", "Worker", 0.7, 0.6, "w"),
        }
        competence = CompetenceVector(models)
        return MetricsCalculator(competence)
    
    def test_calculate_from_state(self, calculator):
        state = AgentState(
            subtask_contributions={"planner": 5, "worker": 3},
            total_tokens=5000,
            productive_actions=8,
            total_reward=8.0,
            steps_taken=12,
            is_done=True
        )
        metrics = calculator.calculate(state)
        assert 0 <= metrics.Q_coord <= 1
        assert metrics.E_norm > 0
        assert 0 <= metrics.A_role <= 1
    
    def test_calculate_from_raw(self, calculator):
        metrics = calculator.calculate_from_raw(
            total_reward=7.5,
            steps_taken=15,
            total_tokens=3000,
            productive_actions=6,
            subtask_contributions={"planner": 3, "worker": 3}
        )
        assert metrics.Q_coord >= 0
