# Полное руководство по LLM-MAS Coordination Metrics

## Содержание

1. [Введение](#введение)
2. [Установка и настройка](#установка-и-настройка)
3. [Базовые концепции](#базовые-концепции)
4. [Пошаговое обучение](#пошаговое-обучение)
5. [API Reference](#api-reference)
6. [Продвинутые сценарии](#продвинутые-сценарии)
7. [Интеграция с LangGraph](#интеграция-с-langgraph)
8. [Частые вопросы](#частые-вопросы)
9. [Troubleshooting](#troubleshooting)

---

## Введение

### Что такое координация LLM-агентов?

Когда несколько языковых моделей работают вместе над одной задачей, им нужно координировать свои действия. Например:

- **Планировщик** декомпозирует задачу на подзадачи
- **Исполнитель** выполняет конкретные действия
- **Критик** проверяет результаты

Проблема: **Как понять, насколько хорошо они работают вместе?**

### Почему недостаточно обычных метрик?

| Метрика | Ограничение |
|---------|-------------|
| Точность ответа | Не учитывает командную работу |
| Время ответа | Игнорирует качество взаимодействия |
| Стоимость токенов | Не показывает эффективность использования |

**Q_coord** — метрика, специально созданная для оценки командной работы LLM-агентов.

---

## Установка и настройка

### Требования

- Python 3.9+
- pip или poetry

### Установка

```bash
# Клонирование репозитория
git clone https://github.com/yourusername/llm-mas-coordination-metrics.git
cd llm-mas-coordination-metrics

# Установка в режиме разработки
pip install -e .

# Или с зависимостями для разработки
pip install -e ".[dev]"
```

### Проверка установки

```python
from src.config import ModelProfile, CompetenceVector
from src.metrics import MetricsCalculator

print("Фреймворк установлен успешно!")
```

---

## Базовые концепции

### 1. Вектор компетенций (Competence Vector)

Каждый агент имеет свои сильные стороны. Вектор компетенций показывает ожидаемое распределение работы.

**Формула:**
```
c_i = (β × mmlu_pro + (1-β) × agentbench) / Σ(все raw scores)
```

**Пример:**
```python
from src.config import ModelProfile, CompetenceVector

models = {
    "strategist": ModelProfile(
        name="GPT-4",
        role="Стратег",
        mmlu_pro=0.92,      # Отлично понимает сложные задачи
        agentbench=0.95,    # Хорошо планирует действия
        api_model_name="gpt-4"
    ),
    "worker": ModelProfile(
        name="Claude-3-Sonnet",
        role="Исполнитель",
        mmlu_pro=0.75,      # Хорошо выполняет инструкции
        agentbench=0.70,
        api_model_name="claude-3-sonnet"
    ),
}

competence = CompetenceVector(models, beta=0.6)

# Результат: strategist ~0.58, worker ~0.42
# Это значит: стратег должен делать ~58% полезной работы
print(competence.vector)
```

### 2. Вклад агентов (Subtask Contributions)

Фактическое распределение работы между агентами.

```python
contributions = {
    "strategist": 6,  # Выполнил 6 полезных действий
    "worker": 4,      # Выполнил 4 полезных действия
}
```

### 3. Семантическая плотность (ρ)

Показывает, сколько полезных действий на один потраченный токен.

```python
ρ = productive_actions / total_tokens

# Пример: 9 полезных действий / 5000 токенов = 0.0018
```

### 4. Состояние агента (AgentState)

Контейнер для всех данных о координации.

```python
from src.core import AgentState

state = AgentState(
    subtask_contributions={"agent1": 5, "agent2": 3},
    total_tokens=5000,
    productive_actions=8,
    total_reward=7.5,
    steps_taken=12,
    is_done=True
)
```

---

## Пошаговое обучение

### Урок 1: Создаём первую конфигурацию

**Задача:** Настроить систему из двух агентов.

```python
from src.config import ModelProfile, CompetenceVector

# Шаг 1: Определяем агентов
models = {
    "leader": ModelProfile(
        name="GPT-4-Turbo",
        role="Лидер",
        mmlu_pro=0.90,
        agentbench=0.92,
        api_model_name="gpt-4-turbo-preview"
    ),
    "assistant": ModelProfile(
        name="GPT-3.5-Turbo",
        role="Помощник",
        mmlu_pro=0.65,
        agentbench=0.60,
        api_model_name="gpt-3.5-turbo"
    ),
}

# Шаг 2: Вычисляем вектор компетенций
competence = CompetenceVector(models)

# Шаг 3: Выводим результат
print("Распределение компетенций:")
for agent_id, value in competence.vector.items():
    print(f"  {agent_id}: {value:.2%}")
```

**Вывод:**
```
Распределение компетенций:
  leader: 62.45%
  assistant: 37.55%
```

### Урок 2: Рассчитываем метрики

**Задача:** Оценить результаты работы команды.

```python
from src.metrics import MetricsCalculator

# Шаг 1: Создаём калькулятор
calculator = MetricsCalculator(
    competence_vector=competence,
    max_steps=20,
    max_possible_reward=10.0
)

# Шаг 2: Вводим данные эксперимента
metrics = calculator.calculate_from_raw(
    total_reward=8.5,          # Заработано награды
    steps_taken=14,            # Потрачено шагов
    total_tokens=3500,         # Потрачено токенов
    productive_actions=7,      # Полезных действий
    subtask_contributions={    # Вклад по агентам
        "leader": 4,
        "assistant": 3
    },
    is_done=True
)

# Шаг 3: Анализируем результаты
print(f"Q_coord = {metrics.Q_coord:.3f}")
print(f"E_norm = {metrics.E_norm:.3f}")
print(f"A_role = {metrics.A_role:.3f}")
```

### Урок 3: Интерпретация метрик

```python
def analyze_coordination(metrics):
    """Анализ качества координации."""
    
    print("=== АНАЛИЗ КООРДИНАЦИИ ===\n")
    
    # 1. Общая оценка
    if metrics.Q_coord > 0.8:
        print("[+] Отличная координация!")
    elif metrics.Q_coord > 0.6:
        print("[~] Хорошая координация")
    elif metrics.Q_coord > 0.4:
        print("[-] Удовлетворительная координация")
    else:
        print("[!] Требуется оптимизация")
    
    # 2. Эффективность
    print(f"\nЭффективность (E_norm = {metrics.E_norm:.2f}):")
    if metrics.E_norm > 0.7:
        print("  - Задачи выполняются быстро и успешно")
    else:
        print("  - Возможно слишком много шагов или низкая награда")
    
    # 3. Стоимость
    print(f"\nСемантическая плотность (ρ = {metrics.rho:.4f}):")
    if metrics.rho > 0.002:
        print("  - Эффективное использование токенов")
    else:
        print("  - Много "пустых" действий")
    
    # 4. Ролевое соответствие
    print(f"\nРолевое соответствие (A_role = {metrics.A_role:.2f}):")
    if metrics.A_role > 0.8:
        print("  - Работа распределена по компетенциям")
    else:
        print("  - Есть дисбаланс в распределении работы")

analyze_coordination(metrics)
```

### Урок 4: Сравнение топологий

**Задача:** Сравнить два подхода к координации.

```python
# Топология 1: Демократия (все голосуют)
democracy_metrics = calculator.calculate_from_raw(
    total_reward=7.8,
    steps_taken=16,
    total_tokens=4500,
    productive_actions=8,
    subtask_contributions={"leader": 4, "assistant": 4},
    is_done=True
)

# Топология 2: Иерархия (лидер решает)
hierarchy_metrics = calculator.calculate_from_raw(
    total_reward=8.2,
    steps_taken=12,
    total_tokens=3200,
    productive_actions=7,
    subtask_contributions={"leader": 5, "assistant": 2},
    is_done=True
)

# Сравнение
print("СРАВНЕНИЕ ТОПОЛОГИЙ")
print("-" * 40)
print(f"{'Метрика':<15} {'Демократия':<12} {'Иерархия':<12}")
print("-" * 40)
print(f"{'Q_coord':<15} {democracy_metrics.Q_coord:<12.3f} {hierarchy_metrics.Q_coord:<12.3f}")
print(f"{'E_norm':<15} {democracy_metrics.E_norm:<12.3f} {hierarchy_metrics.E_norm:<12.3f}")
print(f"{'A_role':<15} {democracy_metrics.A_role:<12.3f} {hierarchy_metrics.A_role:<12.3f}")

# Вывод
if hierarchy_metrics.Q_coord > democracy_metrics.Q_coord:
    print("\n[Рекомендация] Иерархия эффективнее для данной задачи")
else:
    print("\n[Рекомендация] Демократия эффективнее для данной задачи")
```

---

## API Reference

### ModelProfile

```python
ModelProfile(
    name: str,           # Отображаемое имя
    role: str,           # Роль в команде
    mmlu_pro: float,     # MMLU-Pro score [0, 1]
    agentbench: float,   # AgentBench score [0, 1]
    api_model_name: str  # Имя модели для API
)
```

### CompetenceVector

```python
# Создание
cv = CompetenceVector(
    models_config: Dict[str, ModelProfile],
    beta: float = 0.6  # Вес MMLU-Pro
)

# Методы
cv.vector          # Dict[str, float] — вектор компетенций
cv[agent_id]       # float — значение для агента
cv.to_dict()       # Экспорт в словарь
```

### AgentState

```python
state = AgentState(
    messages: List = [],
    subtask_contributions: Dict[str, int] = {},
    total_tokens: int = 0,
    productive_actions: int = 0,
    env_observation: str = "",
    is_done: bool = False,
    total_reward: float = 0.0,
    steps_taken: int = 0,
    routing_decision: Optional[str] = None
)

state.to_dict()           # Экспорт
AgentState.from_dict(d)   # Импорт
```

### MetricsCalculator

```python
calc = MetricsCalculator(
    competence_vector: CompetenceVector,
    max_steps: int = 20,
    max_possible_reward: float = 8.5
)

# Методы
calc.calculate(state, num_orders=3)  # Из AgentState
calc.calculate_from_raw(...)          # Из сырых данных
```

### Функции расчёта

```python
from src.metrics.formulas import (
    calculate_E_norm,
    calculate_C_eff,
    calculate_A_role,
    calculate_Q_coord
)

# E_norm
E = calculate_E_norm(
    total_reward: float,
    steps_taken: int,
    max_steps: int,
    max_possible_reward: float = 8.5,
    is_done: bool = True,
    alpha: float = 0.7
)

# C_eff и rho
C_eff, rho = calculate_C_eff(
    total_tokens: int,
    productive_actions: int,
    phi_max: int = 2_000_000,
    lambda_: float = 1.0
)

# A_role
A_role, w_norm = calculate_A_role(
    subtask_contributions: Dict[str, int],
    competence_vector: Dict[str, float]
)

# Q_coord
Q = calculate_Q_coord(
    E_norm: float,
    C_eff_norm: float,
    A_role: float,
    R_avg: float = 1.0,
    weights: Dict[str, float] = None
)
```

---

## Продвинутые сценарии

### Сценарий 1: Кастомные веса Q_coord

```python
# Бизнес-кейс: Критична скорость, не важна стоимость
weights = {
    "E_norm": 0.6,   # Скорость успешности
    "C_eff": 0.0,    # Игнорируем стоимость
    "R_avg": 0.2,
    "A_role": 0.2,
}

Q = calculate_Q_coord(
    E_norm=0.8,
    C_eff_norm=0.5,
    A_role=0.7,
    weights=weights
)
```

### Сценарий 2: Мониторинг в реальном времени

```python
class CoordinationMonitor:
    """Мониторинг координации в процессе работы."""
    
    def __init__(self, competence, window_size=10):
        self.competence = competence
        self.window_size = window_size
        self.history = []
    
    def update(self, state):
        """Обновить метрики."""
        calculator = MetricsCalculator(self.competence)
        metrics = calculator.calculate(state)
        
        self.history.append(metrics.Q_coord)
        
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        return metrics
    
    def get_trend(self):
        """Получить тренд качества."""
        if len(self.history) < 2:
            return "Недостаточно данных"
        
        avg_recent = sum(self.history[-5:]) / min(5, len(self.history))
        avg_older = sum(self.history[:-5]) / max(1, len(self.history) - 5)
        
        if avg_recent > avg_older * 1.1:
            return "Улучшение"
        elif avg_recent < avg_older * 0.9:
            return "Ухудшение"
        else:
            return "Стабильно"
```

### Сценарий 3: Автоматическая оптимизация

```python
def optimize_agent_roles(models_config, metrics_history):
    """Оптимизация распределения ролей на основе истории."""
    
    # Анализ вкладов
    avg_contributions = {}
    for m in metrics_history:
        for agent, contrib in m.w_norm.items():
            avg_contributions[agent] = avg_contributions.get(agent, 0) + contrib
    
    for agent in avg_contributions:
        avg_contributions[agent] /= len(metrics_history)
    
    # Сравнение с компетенциями
    competence = CompetenceVector(models_config)
    
    suggestions = []
    for agent in models_config:
        actual = avg_contributions.get(agent, 0)
        expected = competence[agent]
        
        if actual > expected * 1.2:
            suggestions.append(
                f"{agent}: перегружен (делает {actual:.1%}, ожидается {expected:.1%})"
            )
        elif actual < expected * 0.8:
            suggestions.append(
                f"{agent}: недогружен (делает {actual:.1%}, ожидается {expected:.1%})"
            )
    
    return suggestions
```

---

## Интеграция с LangGraph

### Полный пример с топологией

```python
import asyncio
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from src.config import ModelProfile, CompetenceVector
from src.core import AgentState
from src.metrics import MetricsCalculator
from src.environment import SemanticSupplyChainEnv


async def coordination_node(state: AgentState, env, llm, agent_id):
    """Узел координации для LangGraph."""
    
    # Вызов LLM
    response = await llm.ainvoke([
        {"role": "system", "content": f"You are {agent_id}"},
        {"role": "user", "content": state.env_observation}
    ])
    
    # Парсинг действия
    import json
    action = json.loads(response.content)
    
    # Выполнение в среде
    obs, reward, done, productive = env.step(agent_id, action)
    
    # Обновление состояния
    new_contrib = state.subtask_contributions.copy()
    if productive:
        new_contrib[agent_id] = new_contrib.get(agent_id, 0) + 1
    
    return AgentState(
        messages=state.messages + [response],
        subtask_contributions=new_contrib,
        total_tokens=state.total_tokens + response.response_metadata.get("token_usage", {}).get("total_tokens", 0),
        productive_actions=state.productive_actions + (1 if productive else 0),
        env_observation=obs,
        is_done=done,
        total_reward=state.total_reward + reward,
        steps_taken=state.steps_taken + 1
    )


async def run_langgraph_experiment():
    """Запуск эксперимента с LangGraph."""
    
    # Конфигурация
    models = {
        "agent1": ModelProfile("GPT-4", "Leader", 0.9, 0.85, "gpt-4"),
        "agent2": ModelProfile("GPT-3.5", "Worker", 0.7, 0.6, "gpt-3.5-turbo"),
    }
    
    competence = CompetenceVector(models)
    calculator = MetricsCalculator(competence)
    
    # Среда
    env = SemanticSupplyChainEnv(grid_size=3, max_steps=20)
    initial_obs = env.reset(list(models.keys()))
    
    # LLM
    llms = {
        k: ChatOpenAI(model=v.api_model_name, temperature=0)
        for k, v in models.items()
    }
    
    # Начальное состояние
    state = AgentState(
        subtask_contributions={k: 0 for k in models},
        env_observation=initial_obs
    )
    
    # Симуляция (упрощённая)
    for step in range(20):
        for agent_id in models:
            state = await coordination_node(state, env, llms[agent_id], agent_id)
            if state.is_done:
                break
        if state.is_done:
            break
    
    # Метрики
    metrics = calculator.calculate(state)
    
    print(f"Q_coord: {metrics.Q_coord}")
    print(f"E_norm: {metrics.E_norm}")
    print(f"A_role: {metrics.A_role}")


asyncio.run(run_langgraph_experiment())
```

---

## Частые вопросы

### Q: Где взять mmlu_pro и agentbench для моей модели?

**A:** Несколько источников:
- [OpenLLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [MMLU-Pro Benchmark](https://github.com/TIGER-AI-Lab/MMLU-Pro)
- [AgentBench](https://github.com/THUDM/AgentBench)

Если нет данных — используйте экспертную оценку (0-1).

### Q: Как определить max_possible_reward?

**A:** Для SemanticSupplyChainEnv:
```
max_reward = n_orders × (0.5 + 0.5 + 2.0) + 1.0
           = 3 × 3.0 + 1.0 = 10.0
```

Для своей среды — определите максимальную возможную награду.

### Q: Что если агентов больше 3?

**A:** Фреймворк поддерживает любое количество агентов. A_role автоматически нормализуется по n.

### Q: Можно ли использовать без LangGraph?

**A:** Да, фреймворк независим. Просто собирайте данные в AgentState и вызывайте calculator.calculate().

---

## Troubleshooting

### Ошибка: "mmlu_pro must be in [0, 1]"

**Проблема:** Значение benchmark вне диапазона.

**Решение:** Нормализуйте benchmark scores:
```python
# Если benchmark в процентах (0-100)
mmlu_pro = benchmark_score / 100
```

### Q_coord всегда низкий

**Диагностика:**
```python
print(f"E_norm: {metrics.E_norm}")  # Низкий = медленно или мало награды
print(f"C_eff: {metrics.C_eff_norm}")  # Высокий = много токенов впустую
print(f"A_role: {metrics.A_role}")  # Низкий = дисбаланс работы
```

### Не работает импорт

**Решение:**
```python
# Добавьте в начало скрипта
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
```

---

## Заключение

Этот фреймворк предоставляет научную основу для оценки координации LLM-агентов. Используйте его для:

- Сравнения архитектур мультиагентных систем
- Оптимизации распределения ролей
- Мониторинга качества командной работы AI

**Успешных экспериментов!**
