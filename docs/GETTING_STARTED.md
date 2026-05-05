# Руководство пользователя LLM-MAS Coordination Metrics

## Для кого это руководство?

Это руководство для тех, кто хочет:
- Использовать фреймворк для оценки своих мультиагентных систем
- Подключить свои LLM-модели (OpenAI, Anthropic, локальные)
- Создать свою среду для тестирования
- Получить метрики качества координации

**Не требуется глубокое знание Python** — все примеры готовы к запуску.

---

## Оглавление

1. [Быстрый старт за 5 минут](#быстрый-старт-за-5-минут)
2. [Как подключить свой API](#как-подключить-свой-api)
3. [Как создать своих агентов](#как-создать-своих-агентов)
4. [Как создать свою среду](#как-создать-свою-среду)
5. [Как запустить эксперимент](#как-запустить-эксперимент)
6. [Как читать результаты](#как-читать-результаты)
7. [Примеры из реальной жизни](#примеры-из-реальной-жизни)

---

## Быстрый старт за 5 минут

### Шаг 1: Установка

Откройте терминал и выполните:

```bash
git clone https://github.com/yourusername/llm-mas-coordination-metrics.git
cd llm-mas-coordination-metrics
pip install -e .
```

### Шаг 2: Первый запуск

Создайте файл `my_first_experiment.py`:

```python
from src.config import ModelProfile, CompetenceVector
from src.metrics import MetricsCalculator

# Определяем двух агентов
models = {
    "agent_1": ModelProfile(
        name="Мой первый агент",
        role="Лидер",
        mmlu_pro=0.8,      # Можно поставить примерное значение
        agentbench=0.75,
        api_model_name="gpt-4"  # Ваша модель
    ),
    "agent_2": ModelProfile(
        name="Мой второй агент",
        role="Исполнитель",
        mmlu_pro=0.6,
        agentbench=0.55,
        api_model_name="gpt-3.5-turbo"
    ),
}

# Создаём калькулятор
competence = CompetenceVector(models)
calculator = MetricsCalculator(competence)

# Симулируем результаты работы
metrics = calculator.calculate_from_raw(
    total_reward=7.5,        # Сколько награды получили
    steps_taken=15,          # Сколько шагов сделали
    total_tokens=4000,       # Сколько токенов потратили
    productive_actions=8,    # Сколько действий было полезным
    subtask_contributions={  # Кто сколько сделал
        "agent_1": 5,
        "agent_2": 3
    },
    is_done=True
)

# Смотрим результат
print(f"Качество координации: {metrics.Q_coord:.2f}")
print(f"Эффективность: {metrics.E_norm:.2f}")
print(f"Ролевое соответствие: {metrics.A_role:.2f}")
```

Запустите:
```bash
python my_first_experiment.py
```

### Шаг 3: Результат

Вы увидите:
```
Качество координации: 0.82
Эффективность: 0.68
Ролевое соответствие: 0.91
```

**Поздравляем!** Вы только что оценили координацию мультиагентной системы.

---

## Как подключить свой API

### Вариант 1: OpenAI

```python
import os
from langchain_openai import ChatOpenAI

# Способ 1: Через переменную окружения
os.environ["OPENAI_API_KEY"] = "sk-ваш-ключ-здесь"

# Способ 2: Напрямую в коде
llm = ChatOpenAI(
    model="gpt-4",
    api_key="sk-ваш-ключ-здесь"
)
```

### Вариант 2: Anthropic (Claude)

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-3-opus-20240229",
    api_key="ваш-anthropic-ключ"
)
```

### Вариант 3: Локальные модели (Ollama)

```python
from langchain_community.llms import Ollama

llm = Ollama(model="llama3")
```

### Вариант 4: Российские провайдеры (Vsegpt, GigaChat и др.)

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek/deepseek-chat",  # Имя модели у провайдера
    base_url="https://api.vsegpt.ru/v1",  # URL провайдера
    api_key="ваш-ключ"
)
```

### Вариант 5: Свой API сервер

```python
llm = ChatOpenAI(
    model="custom-model",
    base_url="http://localhost:8000/v1",
    api_key="не-нужен"  # Можно оставить пустым
)
```

### Где хранить ключи безопасно?

**НЕ ХРАНИТЕ ключи в коде!** Используйте `.env` файл:

1. Создайте файл `.env` в корне проекта:
```
OPENAI_API_KEY=sk-ваш-ключ
ANTHROPIC_API_KEY=ваш-ключ
```

2. Добавьте `.env` в `.gitignore` (уже добавлено)

3. Загружайте в коде:
```python
from dotenv import load_dotenv
load_dotenv()  # Загружает из .env файла

import os
api_key = os.environ["OPENAI_API_KEY"]
```

---

## Как создать своих агентов

### Что такое агент в этом фреймворке?

Агент = модель + роль + компетенции

### Шаг 1: Определите компетенции

Компетенции определяются двумя числами:
- **mmlu_pro** — способность решать сложные задачи (0-1)
- **agentbench** — способность действовать как агент (0-1)

Где взять эти числа?

| Источник | Как найти |
|----------|-----------|
| [OpenLLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) | Таблица с бенчмарками |
| [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro) | Официальный бенчмарк |
| Документация модели | Обычно указаны бенчмарки |

**Если не нашли** — используйте экспертную оценку:

```python
# Примерные значения для популярных моделей
MODEL_COMPETENCES = {
    "GPT-4":          {"mmlu_pro": 0.90, "agentbench": 0.92},
    "GPT-4-Turbo":    {"mmlu_pro": 0.92, "agentbench": 0.94},
    "GPT-3.5-Turbo":  {"mmlu_pro": 0.65, "agentbench": 0.60},
    "Claude-3-Opus":  {"mmlu_pro": 0.88, "agentbench": 0.90},
    "Claude-3-Sonnet":{"mmlu_pro": 0.75, "agentbench": 0.72},
    "Llama-3-70B":    {"mmlu_pro": 0.78, "agentbench": 0.75},
    "Mistral-Large":  {"mmlu_pro": 0.72, "agentbench": 0.70},
    "DeepSeek-V3":    {"mmlu_pro": 0.85, "agentbench": 0.88},
}
```

### Шаг 2: Создайте команду агентов

```python
from src.config import ModelProfile, CompetenceVector

# Команда для анализа данных
analytics_team = {
    "analyst": ModelProfile(
        name="GPT-4",
        role="Аналитик данных",
        mmlu_pro=0.90,
        agentbench=0.92,
        api_model_name="gpt-4"
    ),
    "visualizer": ModelProfile(
        name="Claude-3-Sonnet",
        role="Визуализатор",
        mmlu_pro=0.75,
        agentbench=0.72,
        api_model_name="claude-3-sonnet"
    ),
    "validator": ModelProfile(
        name="GPT-3.5-Turbo",
        role="Проверяющий",
        mmlu_pro=0.65,
        agentbench=0.60,
        api_model_name="gpt-3.5-turbo"
    ),
}

competence = CompetenceVector(analytics_team)
print("Распределение компетенций:", competence.vector)
```

### Шаг 3: Настройте веса под свою задачу

```python
# Если важна скорость, а не качество
competence = CompetenceVector(analytics_team, beta=0.3)  # Меньше веса на сложность

# Если важно качество решений
competence = CompetenceVector(analytics_team, beta=0.8)  # Больше веса на сложность
```

---

## Как создать свою среду

### Что такое среда?

Среда — это задача, которую решают агенты. Фреймворк уже включает `SemanticSupplyChainEnv` для логистики, но вы можете создать свою.

### Пример: Среда для код-ревью

```python
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class CodeReviewState:
    """Состояние код-ревью."""
    code_submitted: bool = False
    review_done: bool = False
    fixes_applied: bool = False
    total_reward: float = 0.0

class CodeReviewEnv:
    """
    Среда для симуляции код-ревью.
    
    Действия:
    - submit_code: отправить код на ревью
    - review: провести ревью
    - apply_fixes: применить исправления
    """
    
    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps
        self.state = CodeReviewState()
        self.step_count = 0
    
    def reset(self, agent_ids):
        """Сбросить среду."""
        self.state = CodeReviewState()
        self.step_count = 0
        return "Код готов к ревью. Действия: submit_code, review, apply_fixes"
    
    def step(self, agent_id: str, action: Dict) -> Tuple[str, float, bool, bool]:
        """
        Выполнить действие.
        
        Возвращает: (наблюдение, награда, завершено, полезно)
        """
        action_type = action.get("type", "wait")
        reward = 0.0
        is_productive = False
        
        if action_type == "submit_code" and not self.state.code_submitted:
            self.state.code_submitted = True
            reward = 0.5
            is_productive = True
            obs = "Код отправлен на ревью"
        
        elif action_type == "review" and self.state.code_submitted and not self.state.review_done:
            self.state.review_done = True
            reward = 0.5
            is_productive = True
            obs = "Ревью завершено. Найдены проблемы"
        
        elif action_type == "apply_fixes" and self.state.review_done and not self.state.fixes_applied:
            self.state.fixes_applied = True
            reward = 2.0
            is_productive = True
            obs = "Исправления применены. Код одобрен!"
        
        else:
            obs = "Действие не выполнено"
            reward = -0.1
        
        self.step_count += 1
        done = self.state.fixes_applied or self.step_count >= self.max_steps
        
        if done and self.state.fixes_applied:
            reward += 1.0  # Бонус за завершение
        
        return obs, reward, done, is_productive
    
    def is_complete(self):
        """Проверить завершение."""
        return self.state.fixes_applied
```

### Пример: Среда для customer service

```python
class CustomerServiceEnv:
    """
    Среда для обслуживания клиентов.
    
    Действия:
    - greet: приветствие
    - understand: понять проблему
    - solve: решить проблему
    - close: закрыть обращение
    """
    
    def __init__(self, max_steps: int = 15):
        self.max_steps = max_steps
        self.reset([])
    
    def reset(self, agent_ids):
        self.stage = "greeting"
        self.step_count = 0
        self.total_reward = 0.0
        return "Клиент обращается в службу поддержки"
    
    def step(self, agent_id: str, action: Dict) -> Tuple[str, float, bool, bool]:
        action_type = action.get("type", "wait")
        reward = 0.0
        is_productive = False
        
        transitions = {
            ("greeting", "greet"): ("understanding", 0.3, True),
            ("understanding", "understand"): ("solving", 0.5, True),
            ("solving", "solve"): ("closing", 1.0, True),
            ("closing", "close"): ("done", 1.5, True),
        }
        
        key = (self.stage, action_type)
        if key in transitions:
            self.stage, reward, is_productive = transitions[key]
            obs = f"Переход на стадию: {self.stage}"
        else:
            obs = f"Неверное действие {action_type} на стадии {self.stage}"
            reward = -0.2
        
        self.step_count += 1
        done = self.stage == "done" or self.step_count >= self.max_steps
        
        return obs, reward, done, is_productive
```

---

## Как запустить эксперимент

### Способ 1: Без реального LLM (симуляция)

Используйте когда хотите протестировать метрики:

```python
from src.config import ModelProfile, CompetenceVector
from src.metrics import MetricsCalculator

models = {
    "agent_1": ModelProfile("Agent1", "Leader", 0.8, 0.75, "model-1"),
    "agent_2": ModelProfile("Agent2", "Worker", 0.6, 0.55, "model-2"),
}

competence = CompetenceVector(models)
calculator = MetricsCalculator(competence)

# Вводите свои данные
metrics = calculator.calculate_from_raw(
    total_reward=8.0,           # Из логов вашей системы
    steps_taken=12,             # Из логов
    total_tokens=5000,          # Из логов
    productive_actions=9,       # Посчитайте вручную
    subtask_contributions={     # Из логов
        "agent_1": 5,
        "agent_2": 4
    },
    is_done=True
)

print(f"Q_coord = {metrics.Q_coord}")
```

### Способ 2: С реальным LLM через LangChain

```python
import asyncio
from langchain_openai import ChatOpenAI
from src.config import ModelProfile, CompetenceVector
from src.core import AgentState
from src.metrics import MetricsCalculator

async def run_with_real_llm():
    # 1. Конфигурация
    models = {
        "planner": ModelProfile("GPT-4", "Планировщик", 0.9, 0.85, "gpt-4"),
        "worker": ModelProfile("GPT-3.5", "Исполнитель", 0.65, 0.6, "gpt-3.5-turbo"),
    }
    
    competence = CompetenceVector(models)
    calculator = MetricsCalculator(competence)
    
    # 2. Создаём LLM
    llms = {
        agent_id: ChatOpenAI(model=profile.api_model_name, temperature=0)
        for agent_id, profile in models.items()
    }
    
    # 3. Инициализируем состояние
    state = AgentState(
        subtask_contributions={aid: 0 for aid in models},
        total_tokens=0,
        productive_actions=0
    )
    
    # 4. Запускаем симуляцию
    for step in range(20):
        for agent_id, llm in llms.items():
            # Вызываем LLM
            response = await llm.ainvoke([
                {"role": "user", "content": "Ваш промпт здесь"}
            ])
            
            # Считаем токены
            tokens = response.response_metadata.get("token_usage", {}).get("total_tokens", 0)
            state.total_tokens += tokens
            
            # Ваша логика обработки ответа
            # ...
            
            # Если действие полезное:
            # state.productive_actions += 1
            # state.subtask_contributions[agent_id] += 1
        
        if state.is_done:
            break
    
    # 5. Считаем метрики
    metrics = calculator.calculate(state)
    return metrics

metrics = asyncio.run(run_with_real_llm())
print(f"Q_coord = {metrics.Q_coord}")
```

### Способ 3: Полный пример с LangGraph

См. файл `examples/custom_environment.py` в репозитории.

---

## Как читать результаты

### Основная метрика: Q_coord

| Значение | Что значит | Действие |
|----------|------------|----------|
| **0.8 - 1.0** | Отлично | Система работает оптимально |
| **0.6 - 0.8** | Хорошо | Есть пространство для улучшений |
| **0.4 - 0.6** | Удовлетворительно | Требуется анализ проблем |
| **< 0.4** | Плохо | Нужно пересмотреть архитектуру |

### Компоненты метрики

#### E_norm (Эффективность)

**Высокий (>0.7):**
- Задачи выполняются быстро
- Высокий success rate

**Низкий (<0.5):**
- Слишком много шагов
- Низкая награда
- Проверьте: правильно ли декомпозируются задачи?

#### C_eff (Стоимость)

**Высокий (>0.5):**
- Много токенов тратится впустую
- Агенты делают много ненужных действий
- Проверьте: оптимизируйте промпты

**Низкий (<0.3):**
- Токены используются эффективно
- Хорошее соотношение действия/токены

#### A_role (Ролевое соответствие)

**Высокий (>0.8):**
- Работа распределена по компетенциям
- Сильные агенты делают сложную работу

**Низкий (<0.5):**
- Дисбаланс в работе
- Проверьте: правильно ли назначены роли?

### Пример анализа

```python
def analyze_metrics(metrics, competence):
    print("=== АНАЛИЗ КООРДИНАЦИИ ===\n")
    
    # 1. Общая оценка
    if metrics.Q_coord > 0.7:
        print("✓ Система работает хорошо\n")
    else:
        print("✗ Требуется оптимизация\n")
    
    # 2. Эффективность
    print(f"Эффективность (E_norm = {metrics.E_norm:.2f})")
    if metrics.E_norm < 0.5:
        print("  Рекомендация: упростите процесс принятия решений")
    
    # 3. Стоимость
    print(f"\nСтоимость токенов (C_eff = {metrics.C_eff_norm:.2f})")
    if metrics.C_eff_norm > 0.5:
        print("  Рекомендация: сократите "пустые" действия")
    
    # 4. Роли
    print(f"\nРолевое соответствие (A_role = {metrics.A_role:.2f})")
    print(f"  Ожидаемое распределение: {competence.vector}")
    print(f"  Фактическое: {metrics.w_norm}")
    
    if metrics.A_role < 0.7:
        print("  Рекомендация: перераспределите задачи по компетенциям")
```

---

## Примеры из реальной жизни

### Пример 1: Система обработки заказов

```python
# Команда для обработки заказов интернет-магазина
order_processing_team = {
    "categorizer": ModelProfile(
        name="GPT-3.5-Turbo",
        role="Классификатор",
        mmlu_pro=0.65,
        agentbench=0.70,
        api_model_name="gpt-3.5-turbo"
    ),
    "processor": ModelProfile(
        name="GPT-4",
        role="Обработчик",
        mmlu_pro=0.90,
        agentbench=0.92,
        api_model_name="gpt-4"
    ),
    "validator": ModelProfile(
        name="Claude-3-Sonnet",
        role="Проверяющий",
        mmlu_pro=0.75,
        agentbench=0.72,
        api_model_name="claude-3-sonnet"
    ),
}

# После месяца работы получили данные:
metrics = calculator.calculate_from_raw(
    total_reward=156.0,     # 156 успешно обработанных заказов
    steps_taken=450,        # В среднем 3 шага на заказ
    total_tokens=180000,    # 180K токенов за месяц
    productive_actions=420, # 420 полезных действий
    subtask_contributions={
        "categorizer": 156,  # Все заказы классифицированы
        "processor": 156,    # Все обработаны
        "validator": 108,    # 108 проверено (не все)
    },
    is_done=True
)

print(f"Q_coord = {metrics.Q_coord}")
# Результат: Q_coord = 0.79 (хорошо!)
```

### Пример 2: Система генерации контента

```python
content_team = {
    "researcher": ModelProfile(
        name="Perplexity",
        role="Исследователь",
        mmlu_pro=0.70,
        agentbench=0.80,
        api_model_name="perplexity-online"
    ),
    "writer": ModelProfile(
        name="GPT-4",
        role="Писатель",
        mmlu_pro=0.90,
        agentbench=0.85,
        api_model_name="gpt-4"
    ),
    "editor": ModelProfile(
        name="Claude-3-Opus",
        role="Редактор",
        mmlu_pro=0.88,
        agentbench=0.82,
        api_model_name="claude-3-opus"
    ),
}

# Данные за неделю
metrics = calculator.calculate_from_raw(
    total_reward=23.5,      # 23 статьи + бонусы за качество
    steps_taken=150,        # ~6 шагов на статью
    total_tokens=85000,
    productive_actions=92,
    subtask_contributions={
        "researcher": 35,   # Исследования
        "writer": 28,       # Написание
        "editor": 29,       # Редактирование
    },
    is_done=True
)
```

### Пример 3: Анализ тональности отзывов

```python
sentiment_team = {
    "collector": ModelProfile(
        name="GPT-3.5",
        role="Сборщик",
        mmlu_pro=0.55,
        agentbench=0.60,
        api_model_name="gpt-3.5-turbo"
    ),
    "analyzer": ModelProfile(
        name="GPT-4",
        role="Аналитик",
        mmlu_pro=0.90,
        agentbench=0.88,
        api_model_name="gpt-4"
    ),
}

# После обработки 100 отзывов
metrics = calculator.calculate_from_raw(
    total_reward=45.0,      # 90% точность
    steps_taken=200,
    total_tokens=25000,
    productive_actions=180,
    subtask_contributions={
        "collector": 100,   # Собрал все
        "analyzer": 80,     # Проанализировал успешно 80
    },
    is_done=True
)

# Анализ
if metrics.A_role < 0.7:
    print("Внимание: аналитик перегружен!")
    print("Рекомендация: добавить ещё одного аналитика")
```

---

## Шпаргалка

### Минимальный код для запуска

```python
from src.config import ModelProfile, CompetenceVector
from src.metrics import MetricsCalculator

models = {
    "agent": ModelProfile("Name", "Role", 0.7, 0.7, "model"),
}

calculator = MetricsCalculator(CompetenceVector(models))
metrics = calculator.calculate_from_raw(
    total_reward=5.0,
    steps_taken=10,
    total_tokens=1000,
    productive_actions=5,
    subtask_contributions={"agent": 5},
    is_done=True
)

print(f"Q_coord = {metrics.Q_coord:.2f}")
```

### Где искать значения mmlu_pro?

1. **HuggingFace Open LLM Leaderboard** — самый полный источник
2. **Документация модели** — обычно в README на GitHub
3. **Papers with Code** — бенчмарки с результатами
4. **Примерные значения** — используйте таблицу выше

### Что делать, если Q_coord низкий?

1. Проверьте **E_norm** — может слишком много шагов?
2. Проверьте **C_eff** — может много пустых токенов?
3. Проверьте **A_role** — может неправильное распределение работы?

---

**Удачи в использовании фреймворка!**

Если остались вопросы — откройте Issue на GitHub.
