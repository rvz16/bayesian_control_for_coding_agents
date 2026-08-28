# Bayesian DP Planner: адаптация example.py на реальную задачу

## Что это

`bayesian_dp.py` — адаптация POMDP контроллера из `article_implementation/bayesian/example.py` на задачу выбора кларифицирующих вопросов в tool-calling агенте SAGE.

В `example.py` агент диагностирует баг (3 гипотезы, 3 теста, фиксированные стоимости). Здесь та же структура, но на реальных объектах: кандидаты tool-call, вопросы пользователю, домены параметров.

## Маппинг example.py → bayesian_dp.py

| example.py | bayesian_dp.py | Пример |
|---|---|---|
| Гипотезы H = {A, B, C} | Кандидаты tool-call | `book_flight(origin=BOS)`, `book_flight(origin=LAX)` |
| Тесты T = {T1, T2, T3} | Кларифицирующие вопросы | "Из какого города?", "Когда?" |
| P(fail\|H,T) — матрица 3×3 | `_compute_resolution_probability()` | P(вопрос резолвит аспект \| кандидаты, домен) |
| `compute_posterior(belief, test, outcome)` | `_simulate_question_outcome(belief, question, resolves)` | Обновление belief после ответа |
| `get_terminal_value(belief)` = R·max(b) - C_PATCH - C_VER | `get_terminal_value(probs, cost_model)` = R·max(p) - C_exec | Ценность "выполнить сейчас" |
| `get_optimal_policy_value(belief, depth)` | `_dp_value(belief, candidates, probs, questions, budget, cost_model)` | DP рекурсия с Bellman equation |
| C_TEST = 1 (одинаковая для всех тестов) | `compute_question_cost(question)` — зависит от сложности | Простой вопрос = 0.02, сложный = 0.07 |
| C_PATCH = 3, C_VER = 20 | `execution_cost` | Цена ошибочного tool-call |
| R = 200 | `reward` | Награда за правильное выполнение |

## Модель стоимости

В `example.py` все тесты стоят одинаково (`C_TEST=1`). Здесь стоимость вопроса зависит от его сложности:

```
C_q = base + aspect_cost × (n_aspects - 1) + open_domain_cost × n_open_domains
```

- "Из какого города?" — 1 аспект, finite домен → дешёвый (0.02)
- "Когда лететь?" — 1 аспект, open домен → дороже (0.04)
- "Откуда и когда?" — 2 аспекта, 1 open → самый дорогой (0.07)

Terminal value (ценность "выполнить сейчас без вопросов"):

```
V_term = reward × max(probs) - execution_cost
```

Три пресета:

| Пресет | Вопросы | C_exec | Когда использовать |
|---|---|---|---|
| `DEFAULT_COST_MODEL` | средние | 0.15 | общий случай |
| `HIGH_STAKES_COST_MODEL` | дешёвые | 0.40 | бронирование, платёж, удаление |
| `LOW_STAKES_COST_MODEL` | дорогие | 0.03 | поиск, lookup, чтение |

**Эффект**: при HIGH_STAKES агент задаёт больше вопросов (ошибка дорогая, вопросы дешёвые). При LOW_STAKES — быстрее выполняет (ошибку легко исправить).

## Структура кода

### Функции (порядок вызова)

```
compute_question_value_dp()          ← точка входа, вызывается из agent.py
  ├── compute_question_cost()        ← стоимость вопроса (аналог C_TEST)
  ├── _compute_resolution_probability()  ← P(resolve|b,q) (аналог P(fail|b,T))
  ├── _simulate_question_outcome()   ← posteriors (аналог compute_posterior)
  └── _dp_value()                    ← DP рекурсия (аналог get_optimal_policy_value)
        ├── get_terminal_value()     ← V_term (аналог get_terminal_value)
        ├── compute_question_cost()
        ├── _compute_resolution_probability()
        ├── _simulate_question_outcome()
        └── _dp_value()  (рекурсия, depth-1)
```

### Bellman equation

```
# example.py:
V(b, d) = max_T [ -C_TEST + P(fail|b,T)·V(b_fail, d-1) + P(pass|b,T)·V(b_pass, d-1) ]

# bayesian_dp.py:
V(b, d) = max_q [ -C_q + P(resolve|b,q)·V(b_resolve, d-1) + P(~resolve|b,q)·V(b_no, d-1) ]
```

Структура идентична. Разница: C_q зависит от вопроса, а P(resolve) вычисляется динамически из belief state (не из фиксированной матрицы).

## Пример работы

Сценарий: пользователь сказал "Забронируй рейс в Нью-Йорк". Агент знает destination=NYC, но не знает origin и date.

```python
from sage_agent.core.bayesian_dp import (
    compute_question_value_dp, get_terminal_value,
    HIGH_STAKES_COST_MODEL, LOW_STAKES_COST_MODEL,
)

# При уверенности 0.33 (три равновероятных кандидата):
#
# HIGH_STAKES:  V_term = -0.067 (выполнить = убыток!)
#   → агент спрашивает "Откуда и когда?" (сложный, но дешёвый вопрос)
#
# LOW_STAKES:   V_term = 0.303 (выполнить уже ок)
#   → агент спрашивает "Из какого города?" (простой дешёвый вопрос)
#
# При уверенности 0.90:
#   → оба пресета: ВЫПОЛНИТЬ (score всех вопросов < 0)
```

## Интеграция с агентом

```python
from sage_agent import SageAgent, SageAgentConfig
from sage_agent.core.bayesian_dp import HIGH_STAKES_COST_MODEL

agent = SageAgent(
    config=SageAgentConfig(
        use_dp_planning=True,          # включить DP вместо myopic EVPI
        dp_cost_model=HIGH_STAKES_COST_MODEL,  # модель стоимости
    ),
    ...
)
```

Когда `use_dp_planning=False` (по умолчанию), агент использует одношаговый EVPI. Когда `True` — DP с lookahead на оставшийся бюджет вопросов.

## Тесты

```bash
pytest tests/test_bayesian_dp.py -v
```

17 тестов: terminal value, resolution probability, simulate outcome, DP recursion, полная интеграция с SageAgent, сравнение DP vs myopic.
