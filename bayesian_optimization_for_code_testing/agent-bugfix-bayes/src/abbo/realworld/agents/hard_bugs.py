"""Hard bug definitions for demonstrating Bayesian orchestration advantage.

These bugs are based on patterns from real open-source issues (CPython, Flask,
requests, etc.) and are designed to be challenging for small LLMs (~7B params).

Key properties:
- Bugs require understanding data flow across multiple functions
- Test output doesn't directly point to the broken line
- Fixes may require changing multiple lines or understanding invariants
- 7B models succeed ~30-60% of the time (vs ~95% on easy router bugs)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Module 1: Task Scheduler with dependency resolution
# ---------------------------------------------------------------------------

SCHEDULER_CLEAN_SOURCE = '''"""Task scheduler with dependency resolution and cycle detection."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Task:
    """A schedulable task."""
    name: str
    duration: int = 1
    priority: int = 0
    deps: list[str] = field(default_factory=list)


class SchedulerError(Exception):
    """Raised on scheduling failures."""


class TaskScheduler:
    """Priority-aware task scheduler with topological ordering."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def add_task(self, task: Task) -> None:
        """Register a task."""
        self._tasks[task.name] = task

    def _build_graph(self) -> tuple[dict[str, list[str]], dict[str, int]]:
        """Build adjacency list and in-degree map."""
        graph: dict[str, list[str]] = defaultdict(list)
        in_degree: dict[str, int] = {name: 0 for name in self._tasks}

        for name, task in self._tasks.items():
            for dep in task.deps:
                if dep not in self._tasks:
                    raise SchedulerError(f"Unknown dependency: {dep}")
                graph[dep].append(name)
                in_degree[name] += 1

        return dict(graph), in_degree

    def detect_cycle(self) -> list[str] | None:
        """Return a cycle if one exists, else None."""
        visited: set[str] = set()
        rec_stack: set[str] = set()
        path: list[str] = []

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self._get_dependents(node):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle - extract it
                    cycle_start = path.index(neighbor)
                    path.append(neighbor)
                    return True

            path.pop()
            rec_stack.remove(node)
            return False

        for name in self._tasks:
            if name not in visited:
                if dfs(name):
                    return path

        return None

    def _get_dependents(self, name: str) -> list[str]:
        """Get tasks that depend on the given task."""
        return [
            t.name for t in self._tasks.values()
            if name in t.deps
        ]

    def schedule(self) -> list[str]:
        """Return tasks in valid execution order.

        Uses Kahn's algorithm with priority-based tie-breaking.
        Higher priority tasks are scheduled first among ready tasks.
        Raises SchedulerError if a cycle is detected.
        """
        cycle = self.detect_cycle()
        if cycle is not None:
            raise SchedulerError(f"Dependency cycle: {' -> '.join(cycle)}")

        graph, in_degree = self._build_graph()

        # Priority queue: tasks with no dependencies, sorted by priority (desc)
        ready = sorted(
            [name for name, deg in in_degree.items() if deg == 0],
            key=lambda n: self._tasks[n].priority,
            reverse=True,
        )

        order: list[str] = []
        while ready:
            task_name = ready.pop(0)  # highest priority first
            order.append(task_name)

            for dependent in graph.get(task_name, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    # Insert in priority order
                    inserted = False
                    for i, r in enumerate(ready):
                        if self._tasks[dependent].priority > self._tasks[r].priority:
                            ready.insert(i, dependent)
                            inserted = True
                            break
                    if not inserted:
                        ready.append(dependent)

        if len(order) != len(self._tasks):
            raise SchedulerError("Could not schedule all tasks")

        return order

    def critical_path(self) -> tuple[list[str], int]:
        """Find the critical path (longest path by duration).

        Returns (path_nodes, total_duration).
        """
        order = self.schedule()

        # Dynamic programming: longest path to each node
        dist: dict[str, int] = {}
        pred: dict[str, str | None] = {}

        for name in order:
            task = self._tasks[name]
            if not task.deps:
                dist[name] = task.duration
                pred[name] = None
            else:
                best_dep = max(task.deps, key=lambda d: dist.get(d, 0))
                dist[name] = dist[best_dep] + task.duration
                pred[name] = best_dep

        # Trace back from the node with max distance
        end_node = max(dist, key=lambda n: dist[n])
        path = []
        node: str | None = end_node
        while node is not None:
            path.append(node)
            node = pred[node]
        path.reverse()

        return path, dist[end_node]

    def parallel_groups(self) -> list[list[str]]:
        """Group tasks into parallel execution waves.

        Each wave contains tasks that can run simultaneously
        (all dependencies satisfied by previous waves).
        """
        order = self.schedule()
        task_wave: dict[str, int] = {}

        for name in order:
            task = self._tasks[name]
            if not task.deps:
                task_wave[name] = 0
            else:
                task_wave[name] = max(task_wave[dep] for dep in task.deps) + 1

        waves: list[list[str]] = []
        for name in order:
            wave_idx = task_wave[name]
            while len(waves) <= wave_idx:
                waves.append([])
            waves[wave_idx].append(name)

        return waves

    def earliest_start(self) -> dict[str, int]:
        """Compute earliest start time for each task."""
        order = self.schedule()
        start: dict[str, int] = {}

        for name in order:
            task = self._tasks[name]
            if not task.deps:
                start[name] = 0
            else:
                start[name] = max(
                    start[dep] + self._tasks[dep].duration
                    for dep in task.deps
                )

        return start
'''

# ---------------------------------------------------------------------------
# Hard bugs: each requires understanding the algorithm
# ---------------------------------------------------------------------------

# Bug D: Broken cycle detection — doesn't properly reset path on backtrack.
# The DFS path list grows but the pop() doesn't account for the cycle
# extraction modifying the path. This causes false positives on some graphs.
BUGGY_SOURCES_HARD = {
    # Bug D: Schedule method has wrong Kahn's termination check —
    # it checks len(order) != len(self._tasks) but the error message
    # says "cycle" even when it's just an unreachable node.
    # More importantly: the initial ready list sort is ascending instead
    # of descending, AND the in-degree decrement is off by one for
    # nodes with multiple dependencies. This is a multi-line bug.
    "bug_d": SCHEDULER_CLEAN_SOURCE.replace(
        """        ready = sorted(
            [name for name, deg in in_degree.items() if deg == 0],
            key=lambda n: self._tasks[n].priority,
            reverse=True,
        )

        order: list[str] = []
        while ready:
            task_name = ready.pop(0)  # highest priority first
            order.append(task_name)

            for dependent in graph.get(task_name, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:""",
        """        ready = sorted(
            [name for name, deg in in_degree.items() if deg == 0],
            key=lambda n: self._tasks[n].priority,
            # BUG: missing reverse=True — sorts ascending instead of descending
        )

        order: list[str] = []
        while ready:
            task_name = ready.pop(0)  # should be highest priority first, but list is wrong order
            order.append(task_name)

            for dependent in graph.get(task_name, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:"""
    ),

    # Bug E: Priority sorting is inverted in the ready queue insertion.
    # When a new task becomes ready, it's inserted at the wrong position.
    "bug_e": SCHEDULER_CLEAN_SOURCE.replace(
        """                    for i, r in enumerate(ready):
                        if self._tasks[dependent].priority > self._tasks[r].priority:
                            ready.insert(i, dependent)
                            inserted = True
                            break""",
        """                    for i, r in enumerate(ready):
                        if self._tasks[dependent].priority < self._tasks[r].priority:  # BUG: inverted comparison
                            ready.insert(i, dependent)
                            inserted = True
                            break"""
    ),

    # Bug F: Critical path DP uses wrong recurrence — adds duration twice
    # for the predecessor, causing wrong total duration.
    "bug_f": SCHEDULER_CLEAN_SOURCE.replace(
        """                best_dep = max(task.deps, key=lambda d: dist.get(d, 0))
                dist[name] = dist[best_dep] + task.duration
                pred[name] = best_dep""",
        """                best_dep = max(task.deps, key=lambda d: dist.get(d, 0))
                dist[name] = dist[best_dep] + self._tasks[best_dep].duration  # BUG: adds dep duration instead of own
                pred[name] = best_dep"""
    ),

    # Bug G: Parallel groups off-by-one — uses dep wave directly instead of +1,
    # putting dependent tasks in the same wave as their dependencies.
    "bug_g": SCHEDULER_CLEAN_SOURCE.replace(
        """                task_wave[name] = max(task_wave[dep] for dep in task.deps) + 1""",
        """                task_wave[name] = max(task_wave[dep] for dep in task.deps)  # BUG: missing +1"""
    ),

    # Bug H: Earliest start miscalculates — uses min instead of max across deps.
    # This means a task starts when its FIRST dep finishes, not its LAST.
    "bug_h": SCHEDULER_CLEAN_SOURCE.replace(
        """                start[name] = max(
                    start[dep] + self._tasks[dep].duration
                    for dep in task.deps
                )""",
        """                start[name] = min(  # BUG: min instead of max
                    start[dep] + self._tasks[dep].duration
                    for dep in task.deps
                )"""
    ),
}


# ---------------------------------------------------------------------------
# Test suite for the scheduler
# ---------------------------------------------------------------------------

SCHEDULER_TEST_CONTENT = '''"""Tests for the task scheduler."""
import sys
sys.path.insert(0, "{src_dir}")
from scheduler import Task, TaskScheduler, SchedulerError
import pytest


@pytest.fixture
def linear_chain():
    """A -> B -> C (linear dependency chain)."""
    s = TaskScheduler()
    s.add_task(Task("A", duration=3, priority=1))
    s.add_task(Task("B", duration=2, priority=2, deps=["A"]))
    s.add_task(Task("C", duration=1, priority=3, deps=["B"]))
    return s


@pytest.fixture
def diamond():
    """Diamond dependency: A -> B, A -> C, B -> D, C -> D."""
    s = TaskScheduler()
    s.add_task(Task("A", duration=2, priority=1))
    s.add_task(Task("B", duration=3, priority=2, deps=["A"]))
    s.add_task(Task("C", duration=1, priority=3, deps=["A"]))
    s.add_task(Task("D", duration=2, priority=4, deps=["B", "C"]))
    return s


@pytest.fixture
def parallel_tasks():
    """Three independent tasks with different priorities."""
    s = TaskScheduler()
    s.add_task(Task("X", duration=1, priority=10))
    s.add_task(Task("Y", duration=2, priority=20))
    s.add_task(Task("Z", duration=3, priority=5))
    return s


@pytest.fixture
def complex_dag():
    """
    A(p=1) -> B(p=5), A -> C(p=3)
    B -> D(p=2)
    C -> D
    C -> E(p=4)
    D -> F(p=1)
    E -> F
    """
    s = TaskScheduler()
    s.add_task(Task("A", duration=2, priority=1))
    s.add_task(Task("B", duration=3, priority=5, deps=["A"]))
    s.add_task(Task("C", duration=1, priority=3, deps=["A"]))
    s.add_task(Task("D", duration=4, priority=2, deps=["B", "C"]))
    s.add_task(Task("E", duration=2, priority=4, deps=["C"]))
    s.add_task(Task("F", duration=1, priority=1, deps=["D", "E"]))
    return s


# --- Cycle detection tests ---

def test_no_cycle_linear(linear_chain):
    """Linear chain should have no cycle."""
    assert linear_chain.detect_cycle() is None


def test_no_cycle_diamond(diamond):
    """Diamond DAG should have no cycle."""
    assert diamond.detect_cycle() is None


def test_cycle_detected():
    """Direct cycle A -> B -> A should be detected."""
    s = TaskScheduler()
    s.add_task(Task("A", deps=["B"]))
    s.add_task(Task("B", deps=["A"]))
    cycle = s.detect_cycle()
    assert cycle is not None
    assert len(cycle) >= 2


def test_cycle_three_nodes():
    """Cycle A -> B -> C -> A should be detected."""
    s = TaskScheduler()
    s.add_task(Task("A", deps=["C"]))
    s.add_task(Task("B", deps=["A"]))
    s.add_task(Task("C", deps=["B"]))
    cycle = s.detect_cycle()
    assert cycle is not None


def test_no_false_positive_cycle():
    """A graph with shared deps but no cycle must return None."""
    s = TaskScheduler()
    s.add_task(Task("A"))
    s.add_task(Task("B", deps=["A"]))
    s.add_task(Task("C", deps=["A"]))
    s.add_task(Task("D", deps=["B", "C"]))
    assert s.detect_cycle() is None


# --- Scheduling order tests ---

def test_schedule_linear(linear_chain):
    """Linear chain must be scheduled in order."""
    order = linear_chain.schedule()
    assert order == ["A", "B", "C"]


def test_schedule_respects_deps(diamond):
    """A must come before B and C; B and C before D."""
    order = diamond.schedule()
    assert order.index("A") < order.index("B")
    assert order.index("A") < order.index("C")
    assert order.index("B") < order.index("D")
    assert order.index("C") < order.index("D")


def test_schedule_priority_tiebreak(parallel_tasks):
    """Independent tasks should be ordered by priority (highest first)."""
    order = parallel_tasks.schedule()
    assert order == ["Y", "X", "Z"]


def test_schedule_priority_among_ready(complex_dag):
    """When multiple tasks become ready, higher priority goes first."""
    order = complex_dag.schedule()
    # After A completes, B(p=5) and C(p=3) are ready
    # B should come before C due to higher priority
    assert order.index("A") == 0
    b_idx = order.index("B")
    c_idx = order.index("C")
    assert b_idx < c_idx, "B(p=5) should come before C(p=3), got order " + str(order)


def test_schedule_cycle_raises():
    """Scheduling a cyclic graph should raise SchedulerError."""
    s = TaskScheduler()
    s.add_task(Task("A", deps=["B"]))
    s.add_task(Task("B", deps=["A"]))
    with pytest.raises(SchedulerError, match="cycle"):
        s.schedule()


def test_schedule_all_tasks_present(complex_dag):
    """All tasks must appear in the schedule."""
    order = complex_dag.schedule()
    expected = set(["A", "B", "C", "D", "E", "F"])
    assert set(order) == expected


# --- Critical path tests ---

def test_critical_path_linear(linear_chain):
    """Critical path of a linear chain is the full chain."""
    path, duration = linear_chain.critical_path()
    assert path == ["A", "B", "C"]
    assert duration == 6  # 3 + 2 + 1


def test_critical_path_diamond(diamond):
    """Critical path through diamond goes through longest branch."""
    path, duration = diamond.critical_path()
    # A(2) -> B(3) -> D(2) = 7, longer than A(2) -> C(1) -> D(2) = 5
    assert path == ["A", "B", "D"]
    assert duration == 7


def test_critical_path_complex(complex_dag):
    """Critical path of complex DAG."""
    path, duration = complex_dag.critical_path()
    # A(2)->B(3)->D(4)->F(1) = 10
    # A(2)->C(1)->D(4)->F(1) = 8
    # A(2)->C(1)->E(2)->F(1) = 6
    assert duration == 10
    assert path == ["A", "B", "D", "F"]


# --- Parallel groups tests ---

def test_parallel_groups_linear(linear_chain):
    """Linear chain has one task per wave."""
    groups = linear_chain.parallel_groups()
    assert len(groups) == 3
    assert groups[0] == ["A"]
    assert groups[1] == ["B"]
    assert groups[2] == ["C"]


def test_parallel_groups_diamond(diamond):
    """Diamond: wave 0 = [A], wave 1 = [B, C], wave 2 = [D]."""
    groups = diamond.parallel_groups()
    assert len(groups) == 3
    assert groups[0] == ["A"]
    assert set(groups[1]) == set(["B", "C"])
    assert groups[2] == ["D"]


def test_parallel_groups_independent(parallel_tasks):
    """All independent tasks should be in one wave."""
    groups = parallel_tasks.parallel_groups()
    assert len(groups) == 1
    assert set(groups[0]) == set(["X", "Y", "Z"])


def test_parallel_groups_no_overlap(complex_dag):
    """Tasks in the same wave must not depend on each other."""
    groups = complex_dag.parallel_groups()
    for wave in groups:
        for t in wave:
            task = complex_dag._tasks[t]
            for dep in task.deps:
                assert dep not in wave, (
                    "Task " + t + " and its dep " + dep + " in same wave"
                )


# --- Earliest start tests ---

def test_earliest_start_linear(linear_chain):
    """Earliest starts in a linear chain: 0, 3, 5."""
    starts = linear_chain.earliest_start()
    assert starts["A"] == 0
    assert starts["B"] == 3   # after A (duration 3)
    assert starts["C"] == 5   # after B (3 + 2)


def test_earliest_start_diamond(diamond):
    """Earliest start respects longest incoming path."""
    starts = diamond.earliest_start()
    assert starts["A"] == 0
    assert starts["B"] == 2   # after A (duration 2)
    assert starts["C"] == 2   # after A (duration 2)
    # D waits for both B (finishes at 5) and C (finishes at 3)
    assert starts["D"] == 5   # max(2+3, 2+1) = 5


def test_earliest_start_complex(complex_dag):
    """Complex DAG earliest start times."""
    starts = complex_dag.earliest_start()
    assert starts["A"] == 0
    assert starts["B"] == 2          # after A
    assert starts["C"] == 2          # after A
    assert starts["D"] == 5          # max(B@2+3, C@2+1) = 5
    assert starts["E"] == 3          # after C (2+1)
    assert starts["F"] == 9          # max(D@5+4, E@3+2) = 9
'''


# ---------------------------------------------------------------------------
# Prompt templates tuned for harder bugs
# ---------------------------------------------------------------------------

HARD_PROMPT_DIRECT = """You are a software engineer. Fix the bug in the following Python code.

## Source file: scheduler.py
```python
{source_code}
```

## Failing test output
```
{test_output}
```

## Instructions
- Find and fix the bug that causes the tests to fail.
- Return ONLY the complete fixed source code of scheduler.py.
- Do NOT include any explanation, just the code.
- Start your response with ```python and end with ```
"""

HARD_PROMPT_LOCALIZED = """You are a software engineer debugging a Python task scheduler.

## Source file: scheduler.py
```python
{source_code}
```

## Failing test output
```
{test_output}
```

## Step 1: Analyze the failures
Look at which tests fail and what they expect. Trace the algorithm to find the exact line(s) that are wrong.

## Step 2: Fix
Return ONLY the complete fixed source code of scheduler.py.
- Start your response with ```python and end with ```
- Do NOT modify test logic, only fix scheduler.py
"""

HARD_PROMPT_TEST_GUIDED = """You are a software engineer. Some tests are failing. Read the test expectations carefully and fix the production code.

## Source file: scheduler.py
```python
{source_code}
```

## Test file (DO NOT MODIFY):
```python
{test_code}
```

## Failing test output
```
{test_output}
```

## Instructions
Fix scheduler.py so all tests pass. Return ONLY the complete fixed scheduler.py.
Start with ```python and end with ```
"""

HARD_PROMPT_MINIMAL = """You are a software engineer. Fix this bug with the SMALLEST possible change.

## Source file: scheduler.py
```python
{source_code}
```

## Failing test output
```
{test_output}
```

## Instructions
- Find the line(s) that are wrong and fix them.
- Return ONLY the complete fixed source code.
- Change as few characters as possible.
- Start with ```python and end with ```
"""

HARD_ARM_PROMPTS = {
    "g1_direct_fix": HARD_PROMPT_DIRECT,
    "g2_localized_fix": HARD_PROMPT_LOCALIZED,
    "g3_test_guided_fix": HARD_PROMPT_TEST_GUIDED,
    "g4_minimal_diff_fix": HARD_PROMPT_MINIMAL,
}

# Critic test subsets for the scheduler
HARD_CRITIC_TESTS = {
    "cycle_check": [
        "test_no_cycle_linear",
        "test_no_cycle_diamond",
        "test_no_false_positive_cycle",
    ],
    "ordering_check": [
        "test_schedule_linear",
        "test_schedule_priority_tiebreak",
        "test_schedule_priority_among_ready",
    ],
    "timing_check": [
        "test_critical_path_linear",
        "test_earliest_start_diamond",
        "test_parallel_groups_diamond",
    ],
}
