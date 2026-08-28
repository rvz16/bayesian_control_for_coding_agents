from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Set


ConstraintFn = Callable[[object], bool]


@dataclass(frozen=True)
class ParameterDomain:
    values: Optional[Set[object]] = None
    constraints: tuple[ConstraintFn, ...] = field(default_factory=tuple)

    @staticmethod
    def from_values(values: Iterable[object]) -> "ParameterDomain":
        return ParameterDomain(values=set(values))

    @staticmethod
    def continuous(constraints: Optional[Iterable[ConstraintFn]] = None) -> "ParameterDomain":
        return ParameterDomain(values=None, constraints=tuple(constraints or ()))

    def is_finite(self) -> bool:
        return self.values is not None

    def size(self) -> Optional[int]:
        if self.values is None:
            return None
        return len(self.values)

    def contains(self, value: object) -> bool:
        if self.values is not None:
            if isinstance(value, (list, tuple, set)):
                try:
                    if tuple(value) in self.values:
                        return True
                except TypeError:
                    pass
                return all(item in self.values for item in value)
            return value in self.values
        return all(constraint(value) for constraint in self.constraints)

    def intersect_values(self, values: Iterable[object]) -> "ParameterDomain":
        if self.values is None:
            allowed = set(values)
            return self.with_constraints(lambda v: v in allowed)
        return ParameterDomain(values=self.values.intersection(values))

    def exclude_values(self, values: Iterable[object]) -> "ParameterDomain":
        if self.values is None:
            excluded = set(values)
            return self.with_constraints(lambda v: v not in excluded)
        return ParameterDomain(values=self.values.difference(values))

    def with_constraints(self, *constraints: ConstraintFn) -> "ParameterDomain":
        return ParameterDomain(values=self.values, constraints=self.constraints + tuple(constraints))
