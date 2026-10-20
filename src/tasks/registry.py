"""
Name -> object registries: TASKS (by data_source), REWARDS and HACK_LABELS.
"""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """
    Name -> object map; unknown names and duplicate registrations fail loudly.
    """

    def __init__(self, kind: str):
        self._kind = kind
        self._items: dict[str, T] = {}

    def add(self, name: str, obj: T) -> None:
        if name in self._items:
            raise ValueError(f"{self._kind} {name!r} is already registered.")
        self._items[name] = obj

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(sorted(self._items)) or "(none registered)"
            raise KeyError(f"Unknown {self._kind}: {name!r}. Registered: {known}.") from None


TASKS: Registry = Registry("task (data_source)")
REWARDS: Registry = Registry("reward")
HACK_LABELS: Registry = Registry("hack label")


def register_task(data_source: str) -> Callable[[type], type]:
    """
    Instantiate the decorated Task (no-arg constructor) and register it.
    """

    def deco(cls: type) -> type:
        TASKS.add(data_source, cls())
        return cls

    return deco


def register_reward(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        REWARDS.add(name, cls())
        return cls

    return deco


def register_hack_label(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        HACK_LABELS.add(name, cls())
        return cls

    return deco


def get_task(data_source: str):
    return TASKS.get(data_source)


def get_reward(name: str):
    return REWARDS.get(name)


def get_hack_label(name: str):
    return HACK_LABELS.get(name)
