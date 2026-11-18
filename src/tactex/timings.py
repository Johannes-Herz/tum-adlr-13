import functools
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch.utils.data import DataLoader, default_collate

from tactex.settings import get_settings

type Unit = Literal["ms", "s"]


@dataclass()
class Timings:
    data: dict[str, list[float]] = field(default_factory=dict)

    def add(self, fn_name: str, elapsed_ns: int) -> None:
        if fn_name not in self.data:
            self.data[fn_name] = []
        self.data[fn_name].append(elapsed_ns)

    @property
    def mean(self) -> dict[str, float]:
        return {
            fn_name: sum(timings) / len(timings)
            for fn_name, timings in self.data.items()
        }

    def drain(self) -> dict[str, list[int]]:
        out = {k: v[:] for k, v in self.data.items()}
        self.data.clear()
        return out

    def merge(self, other: dict[str, list[int]]) -> None:
        for k, v in other.items():
            self.data.setdefault(k, []).extend(v)

    def stats(self, unit: Unit) -> dict[str, dict[str, int | float]]:
        factor: float = 1_000_000 if unit == "ms" else 1_000_000_000
        return {
            fn_name: {
                "count": len(timings),
                "total": sum(timings) / factor,
                "mean": statistics.mean(timings) / factor,
                "median": statistics.median(timings) / factor,
                "min": min(timings) / factor,
                "max": max(timings) / factor,
            }
            for fn_name, timings in self.data.items()
        }

    def reset(self) -> None:
        self.data.clear()

    def _to_table(self, unit: Unit, decimals: int = 4) -> str:
        stats: dict[str, dict[str, float]] = self.stats(unit=unit)
        total: float = sum(fn_stats["total"] for fn_stats in stats.values())
        headers: list[str] = [
            "function",
            "count",
            f"total [{unit}]",
            "%",
            f"mean [{unit}]",
            f"median [{unit}]",
            f"min [{unit}]",
            f"max [{unit}]",
        ]

        rows: list[str] = [
            [
                name,
                str(s["count"]),
                f"{s['total']:.{decimals}f}",
                f"{100 * s['total'] / total:.1f}",
                *(f"{s[k]:.{decimals}f}" for k in ("mean", "median", "min", "max")),
            ]
            for name, s in sorted(
                stats.items(), key=lambda kv: kv[1]["total"], reverse=True
            )
        ]

        widths: list[int] = [
            max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))
        ]

        def fmt(row: list[str]) -> str:
            return "  ".join(
                cell.ljust(w) if i == 0 else cell.rjust(w)
                for i, (cell, w) in enumerate(zip(row, widths))
            )

        return "\n".join([fmt(headers), "-" * len(fmt(headers)), *map(fmt, rows)])

    def __str__(self) -> str:
        return self._to_table(unit="ms")


_TIMINGS: Timings = Timings()
_ENABLED: bool = get_settings().log_timings
_CUDA: bool = torch.cuda.is_available()


def _sync():
    if _CUDA:
        torch.cuda.synchronize()


def set_enabled(enabled: bool):
    global _ENABLED
    _ENABLED = enabled


def is_enabled():
    return _ENABLED


def get_timings() -> Timings:
    return _TIMINGS


def _name_of_fn(fn: Callable) -> str:
    qualname: str = getattr(fn, "__qualname__", None) or type(fn).__qualname__
    module: str = getattr(fn, "__module__", None) or type(fn).__module__
    return f"{module}.{qualname}"


def timed():
    def decorator(fn: Callable):
        fn_name: str = _name_of_fn(fn=fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not _ENABLED:
                return fn(*args, **kwargs)

            _sync()
            start_time_ns: int = time.perf_counter_ns()
            try:
                return fn(*args, **kwargs)
            finally:
                _sync()
                _TIMINGS.add(
                    fn_name=fn_name,
                    elapsed_ns=(time.perf_counter_ns() - start_time_ns),
                )

        return wrapper

    return decorator


class TimedCollate:
    def __init__(self, collate_fn, use_workers: bool):
        self.collate_fn = collate_fn
        self.use_workers: bool = use_workers

    def __call__(self, batch):
        out = self.collate_fn(batch)
        return (*out, _TIMINGS.drain() if self.use_workers else {})


class TimedDataLoader(DataLoader):
    def __init__(self, *args, name="dataloader", collate_fn, **kwargs):
        super().__init__(
            *args,
            collate_fn=TimedCollate(
                collate_fn or default_collate, kwargs.get("num_workers", 0) > 0
            ),
            **kwargs,
        )
        self.name: str = f"{name}.wait"

    def __iter__(self):
        it = super().__iter__()
        while True:
            start = time.perf_counter_ns()
            try:
                *batch, worker_timings = next(it)
            except StopIteration:
                return
            _TIMINGS.add(self.name, time.perf_counter_ns() - start)
            _TIMINGS.merge(worker_timings)
            yield tuple(batch)
