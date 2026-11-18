import functools
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from tactex.settings import get_settings


@dataclass()
class Timings:
    data: dict[str, list[float]] = field(default_factory=dict)

    def add(self, fn_name: str, elapsed_in_ms: float) -> None:
        if fn_name not in self.data:
            self.data[fn_name] = []
        self.data[fn_name].append(elapsed_in_ms)

    @property
    def mean(self) -> dict[str, float]:
        return {
            fn_name: sum(timings) / len(timings)
            for fn_name, timings in self.data.items()
        }

    def stats(self) -> dict[str, dict[str, int | float]]:
        return {
            fn_name: {
                "count": len(timings),
                "total": sum(timings),
                "mean": statistics.mean(timings),
                "median": statistics.median(timings),
                "min": min(timings),
                "max": max(timings),
            }
            for fn_name, timings in self.data.items()
        }

    def reset(self) -> None:
        self.data.clear()

    def _to_table(self, decimals: int = 4) -> str:
        stats: dict[str, dict[str, float]] = self.stats()
        total: float = sum(fn_stats["total"] for fn_stats in stats.values())
        headers: list[str] = [
            "function",
            "count",
            "total [ms]",
            "%",
            "mean [ms]",
            "median [ms]",
            "min [ms]",
            "max [ms]",
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
        return self._to_table()


TIMINGS: Timings = Timings()
_ENABLED: bool = get_settings().log_timings
_CUDA: bool = torch.cuda.is_available()


def _sync():
    if _CUDA:
        torch.cuda.synchronize()


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
            start_time: int = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                _sync()
                TIMINGS.add(
                    fn_name=fn_name,
                    elapsed_in_ms=(time.perf_counter() - start_time) * 1_000,
                )

        return wrapper

    return decorator
