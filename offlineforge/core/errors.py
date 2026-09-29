"""Error taxonomy for OfflineForge.

Numeric codes follow the SOP convention (E1xx..E5xx) so pipelines can branch
on category without string matching.
"""

from __future__ import annotations


class OfflineForgeError(Exception):
    code = "E000"
    title = "OfflineForge error"

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.code}] {self.title}: {self.message}"


class ConfigError(OfflineForgeError):
    code = "E100"
    title = "Configuration error"


class DataError(OfflineForgeError):
    code = "E200"
    title = "Dataset error"


class AlgoError(OfflineForgeError):
    code = "E300"
    title = "Algorithm error"


class OPEError(OfflineForgeError):
    code = "E400"
    title = "Off-policy evaluation error"


class EvalError(OfflineForgeError):
    code = "E500"
    title = "Evaluation / comparison error"
