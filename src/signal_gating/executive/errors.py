"""Structured errors for deterministic executive boundary failures."""

from __future__ import annotations


class ExecutiveError(Exception):
    """A machine-readable reason plus an operator-facing explanation."""

    def __init__(self, code: str, message: str, *, record_id: str | None = None) -> None:
        self.code = code
        self.message = message
        self.record_id = record_id
        super().__init__(message)


class ContractError(ExecutiveError):
    """An unsupported or invalid wire contract."""


class ConflictError(ExecutiveError):
    """A duplicate identity or changed revision conflicts with a write."""


class AuthorityError(ExecutiveError):
    """The authenticated principal lacks authority."""


class BudgetError(ExecutiveError):
    """A resource reservation cannot be admitted."""


class ContextBudgetError(ExecutiveError):
    """Required context cannot fit without losing required instructions."""


class StoreError(ExecutiveError):
    """Storage ownership, integrity, or platform support failed."""
