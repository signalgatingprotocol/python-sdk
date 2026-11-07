"""Validated operator configuration; JSON never imports executable code."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from .records import (
    Currency,
    Grant,
    Model,
    NonEmpty,
    NonNegativeInt,
    PositiveInt,
    PositiveNumber,
    Principal,
)


class PriceConfig(Model):
    version: NonEmpty
    currency: Currency = "USD"
    input_microunits_per_million_tokens: NonNegativeInt
    output_microunits_per_million_tokens: NonNegativeInt
    maximum_charge_microunits: NonNegativeInt | None = None


# Environment-variable names only, never credential contents.
AnnotatedEnv = Annotated[str, Field(strict=True, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]


class WorkerConfig(Model):
    id: NonEmpty
    kind: Literal["fixture", "openai"]
    model: NonEmpty | None = None
    api_key_env: AnnotatedEnv | None = None
    context_token_limit: PositiveInt = 8192
    max_output_tokens: PositiveInt = 2048
    token_counter_id: NonEmpty = "utf8-bytes-v1"
    framing_tokens_reserved: NonNegativeInt = 256
    prices: PriceConfig | None = None

    @model_validator(mode="after")
    def _provider_configuration(self) -> WorkerConfig:
        if self.kind == "openai" and (self.model is None or self.api_key_env is None):
            raise ValueError("openai worker requires model and an environment-variable reference")
        if self.max_output_tokens + self.framing_tokens_reserved >= self.context_token_limit:
            raise ValueError("context capacity must leave room for input")
        return self


class ArgvCheck(Model):
    id: NonEmpty
    argv: tuple[NonEmpty, ...] = Field(min_length=1)
    timeout_seconds: PositiveNumber = 30


class WorkspaceConfig(Model):
    root: Path
    readable_paths: tuple[NonEmpty, ...] = (".",)
    writable_paths: tuple[NonEmpty, ...] = ()
    named_checks: tuple[ArgvCheck, ...] = ()
    allow_symlinks: bool = False


class SchedulerConfig(Model):
    policy_version: Literal["deterministic-v1"] = "deterministic-v1"
    control_batch_size: PositiveInt = 64
    reconciliation_quota: PositiveInt = 4
    urgent_quota: PositiveInt = 2
    ordinary_quota: PositiveInt = 1
    aging_interval_seconds: PositiveNumber = 60
    max_aging_points: NonNegativeInt = 100
    urgent_deadline_seconds: PositiveNumber = 60
    control_overload_seconds: PositiveNumber = 10


class KernelConfig(Model):
    owner: Principal | None = None
    bootstrap_owner: bool = False
    bootstrap_grants: tuple[Grant, ...] = ()
    workers: tuple[WorkerConfig, ...] = ()
    workspace: WorkspaceConfig | None = None
    verifier_ids: tuple[NonEmpty, ...] = ()
    approver_ids: tuple[NonEmpty, ...] = ()
    max_active_episodes: PositiveInt = 2
    episode_timeout_seconds: PositiveNumber = 120
    max_output_tokens: PositiveInt = 2048
    context_token_limit: PositiveInt = 8192
    framing_tokens_reserved: NonNegativeInt = 256
    max_signal_bytes: PositiveInt = 65_536
    max_artifact_bytes: PositiveInt = 8_388_608
    identical_failure_limit: PositiveInt = 2
    verification_repair_rounds: NonNegativeInt = 2
    max_queued_ordinary_items: PositiveInt = 10_000
    strict_money: bool = True
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)

    @model_validator(mode="after")
    def _configuration(self) -> KernelConfig:
        if self.bootstrap_owner and self.owner is None:
            raise ValueError("bootstrap requires an owner identity")
        ids = [worker.id for worker in self.workers]
        if len(ids) != len(set(ids)):
            raise ValueError("worker IDs must be unique")
        if self.max_output_tokens + self.framing_tokens_reserved >= self.context_token_limit:
            raise ValueError("context capacity must leave room for input")
        return self
