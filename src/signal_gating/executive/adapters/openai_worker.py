"""Proposal-only OpenAI-compatible chat completions; no tool execution or retries.

The injected LLMClient must have retries disabled by its operator. Clients that
expose max_retries are rejected unless it is zero; structural clients without
that attribute require the same operator guarantee. This boundary cannot audit a
custom client's internals or guarantee external pricing/charges. It is not a
spending sandbox. No provider options or extra prompts are supplied implicitly.

quote is a trusted per-call MAXIMUM, with quote_version and quote_source naming
versioned operator/provider configuration. It is never model output. Use
reservation_batch with BudgetPolicy; combine the returned event-only batch with
subject authorization in the coordinator before any call. Persist quote metadata
with admission. Settlement uses provider observations, not the capacity estimate,
quote, or the JSON model's usage claims. Missing billing data stays unknown.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from time import monotonic
from typing import cast

from pydantic import ValidationError

from signal_gating.llm import LLMClient

from ..budgets import BudgetPolicy
from ..context import TokenCounter, render_packet
from ..messages import CommitBatch, EpisodeResult
from ..records import ContextPacket, Episode, JournalEvent, Limits, Ref, StateView, Usage
from ..workers import WorkerError

_OUTPUT_PARAMETERS = frozenset({"max_tokens", "max_completion_tokens", "max_output_tokens"})


def _optional_integer(value: object, name: str) -> int | None:
    candidate = value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)
    return candidate if type(candidate) is int and candidate >= 0 else None


class OpenAIWorker:
    def __init__(
        self,
        client: LLMClient,
        *,
        model: str,
        counter: TokenCounter,
        capacity: int,
        quote: Limits | None,
        quote_version: str | None = None,
        quote_source: str | None = None,
        output_limit_parameter: str = "max_tokens",
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("An explicit model is required")
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if output_limit_parameter not in _OUTPUT_PARAMETERS:
            raise ValueError("Unsupported output limit parameter")
        if getattr(client, "max_retries", 0) != 0:
            raise ValueError("Client retries must be disabled")
        if quote is not None and (not quote_version or not quote_source):
            raise ValueError("A quote requires versioned operator provenance")
        self.client = client
        self.model = model
        self.counter = counter
        self.capacity = capacity
        self.quote = quote
        self.quote_version = quote_version
        self.quote_source = quote_source
        self.output_limit_parameter = output_limit_parameter

    @classmethod
    def from_environment(
        cls,
        *,
        api_key_env: str,
        model: str,
        counter: TokenCounter,
        capacity: int,
        quote: Limits,
        quote_version: str,
        quote_source: str,
        output_limit_parameter: str = "max_tokens",
        base_url: str | None = None,
    ) -> OpenAIWorker:
        """Construct the optional real client with max_retries=0. Makes no request.

        Requires signal-gating[llm]. api_key_env is a name, never a credential.
        The caller owns await worker.client.close() on the concrete provider
        client; EpisodeRunner.close does not imply transport/provider shutdown.
        """
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env) is None:
            raise ValueError("Invalid credential environment-variable name")
        key = os.environ.get(api_key_env)
        if not key:
            raise ValueError("Credential environment variable is unset")
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=key, base_url=base_url, max_retries=0)
        return cls(
            cast(LLMClient, client),
            model=model,
            counter=counter,
            capacity=capacity,
            quote=quote,
            quote_version=quote_version,
            quote_source=quote_source,
            output_limit_parameter=output_limit_parameter,
        )

    def _input(self, packet: ContextPacket) -> tuple[list[dict[str, str]], int]:
        messages = list(render_packet(packet))
        try:
            count = self.counter.count(
                json.dumps(messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            )
        except Exception:
            raise WorkerError("invalid_token_counter", "Input counter failed") from None
        if type(count) is not int or count <= 0:
            raise WorkerError("invalid_token_counter", "Input counter returned invalid capacity")
        output = packet.manifest.output_tokens_reserved
        framing = packet.manifest.framing_tokens_reserved
        capacities = [self.capacity]
        if packet.manifest.context_token_limit is not None:
            capacities.append(packet.manifest.context_token_limit)
        if output <= 0 or count + output + framing > min(capacities):
            raise WorkerError("context_budget_insufficient", "Required context exceeds capacity")
        limits = packet.remaining_limits
        if (
            (limits.input_tokens is not None and count > limits.input_tokens)
            or (limits.output_tokens is not None and output > limits.output_tokens)
            or (limits.calls is not None and limits.calls < 1)
            or (limits.elapsed_seconds is not None and limits.elapsed_seconds <= 0)
        ):
            raise WorkerError("worker_limit_exceeded", "Per-call limits do not permit inference")
        if self.quote is not None and (
            self.quote.input_tokens is None
            or self.quote.input_tokens < count + framing
            or self.quote.output_tokens is None
            or self.quote.output_tokens < output
            or self.quote.calls != 1
            or self.quote.elapsed_seconds is None
            or self.quote.elapsed_seconds <= 0
        ):
            raise WorkerError("invalid_worker_quote", "Quote does not cover per-call limits")
        return messages, count

    def per_call_limits(self, packet: ContextPacket) -> Limits:
        """Trusted capacity/maxima for coordinator admission, never billed usage.

        Without a monetary quote this does not authorize strict-money calls.
        With one, the configured quote remains the reserved maximum. Construct
        Episode.deadline within elapsed_seconds and use EpisodeRunner, including
        its current-state admission hook; run_episode alone is not a sandbox.
        """
        _, count = self._input(packet)
        if self.quote is not None:
            return self.quote
        return Limits(
            input_tokens=count + packet.manifest.framing_tokens_reserved,
            output_tokens=packet.manifest.output_tokens_reserved,
            calls=1,
            elapsed_seconds=packet.remaining_limits.elapsed_seconds,
            currency=packet.remaining_limits.currency,
        )

    def reservation_batch(
        self,
        policy: BudgetPolicy,
        state: StateView,
        budget_id: str,
        *,
        episode: Episode,
        packet: ContextPacket,
    ) -> CommitBatch:
        """Prepare/recheck BudgetPolicy admission for the actual episode subject.

        Does NOT commit. When subject is new, merge subject and reservation into
        one state.changed post-image bundle, union original CAS reads and persist
        context/artifact/quote audit atomically. Reinvoke against current state at
        dispatch, including when reusing a reservation, so ancestor overages deny.
        Coordinator enforces deadline <= quote.elapsed_seconds from dispatch and
        uses EpisodeRunner's current-state hook. Never settle from quote maxima.
        """
        self._input(packet)
        if (
            self.quote is None
            or episode.reservation_id is None
            or episode.id != packet.episode_id
            or episode.context_packet_id != packet.id
            or episode.task_id != packet.task.id
            or episode.fencing_token != packet.fencing_token
            or episode.model != self.model
            or self.quote.elapsed_seconds is None
            or not 0
            < (episode.deadline - policy.clock()).total_seconds()
            <= self.quote.elapsed_seconds
        ):
            raise WorkerError("invalid_worker_quote", "Episode requires a bound trusted quote")
        batch = policy.reserve(
            state,
            budget_id,
            self.quote,
            episode.reservation_id,
            subject_id=episode.id,
            trusted_maximum_quote=True,
        )
        audit = JournalEvent(
            created_at=policy.clock(),
            sequence=state.sequence + len(batch.events) + 1,
            principal_id=batch.events[0].principal_id,
            event_type="audit.recorded",
            producer_id="openai-worker",
            subject=Ref(id=episode.id),
            payload={
                "reason_code": "worker_quote",
                "details": {
                    "quote_version": self.quote_version,
                    "quote_source": self.quote_source,
                    "model": self.model,
                    "maximum": self.quote.model_dump(mode="json"),
                },
            },
        )
        return batch.model_copy(update={"events": (*batch.events, audit)})

    async def run_episode(self, packet: ContextPacket) -> EpisodeResult:
        messages, _ = self._input(packet)
        started = monotonic()
        currency = (
            self.quote.currency if self.quote is not None else packet.remaining_limits.currency
        )
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=cast(list[dict[str, object]], messages),
                **{self.output_limit_parameter: packet.manifest.output_tokens_reserved},
            )
        except Exception:
            raise WorkerError(
                "provider_failure",
                "Provider invocation failed",
                usage=Usage(
                    calls=1,
                    elapsed_seconds=max(0, monotonic() - started),
                    source="provider-observation",
                    currency=currency,
                ),
            ) from None
        usage = Usage(
            input_tokens=_optional_integer(getattr(response, "usage", None), "prompt_tokens"),
            output_tokens=_optional_integer(getattr(response, "usage", None), "completion_tokens"),
            calls=1,
            elapsed_seconds=max(0, monotonic() - started),
            source="provider-observation",
            currency=currency,
        )
        try:
            if len(response.choices) != 1:
                raise WorkerError(
                    "invalid_worker_result", "Provider must return one result", usage=usage
                )
            message = response.choices[0].message
            if message.tool_calls or getattr(message, "function_call", None):
                raise WorkerError(
                    "provider_tools_rejected", "Provider tools are not permitted", usage=usage
                )
            if not isinstance(message.content, str):
                raise ValueError("missing content")
            try:
                output_count = self.counter.count(message.content)
            except Exception:
                raise WorkerError(
                    "invalid_token_counter", "Output counter failed", usage=usage
                ) from None
            if type(output_count) is not int or output_count <= 0:
                raise WorkerError("invalid_token_counter", "Invalid output capacity", usage=usage)
            if output_count > packet.manifest.output_tokens_reserved:
                raise WorkerError(
                    "worker_bound_exceeded", "Provider output bound exceeded", usage=usage
                )
            result = EpisodeResult.model_validate_json(message.content)
        except WorkerError:
            raise
        except (ValidationError, ValueError, TypeError, AttributeError, IndexError):
            raise WorkerError(
                "invalid_worker_result", "Invalid worker result", usage=usage
            ) from None
        # The JSON usage field is model-generated and cannot establish billing.
        result = result.model_copy(update={"usage": usage})
        if any(
            p.episode_id != packet.episode_id or p.fencing_token != packet.fencing_token
            for p in result.proposals
        ):
            raise WorkerError("invalid_worker_result", "Proposal binding rejected", usage=usage)
        if (
            usage.output_tokens is not None
            and usage.output_tokens > packet.manifest.output_tokens_reserved
        ):
            raise WorkerError(
                "worker_bound_exceeded", "Provider output bound exceeded", usage=usage
            )
        if self.quote is not None and (
            (
                usage.input_tokens is not None
                and usage.input_tokens > cast(int, self.quote.input_tokens)
            )
            or cast(float, usage.elapsed_seconds) > cast(float, self.quote.elapsed_seconds)
        ):
            raise WorkerError("worker_bound_exceeded", "Provider quote bound exceeded", usage=usage)
        return result
