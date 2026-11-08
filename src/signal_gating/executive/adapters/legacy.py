"""Explicit compatibility bridges, without promoting legacy runtime authority.

Legacy payload roles, commands and actions are untrusted observation content.
Code running outside the executive dispatcher remains an unmediated external
runtime. Wrapping it does not sandbox it or grant it capabilities.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid5

from signal_gating.signal import Signal

from ..messages import EpisodeResult
from ..records import ContextPacket, ExecutiveSignal, Principal, Scope, Usage, utc_now
from ..workers import WorkerError


class LegacySignalAdapter:
    """Convert only to observations, bound to explicit authenticated scope.

    Caller authentication and admission remain the host/Intake responsibilities.
    Valid executive causal IDs survive exactly. Other legacy string IDs map via
    a deterministic principal/goal namespace; the original wire identity stays
    in the untrusted payload. Repeated identity with changed content is left for
    Intake's durable duplicate-conflict check, never silently overwritten.
    """

    def __init__(self, *, clock: Callable[[], datetime] = utc_now) -> None:
        self._clock = clock

    def convert(self, signal: Signal, *, principal: Principal, goal_id: str) -> ExecutiveSignal:
        if not isinstance(signal, Signal) or not isinstance(principal, Principal):
            raise TypeError("Expected a legacy Signal and authenticated Principal")
        scope = Scope(goal_id=goal_id, principal_ids=(principal.id,))
        namespace = uuid5(NAMESPACE_URL, f"sgp:legacy-observation:v1:{principal.id}:{goal_id}")
        identity = uuid5(namespace, "signal:" + signal.id).hex
        causal_id = (
            signal.id
            if re.fullmatch(r"[0-9a-f]{32}", signal.id)
            else uuid5(namespace, "causal:" + signal.id).hex
        )
        observed = datetime.fromtimestamp(signal.timestamp, timezone.utc)
        received = self._clock()
        return ExecutiveSignal(
            id=identity,
            created_at=received,
            received_at=received,
            observed_at=observed,
            signal_kind="observation",
            principal_id=principal.id,
            producer_id="legacy-signal-adapter-v1",
            external_id=identity,
            causal_ids=(causal_id,),
            scope=scope,
            payload={
                "mediation": "external_runtime_unmediated",
                "authority": "observation_only",
                "legacy": signal.to_wire(),
            },
        )


class LegacyWorker:
    """Adapt an explicit async run_episode callable, with no inferred privileges.

    No Agent, Mesh, tool registration, credential lookup or dynamic import is
    inferred. Results remain proposals for ordinary executive validation. The
    same-process callable itself can act outside mediation; the host owns that
    boundary, cancellation, and any isolation it needs.
    """

    mediation = "external_runtime_unmediated"

    def __init__(self, run_episode: Callable[[ContextPacket], Awaitable[EpisodeResult]]) -> None:
        if not callable(run_episode):
            raise TypeError("Supply an explicit run_episode callable")
        inspect.signature(run_episode).bind(object())
        self._run_episode = run_episode

    async def run_episode(self, packet: ContextPacket) -> EpisodeResult:
        result = await self._run_episode(packet)
        if not isinstance(result, EpisodeResult):
            raise WorkerError(
                "invalid_worker_result",
                "Legacy callable returned an invalid episode result",
                usage=Usage(calls=1, currency=packet.remaining_limits.currency),
            )
        return result
