"""Jev-backed risk classification.

Replaces the fabricated heuristic confidences in ``classify_tool`` with
calibrated probabilities from TypeSafe AI's Jev decision model. Jev returns
typed answers (Choice, Noul, Score) with calibrated confidence instead of
generated text, which is exactly what risk classification needs: a verdict,
not prose.

Design rules:

1. Classification happens at registration time, never on the per-call hot
   path. ``PolicyEngine.evaluate`` stays at microsecond overhead with no
   network calls. Jev classifies once per tool; the engine enforces forever.
2. Jev is opt-in. The regex ``classify_tool`` remains the zero-dependency
   default. Nothing here imports typesafe_sdk at module load.
3. Failure is UNKNOWN. Timeout, HTTP error, invalid answer, or low
   confidence all collapse to ``RiskLevel.UNKNOWN``, which gates as
   DESTRUCTIVE. A classifier that cannot answer never grants permission.

API shape follows the TypeSafe quickstart (docs.typesafe.ai, verified
2026-10-10): POST https://api.typesafe.ai/v1/systemone with a Bearer
``TYPESAFE_API_KEY``, one Choice question over the four risk levels.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

from signal_gating.security.risk import RiskLevel, ToolRisk, classify_tool

_JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
_JEV_MODEL = "jev-latest"

# Minimum top-choice confidence before the answer is trusted.
# Below this, the tool is UNKNOWN (fail closed).
MIN_CONFIDENCE = 0.7

_RISK_CRITERIA = {
    "read": (
        "No side effects. Reading files, querying state, pure computation. "
        "Examples: get, fetch, search, describe, validate."
    ),
    "draft": (
        "Reversible side effects only. Writing drafts, staging changes, "
        "temporary files. Nothing leaves the local environment."
    ),
    "send": (
        "External effects. Network calls, messages sent, issues filed, "
        "anything another party can observe."
    ),
    "destructive": (
        "Irreversible or high-blast-radius effects. Deletes, money movement, "
        "privilege changes, mass writes, code execution, or anything whose "
        "risk cannot be determined."
    ),
}

_CRITERION_TO_LEVEL = {
    "read": RiskLevel.READ,
    "draft": RiskLevel.DRAFT,
    "send": RiskLevel.SEND,
    "destructive": RiskLevel.DESTRUCTIVE,
}


def _unknown(reason: str) -> ToolRisk:
    return ToolRisk(RiskLevel.UNKNOWN, (reason,), 0.0)


@dataclass(slots=True)
class JevRiskClassifier:
    """Classify tool risk with Jev. Opt-in; failures are UNKNOWN.

    Usage::

        classifier = JevRiskClassifier()  # reads TYPESAFE_API_KEY
        risk = classifier.classify("send_email", "Send an email via SES")
        engine.register("send_email", risk)

    Set ``fallback_to_heuristic=True`` to fall back to the regex
    ``classify_tool`` when Jev is unreachable. The default is strict:
    unreachable Jev means UNKNOWN.
    """

    api_key: str | None = None
    model: str = _JEV_MODEL
    timeout: float = 10.0
    min_confidence: float = MIN_CONFIDENCE
    fallback_to_heuristic: bool = False

    def _key(self) -> str | None:
        return self.api_key or os.environ.get("TYPESAFE_API_KEY")

    def classify(
        self,
        name: str,
        description: str = "",
        source: str = "",
    ) -> ToolRisk:
        """Classify one tool. Never raises; failure is UNKNOWN."""
        key = self._key()
        if not key:
            return self._no_jev("no TYPESAFE_API_KEY available", name, description, source)

        state = f"Tool name: {name}\nDescription: {description}\nSource: {source}".strip()
        try:
            import json

            body = json.dumps(
                {
                    "state": state,
                    "model": self.model,
                    "questions": {
                        "risk": {
                            "type": "choice",
                            "instructions": (
                                "Classify this tool's risk level. Choose the "
                                "highest applicable level. When in doubt, "
                                "choose destructive."
                            ),
                            "criteria": _RISK_CRITERIA,
                        }
                    },
                }
            ).encode("utf-8")
            request = urllib.request.Request(
                _JEV_ENDPOINT,
                data=body,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # network, timeout, invalid JSON: fail closed
            return self._no_jev(
                f"jev request failed: {type(exc).__name__}", name, description, source
            )

        try:
            answer = payload["answers"]["risk"]
            choice = answer["choice"]
            confidence = float(answer.get("confidence", 0.0))
        except (KeyError, TypeError, ValueError):
            return self._no_jev("jev returned an unparseable answer", name, description, source)

        level = _CRITERION_TO_LEVEL.get(choice)
        if level is None:
            return self._no_jev(
                f"jev returned unknown choice {choice!r}", name, description, source
            )

        if confidence < self.min_confidence:
            return ToolRisk(
                RiskLevel.UNKNOWN,
                (f"jev confidence {confidence:.2f} below threshold {self.min_confidence}",),
                0.0,
            )

        return ToolRisk(
            level,
            (f"jev-{self.model} choice={choice}",),
            confidence,
        )

    def _no_jev(
        self, reason: str, name: str = "", description: str = "", source: str = ""
    ) -> ToolRisk:
        if self.fallback_to_heuristic:
            return classify_tool(name, description, source)
        return _unknown(reason)


def classify_with_jev(
    name: str,
    description: str = "",
    source: str = "",
    api_key: str | None = None,
    model: str = _JEV_MODEL,
    timeout: float = 10.0,
    min_confidence: float = MIN_CONFIDENCE,
    fallback_to_heuristic: bool = False,
) -> ToolRisk:
    """Convenience wrapper around ``JevRiskClassifier.classify``."""
    return JevRiskClassifier(
        api_key=api_key,
        model=model,
        timeout=timeout,
        min_confidence=min_confidence,
        fallback_to_heuristic=fallback_to_heuristic,
    ).classify(name, description, source)
