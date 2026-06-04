"""Optional integrations for external agent runtimes and tool systems."""

from signal_gating.integrations.claude import (
    ClaudeAgent,
    ClaudeAgentResultSignal,
    ClaudeAgentRunSignal,
    ClaudeAgentSDKResult,
    ClaudeAgentSDKRunner,
    ClaudeAgentSDKSession,
    ClaudeClientFactory,
    ClaudePermissionDecision,
    ClaudePermissionDecisionSignal,
    ClaudeQueryFn,
    ClaudeToolEventKind,
    ClaudeToolEventSignal,
    claude_options,
    mcp_tool_name,
)

__all__ = [
    "ClaudeAgent",
    "ClaudeAgentSDKResult",
    "ClaudeAgentSDKRunner",
    "ClaudeAgentSDKSession",
    "ClaudeClientFactory",
    "ClaudeAgentResultSignal",
    "ClaudeAgentRunSignal",
    "ClaudePermissionDecision",
    "ClaudePermissionDecisionSignal",
    "ClaudeQueryFn",
    "ClaudeToolEventKind",
    "ClaudeToolEventSignal",
    "claude_options",
    "mcp_tool_name",
]
