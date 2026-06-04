"""Re-export the Claude Agent SDK integration boundary."""

from signal_gating.claude import (
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
