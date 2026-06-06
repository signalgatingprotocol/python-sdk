"""Command-line entrypoints for Signal Gating Protocol."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import sys
from collections.abc import Sequence
from contextlib import asynccontextmanager, redirect_stdout
from importlib import import_module
from typing import Any

from signal_gating.claude import ClaudeMeshMCPAdapter, ClaudeMeshMCPStdioServer
from signal_gating.mesh import Mesh


def main(argv: Sequence[str] | None = None) -> int:
    """Console entrypoint for ``signal-gating-mcp``."""
    args = _parser().parse_args(argv)

    return asyncio.run(
        run_mcp_stdio(
            args.factory,
            server_name=args.server_name,
            server_version=args.server_version,
        )
    )


async def run_mcp_stdio(
    factory_ref: str,
    *,
    server_name: str = "mesh",
    server_version: str = "0.1.0",
    input_stream: Any | None = None,
    output_stream: Any | None = None,
    error_stream: Any | None = None,
) -> int:
    """Serve an MCP stdio session and return a process-style exit code."""
    error_stream = error_stream if error_stream is not None else sys.stderr
    try:
        await serve_mcp_stdio(
            factory_ref,
            server_name=server_name,
            server_version=server_version,
            input_stream=input_stream,
            output_stream=output_stream,
            error_stream=error_stream,
        )
    except Exception as e:
        print(f"signal-gating-mcp: {type(e).__name__}: {e}", file=error_stream)
        return 1
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signal-gating-mcp",
        description="Serve an SGP Mesh or ClaudeMeshMCPAdapter as an MCP stdio server.",
    )
    parser.add_argument(
        "factory",
        help="Python reference in module:attribute form. The target may return a Mesh or adapter.",
    )
    parser.add_argument("--server-name", default="mesh", help="MCP server name for Mesh targets.")
    parser.add_argument(
        "--server-version",
        default="0.1.0",
        help="MCP server version for Mesh targets.",
    )
    return parser


async def serve_mcp_stdio(
    factory_ref: str,
    *,
    server_name: str = "mesh",
    server_version: str = "0.1.0",
    input_stream: Any | None = None,
    output_stream: Any | None = None,
    error_stream: Any | None = None,
) -> int:
    """Load a Mesh or MCP adapter factory and serve it over stdio."""
    error_stream = error_stream if error_stream is not None else sys.stderr
    with redirect_stdout(error_stream):
        target = await _load_factory_target(factory_ref)
    adapter, mesh = _adapter_from_target(
        target,
        server_name=server_name,
        server_version=server_version,
    )
    server = ClaudeMeshMCPStdioServer(adapter, diagnostic_stream=error_stream)
    input_stream = input_stream if input_stream is not None else sys.stdin
    output_stream = output_stream if output_stream is not None else sys.stdout

    async with _maybe_mesh_lifecycle(mesh, error_stream):
        return await server.serve(input_stream, output_stream)


async def _load_factory_target(factory_ref: str) -> Any:
    module_name, sep, attribute = factory_ref.partition(":")
    if not sep or not module_name or not attribute:
        raise ValueError("factory must be in module:attribute form")
    module = import_module(module_name)
    target = module
    for part in attribute.split("."):
        target = getattr(target, part)
    if callable(target):
        value = target()
        if inspect.isawaitable(value):
            return await value
        return value
    return target


def _adapter_from_target(
    target: Any,
    *,
    server_name: str,
    server_version: str,
) -> tuple[ClaudeMeshMCPAdapter, Mesh | None]:
    if isinstance(target, ClaudeMeshMCPAdapter):
        return target, None
    if isinstance(target, Mesh):
        return (
            ClaudeMeshMCPAdapter(
                target,
                server_name=server_name,
                server_version=server_version,
            ),
            target,
        )
    raise TypeError("factory must return a Mesh or ClaudeMeshMCPAdapter")


@asynccontextmanager
async def _maybe_mesh_lifecycle(mesh: Mesh | None, diagnostic_stream: Any) -> Any:
    if mesh is None:
        yield
        return
    with redirect_stdout(diagnostic_stream):
        await mesh.start()
    try:
        yield
    finally:
        with redirect_stdout(diagnostic_stream):
            await mesh.stop()


__all__ = ["main", "run_mcp_stdio", "serve_mcp_stdio"]
