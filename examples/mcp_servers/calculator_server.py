"""A small, standalone calculator MCP server.

Run it directly with:
    uv run python examples/mcp_servers/calculator_server.py

The process uses stdio for MCP messages, so do not print application output to
stdout. FastMCP sends diagnostics to stderr when needed.
"""

from __future__ import annotations

import math

from mcp.server.fastmcp import FastMCP


mcp = FastMCP(
    "Lorcy Calculator",
    instructions="提供基础算术、幂和平方根计算。",
)


@mcp.tool()
def add(a: float, b: float) -> float:
    """Return a plus b."""
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Return a minus b."""
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Return a multiplied by b."""
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Return a divided by b. Division by zero is rejected."""
    if b == 0:
        raise ValueError("除数不能为 0")
    return a / b


@mcp.tool()
def power(base: float, exponent: float) -> float:
    """Raise base to exponent and return a finite real-number result."""
    try:
        result = math.pow(base, exponent)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"无法计算该幂运算: {exc}") from exc
    if not math.isfinite(result):
        raise ValueError("计算结果不是有限数")
    return result


@mcp.tool()
def square_root(value: float) -> float:
    """Return the real square root of a non-negative number."""
    if value < 0:
        raise ValueError("实数平方根的输入不能小于 0")
    return math.sqrt(value)


if __name__ == "__main__":
    mcp.run(transport="stdio")
