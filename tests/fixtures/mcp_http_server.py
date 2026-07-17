import os

from mcp.server.fastmcp import FastMCP


server = FastMCP(
    "LorcyCode HTTP test server",
    host="127.0.0.1",
    port=int(os.environ["MCP_TEST_PORT"]),
    stateless_http=True,
)


@server.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


if __name__ == "__main__":
    server.run(transport="streamable-http")
