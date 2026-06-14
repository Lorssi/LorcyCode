from __future__ import annotations

import os
import tempfile
import uuid
from dataclasses import dataclass

MAX_OUTPUT_LINES = 2000
MAX_OUTPUT_BYTES = 51200
MAX_PERSISTED_BYTES = 64 * 1024 * 1024


@dataclass
class TruncatedOutput:
    content: str
    truncated: bool = False
    persisted_path: str | None = None
    total_bytes: int = 0


def truncate_output(stdout: str) -> TruncatedOutput:
    total_bytes = len(stdout.encode("utf-8", errors="replace"))
    lines = stdout.splitlines(keepends=True)
    needs_line_trunc = len(lines) > MAX_OUTPUT_LINES
    needs_byte_trunc = total_bytes > MAX_OUTPUT_BYTES

    if not needs_line_trunc and not needs_byte_trunc:
        return TruncatedOutput(content=stdout, total_bytes=total_bytes)

    persisted_path = _persist_to_file(stdout)

    preview_lines = lines[:MAX_OUTPUT_LINES]
    preview = "".join(preview_lines)
    if len(preview.encode("utf-8", errors="replace")) > MAX_OUTPUT_BYTES:
        preview = preview[:MAX_OUTPUT_BYTES]

    truncation_notice = (
        f"\n\n[Output truncated: {len(lines)} lines, {total_bytes} bytes total. "
        f"Full output saved to: {persisted_path}]"
    )

    return TruncatedOutput(
        content=preview + truncation_notice,
        truncated=True,
        persisted_path=persisted_path,
        total_bytes=total_bytes,
    )


def _persist_to_file(content: str) -> str:
    output_dir = os.path.join(tempfile.gettempdir(), "chcode-output")
    os.makedirs(output_dir, exist_ok=True)
    file_id = uuid.uuid4().hex[:8]
    path = os.path.join(output_dir, f"output-{file_id}.txt")

    encoded = content.encode("utf-8", errors="replace")
    if len(encoded) > MAX_PERSISTED_BYTES:
        encoded = encoded[:MAX_PERSISTED_BYTES]

    with open(path, "wb") as f:
        f.write(encoded)

    return path
