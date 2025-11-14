import json
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import ensure_dir


@dataclass
class ToolResult:
    tool: str
    ok: bool
    result: str


class MockOSEnv:
    """
    A tiny sandboxed environment rooted in a temporary directory under outputs/samples/.
    Provides a few tools suitable for qualitative comparisons.
    """

    def __init__(self, session_dir: Optional[str] = None) -> None:
        if session_dir is None:
            tmp_root = Path("outputs/samples")
            ensure_dir(str(tmp_root))
            self._tmp = tempfile.TemporaryDirectory(dir=tmp_root)
            self.root = Path(self._tmp.name)
        else:
            ensure_dir(session_dir)
            self.root = Path(session_dir)
            self._tmp = None  # not managed

    def _abs(self, p: str) -> Path:
        pth = Path(p)
        if pth.is_absolute():
            # disallow escaping the sandbox
            raise ValueError("Absolute paths are not allowed in MockOSEnv.")
        return self.root / pth

    def list_files(self, path: str) -> ToolResult:
        try:
            target = self._abs(path)
            if not target.exists():
                return ToolResult("list_files", False, f"Path not found: {path}")
            if target.is_file():
                return ToolResult("list_files", True, str(target))
            files = sorted([str(p) for p in target.iterdir()])
            return ToolResult("list_files", True, "\n".join(files))
        except Exception as e:
            return ToolResult("list_files", False, str(e))

    def read_file(self, path: str, max_bytes: Optional[int] = None) -> ToolResult:
        try:
            target = self._abs(path)
            if not target.exists() or not target.is_file():
                return ToolResult("read_file", False, f"File not found: {path}")
            data = target.read_bytes()
            if max_bytes is not None:
                data = data[: max(0, max_bytes)]
            try:
                return ToolResult("read_file", True, data.decode("utf-8"))
            except Exception:
                return ToolResult("read_file", False, "File is not valid UTF-8")
        except Exception as e:
            return ToolResult("read_file", False, str(e))

    def write_file(self, path: str, content: str, append: bool = False) -> ToolResult:
        try:
            target = self._abs(path)
            ensure_dir(str(target.parent))
            mode = "a" if append else "w"
            with open(target, mode, encoding="utf-8") as f:
                f.write(content)
            return ToolResult("write_file", True, f"Wrote {len(content)} chars to {path}")
        except Exception as e:
            return ToolResult("write_file", False, str(e))

    def execute_bash(self, command: str, timeout_sec: int = 5) -> ToolResult:
        try:
            # Disallow dangerous characters; run with cwd=self.root
            args = shlex.split(command)
            proc = subprocess.run(
                args,
                cwd=self.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout_sec,
                check=False,
                text=True,
            )
            return ToolResult("execute_bash", proc.returncode == 0, proc.stdout.strip())
        except Exception as e:
            return ToolResult("execute_bash", False, str(e))

    def run_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        if tool_name == "list_files":
            return self.list_files(arguments.get("path", "."))
        if tool_name == "read_file":
            return self.read_file(arguments.get("path", ""), arguments.get("max_bytes"))
        if tool_name == "write_file":
            return self.write_file(arguments.get("path", ""), arguments.get("content", ""), arguments.get("append", False))
        if tool_name == "execute_bash":
            return self.execute_bash(arguments.get("command", ""), arguments.get("timeout_sec", 5))
        return ToolResult(tool_name, False, "Unknown tool")


