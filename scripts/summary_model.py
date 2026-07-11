#!/usr/bin/env python3
"""Stable stdin/stdout wrapper for Claude or Codex summary generation.

The notes pipelines historically accepted arbitrary commands.  This module keeps
that escape hatch while providing a reproducible provider/model abstraction.
Codex runs are ephemeral and outside the repository so summary jobs do not add
their own sessions to the transcript archive or inherit repository instructions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PROVIDER = "codex"
DEFAULT_CODEX_MODEL = "gpt-5.6-luna"
DEFAULT_CODEX_REASONING = "medium"
DEFAULT_CLAUDE_MODEL = "sonnet"
PROVIDERS = ("codex", "claude")


@dataclass(frozen=True)
class SummaryModelSpec:
    provider: str
    model: str
    reasoning: str | None = None

    def provenance(self) -> dict[str, str]:
        result = {"provider": self.provider, "model": self.model}
        if self.reasoning:
            result["reasoning"] = self.reasoning
        return result


def resolve_spec(provider: str, model: str | None, reasoning: str | None) -> SummaryModelSpec:
    if provider not in PROVIDERS:
        raise ValueError(f"unsupported summary provider: {provider!r}")
    if provider == "codex":
        return SummaryModelSpec(
            provider=provider,
            model=model or DEFAULT_CODEX_MODEL,
            reasoning=reasoning or DEFAULT_CODEX_REASONING,
        )
    return SummaryModelSpec(provider=provider, model=model or DEFAULT_CLAUDE_MODEL)


def wrapper_argv(spec: SummaryModelSpec) -> list[str]:
    argv = [sys.executable, str(Path(__file__).resolve()), "--provider", spec.provider, "--model", spec.model]
    if spec.reasoning:
        argv.extend(["--reasoning", spec.reasoning])
    return argv


def wrapper_command(spec: SummaryModelSpec) -> str:
    return shlex.join(wrapper_argv(spec))


def custom_command_provenance(command: str | list[str]) -> dict[str, str]:
    rendered = command if isinstance(command, str) else "\0".join(command)
    return {
        "provider": "custom",
        "command_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
    }


def require_cli(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"summary provider CLI not found on PATH: {name!r}")
    return path


def resolve_codex_cli() -> str:
    override = os.environ.get("VALUEGRAFT_CODEX_CLI")
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise RuntimeError(f"VALUEGRAFT_CODEX_CLI is not a file: {path}")
        return str(path)
    bundled = Path("/Applications/ChatGPT.app/Contents/Resources/codex")
    if bundled.is_file():
        return str(bundled)
    return require_cli("codex")


def run_codex(spec: SummaryModelSpec, prompt: str) -> str:
    codex = resolve_codex_cli()
    with tempfile.TemporaryDirectory(prefix="valuegraft-luna-summary-") as tmp:
        tmp_path = Path(tmp)
        output_path = tmp_path / "last-message.md"
        cmd = [
            codex,
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--model",
            spec.model,
            "-c",
            f'model_reasoning_effort={json.dumps(spec.reasoning or DEFAULT_CODEX_REASONING)}',
            "--output-last-message",
            str(output_path),
            "-",
        ]
        proc = subprocess.run(
            cmd,
            cwd=tmp_path,
            input=prompt,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Codex summarizer failed with exit {proc.returncode}\nSTDERR:\n{proc.stderr}"
            )
        if not output_path.exists():
            raise RuntimeError("Codex summarizer did not write its final message")
        return output_path.read_text(encoding="utf-8").strip() + "\n"


def run_claude(spec: SummaryModelSpec, prompt: str) -> str:
    claude = require_cli("claude")
    cmd = [
        claude,
        "--print",
        "--model",
        spec.model,
        "--no-session-persistence",
        "--permission-mode",
        "dontAsk",
        "--disallowedTools",
        "*",
    ]
    proc = subprocess.run(cmd, input=prompt, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Claude summarizer failed with exit {proc.returncode}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout.strip() + "\n"


def run(spec: SummaryModelSpec, prompt: str) -> str:
    if spec.provider == "codex":
        return run_codex(spec, prompt)
    return run_claude(spec, prompt)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=PROVIDERS, default=DEFAULT_PROVIDER)
    parser.add_argument("--model")
    parser.add_argument("--reasoning", default=DEFAULT_CODEX_REASONING)
    args = parser.parse_args()
    spec = resolve_spec(args.provider, args.model, args.reasoning)
    sys.stdout.write(run(spec, sys.stdin.read()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
