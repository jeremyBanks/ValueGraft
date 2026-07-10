"""Discrete next-action-match metric for SWE-Gym coding trajectories (CPU, no torch).

The teacher-forced logprob is a PROXY; this adds a behavioral, checkable metric that
does NOT need Docker / test execution: parse the model's FREE generation for the
action it emitted (tool name, file path, command) and ask whether it matches the
TRUE next action. Shared by the harness (records it per arm at run time) and the
offline backfill/analysis (recomputes it on already-persisted `gen` fields).

An OpenHands/Qwen action is `<function=NAME> <parameter=KEY>VALUE</parameter> ... `.
Some gold turns are prose (no call) -> no discrete action, excluded from the metric.

Match components (all recorded so the analysis can pick a strictness):
  tool_match : same function name (str_replace_editor / execute_bash / ...)
  path_match : same `path` parameter (only when the gold action has a path)
  cmd_match  : same `command` parameter (editor verb like view/str_replace, or the
               bash command string), whitespace-normalized
  match      : COMPOSITE -- tool_match AND (path_match if gold has a path) AND
               (cmd_match if gold has a command). The strict "correct next action".
"""
from __future__ import annotations

import re

_FUNC = re.compile(r"<function=([A-Za-z0-9_]+)>(.*?)</function>", re.S)
_FUNC_OPEN = re.compile(r"<function=([A-Za-z0-9_]+)>(.*)", re.S)  # truncated gen
_PARAM = re.compile(r"<parameter=([A-Za-z0-9_]+)>(.*?)</parameter>", re.S)


def _norm(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def extract_action(text):
    """Parse the FIRST emitted action from an action string / free generation.
    Returns {tool, path, command} or None if no function call is present."""
    text = str(text or "")
    m = _FUNC.search(text) or _FUNC_OPEN.search(text)
    if not m:
        return None
    tool = m.group(1)
    params = {k: _norm(v) for k, v in _PARAM.findall(m.group(2))}
    return {"tool": tool,
            "path": params.get("path"),
            "command": params.get("command")}


def match_action(gold, arm):
    """Component + composite match of an ARM's emitted action against the GOLD.
    Returns a dict of booleans (all False / has_action False when arm has no call).
    `gold` and `arm` are extract_action() outputs (arm may be None)."""
    out = {"has_action": bool(arm), "tool_match": False,
           "path_match": None, "cmd_match": None, "match": False}
    if not gold or not arm:
        return out
    out["tool_match"] = (arm["tool"] == gold["tool"])
    if gold["path"] is not None:
        out["path_match"] = (arm["path"] == gold["path"])
    if gold["command"] is not None:
        out["cmd_match"] = (arm["command"] == gold["command"])
    out["match"] = (out["tool_match"]
                    and (gold["path"] is None or out["path_match"])
                    and (gold["command"] is None or out["cmd_match"]))
    return out


def gold_has_action(gold):
    return gold is not None
