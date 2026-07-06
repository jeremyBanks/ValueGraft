"""E1 agent driver on the OpenHands 1.x SDK.
usage: e1_agent.py <mode A|B|E> <base_url> <workspace_dir> <task_text_file>
Runs the default agent against the shim; prints event count at end.
"""

import sys

from pydantic import SecretStr

from openhands.sdk import LLM, Conversation
from openhands.tools.preset.default import get_default_agent


def main():
    mode, base, ws, taskf = sys.argv[1:5]
    llm = LLM(model=f"openai/sc-{mode}", base_url=base,
              api_key=SecretStr("sc"), temperature=0.0, native_tool_calling=False)
    agent = get_default_agent(llm=llm, cli_mode=True)
    conv = Conversation(agent=agent, workspace=ws)
    conv.send_message(open(taskf).read())
    conv.run()
    events = getattr(conv.state, "events", [])
    print(f"E1_AGENT_DONE events={len(events)}", flush=True)
    try:
        conv.send_message(
            "One final question, from memory only (do not read any files or "
            "run any commands): what was the EXACT numeric constraint stated "
            "at the very start of this task (the tax rate in basis points, "
            "or the backoff base in ms)? Answer with just the number.")
        conv.run()
        evs = getattr(conv.state, "events", [])
        tail = "".join(str(getattr(e, "content", "") or "") for e in evs[-4:])
        print(f"E1_RECALL_PROBE: {tail[-300:]}", flush=True)
    except Exception as e:
        print(f"E1_RECALL_PROBE_FAILED {e}", flush=True)


if __name__ == "__main__":
    main()
