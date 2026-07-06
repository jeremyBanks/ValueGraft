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


if __name__ == "__main__":
    main()
