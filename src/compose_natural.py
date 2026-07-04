"""Compose 'natural' long conversations for the continuation-logprob metric.

Two model contexts: the assistant context (normal chat) and a user simulator
(role-play prompt). They cross-feed. The final HOLDOUT_MSGS messages are the
held-out continuation (>= ~1K tokens), scored teacher-forced by run_arms in
natural mode. Personas/agendas are hand-written; all turns are sampled.
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, "src")
from compose import ConversationBuilder

MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
N_TURNS = 16  # user turns total
HOLDOUT_MSGS = 4  # last messages held out (user, asst, user, asst)

PERSONAS = [
    {"id": "n01", "persona": "a high-school physics teacher redesigning their curriculum around project-based learning", "opening": "I teach high school physics and I'm rebuilding my whole curriculum around projects instead of lectures. Can we work through what that could look like over a semester?"},
    {"id": "n02", "persona": "a first-time landlord dealing with an inherited duplex and its two existing tenants", "opening": "I just inherited a duplex with two long-term tenants and I've never been a landlord. Help me figure out what I need to do first, legally and practically."},
    {"id": "n03", "persona": "a hobbyist genealogist trying to trace a great-grandmother who emigrated from Sicily around 1910", "opening": "I'm trying to trace my great-grandmother who left Sicily for New York around 1910. I have her married name and a ship name, maybe misspelled. Where do I start?"},
    {"id": "n04", "persona": "a software engineer training for their first marathon while recovering from a mild knee injury", "opening": "I'm 4 months out from my first marathon and my physio just cleared me after a minor knee issue. Help me build a training plan that won't wreck my knee."},
    {"id": "n05", "persona": "a small bakery owner deciding whether to expand into wholesale supply for local cafes", "opening": "I run a small bakery and three cafes have asked about wholesale. I'm tempted but terrified of the margins. Can we think this through?"},
    {"id": "n06", "persona": "a grad student preparing to defend a thesis on urban heat islands while managing committee politics", "opening": "My thesis defense on urban heat islands is in six weeks. My committee members disagree with each other about methodology. Help me prepare."},
    {"id": "n07", "persona": "a parent organizing a multi-family summer road trip with three cars and seven kids", "opening": "Three families, three cars, seven kids aged 4 to 15, ten days, national parks. I've been nominated chief planner. Save me."},
    {"id": "n08", "persona": "an amateur astronomer planning a backyard observatory build on a limited budget", "opening": "I want to build a small backyard observatory — roll-off roof, maybe a pier for my 8-inch scope. Budget is tight. Where do we start?"},
]

SIM_SYSTEM = (
    "You are role-playing {persona}. You are chatting with an AI assistant "
    "about your project. Stay in character: be specific, reference concrete "
    "details from earlier in the conversation, occasionally change subtopic, "
    "make decisions, express preferences, and rule things out. Write ONLY "
    "your next message to the assistant (2-5 sentences, conversational)."
)


def build_natural(model, tokenizer, spec, seed):
    mx.random.seed(seed)
    asst = ConversationBuilder(model, tokenizer,
                               "You are a helpful, knowledgeable assistant.")
    sim = ConversationBuilder(model, tokenizer,
                              SIM_SYSTEM.format(persona=spec["persona"]))
    user_text = spec["opening"]
    for turn in range(N_TURNS):
        reply = asst.user_turn_and_reply(user_text)
        if turn == N_TURNS - 1:
            break
        # simulator sees the assistant's reply as its "user" input
        user_text = sim.user_turn_and_reply(
            reply if turn > 0 else
            f"(You said: \"{spec['opening']}\")\nAssistant replied: {reply}"
        )
    return {
        "id": spec["id"],
        "persona": spec["persona"],
        "messages": asst.msgs,
        "holdout_msgs": HOLDOUT_MSGS,
        "sections": {"total_tokens": asst.n_tokens()},
        "meta": {"model": MODEL, "seed": seed, "n_turns": N_TURNS},
    }


def main():
    model, tokenizer = load(MODEL)
    outdir = Path("data/natural")
    outdir.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])
    for i, spec in enumerate(PERSONAS):
        if only and spec["id"] not in only:
            continue
        out = outdir / f"{spec['id']}.json"
        if out.exists():
            print(f"{spec['id']}: exists, skipping")
            continue
        conv = build_natural(model, tokenizer, spec, seed=2000 + i)
        json.dump(conv, open(out, "w"), indent=1, ensure_ascii=False)
        print(f"{spec['id']}: {len(conv['messages'])} msgs, "
              f"{conv['sections']['total_tokens']} tokens")


if __name__ == "__main__":
    main()
