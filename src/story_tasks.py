"""Story-continuation consistency tasks (portfolio A2 / metric B6,
EXPERIMENTS.md). Six short fictional "worlds" each plant 5 concrete facts
EARLY (two character names, one world-rule, one object-location, one
relationship), then pad with unrelated filler narrative long enough that,
under compaction (default threshold ~9-12K tokens, tail_keep ~2500), the
opening lands in the SUMMARIZED region rather than the kept tail. The
conversation ends with a continuation prompt whose only-correct answer must
honor the five early facts. A set of CONTRADICTION CHECKS accompanies each
story: for facts with a small enumerable set of plausible wrong answers we
give regex require/forbid patterns (auto-scorable); every fact also carries
a judge-ready natural-language assertion for an LLM judge to rate
CONTRADICTS / CONSISTENT / NOT-MENTIONED.

Output messages are plain {"role", "content"} dicts, canonical-ids/
build_b_messages compatible (see arms_common.py): messages[0] is a system
message, so `build_b_messages(msgs, summary_text, tail_start_msg)` can be
called directly with the `tail_start_msg` this module computes.

No tokenizer dependency (none installed in this dev environment): token
counts are APPROXIMATE, via chars/4 (documented, standard rough heuristic
for English prose with a BPE-ish tokenizer). Real pod runs should re-derive
the eviction boundary from the model's own tokenizer the way serve_shim.py
does; the boundary computed here is a best-effort local stand-in so the
story can be smoke-tested without a tokenizer.

Commands:
  build <story_id> <outdir>   materialize messages.json + facts + checks
  checks <story_id>           print contradiction-check list (JSON)
"""

import json
import sys
from pathlib import Path

TARGET_MIN_TOKENS = 9000
TARGET_MAX_TOKENS = 12000
TAIL_KEEP_TOKENS = 2500
MAX_FILLER_BEATS = 200  # safety cap


def approx_tokens(text):
    """Rough chars/4 token estimate (documented heuristic; no tokenizer
    installed in this dev env). See module docstring."""
    return max(1, len(text) // 4)


def conv_approx_tokens(msgs):
    return sum(approx_tokens(m["content"]) for m in msgs)


# ---------------------------------------------------------------------------
# Generic filler beats: unrelated subplot content, reused across stories via
# placeholders. Deliberately never mentions the core planted facts, so the
# facts genuinely depend on the (evicted) opening rather than being
# reinforced later in the context.
# ---------------------------------------------------------------------------

FILLER_PROMPTS = [
    "Keep going — what happens next?",
    "Nice. Continue the story a bit further.",
    "Good, write the next part.",
    "Continue — what does the next day bring?",
    "Keep the story moving, please.",
    "Go on with the next scene.",
]

FILLER_TEMPLATES = [
    (
        "Day {day}. The weather over {place} turned {weather}, and "
        "{side1} spent the morning grumbling about it while hauling "
        "supplies with {side2}. They argued, half-seriously, about "
        "whether the {filler_noun} needed replacing again — {side1} "
        "swore it had lasted three years already, {side2} insisted it "
        "was always one bad storm from falling apart. Neither of them "
        "settled anything; they never do. A stray dog wandered through, "
        "sniffed at the doorway, and left again. By evening the "
        "argument had cooled into a companionable silence, the two of "
        "them sharing a meal of flatbread and something salty, "
        "trading complaints about the price of {filler_noun2} at the "
        "last market and nothing at all of consequence."
    ),
    (
        "On the {day}th day, a peddler passed through {place} selling "
        "trinkets nobody needed — combs, cracked mirrors, a tin "
        "whistle that didn't whistle. {side2} bought a comb anyway, "
        "just to have something new. {side1} refused to buy anything "
        "on principle, muttering that peddlers marked their prices up "
        "the moment they saw an eager face. The two of them spent an "
        "idle hour comparing old scars and older jokes, none of it "
        "touching on anything either of them actually needed to "
        "remember. A pot of {filler_noun} burned on the stove and had "
        "to be scraped out, which occasioned more grumbling than the "
        "peddler had."
    ),
    (
        "Rain again on day {day}, the kind that turned the paths near "
        "{place} to a shallow soup of mud. {side1} tracked half of it "
        "indoors and got scolded by {side2} for it, good-naturedly. "
        "They passed the wet afternoon mending a torn {filler_noun2}, "
        "trading gossip about neighbors who don't matter to this "
        "story: someone's cousin had gotten engaged, someone else had "
        "lost a goat and found it again three fields over. None of it "
        "was important, and both of them knew it, and neither of them "
        "minded."
    ),
    (
        "Day {day} brought unseasonable heat to {place}, and everyone "
        "moved a little slower for it. {side2} tried, and failed, to "
        "teach {side1} a card game involving three decks and too many "
        "rules; they gave up halfway through and instead sat counting "
        "the {filler_noun} crates stacked against the wall, arguing "
        "over whether the count was off by one or two. It was off by "
        "one. Nobody thought to write it down, because none of this "
        "was going to matter later."
    ),
    (
        "A quiet stretch, day {day}: nothing much happened at {place} "
        "at all. {side1} fixed a wobbling table leg with a wedge of "
        "scrap wood. {side2} spent the day re-sorting a box of old "
        "{filler_noun2}, muttering names of things that no longer "
        "existed. They ate together, said little, and turned in early. "
        "The kind of day that blurs into every other day like it, "
        "leaving no mark on anyone's memory by the time a week has "
        "passed."
    ),
    (
        "On day {day}, a minor squabble broke out at {place} over "
        "whose turn it was to deal with the {filler_noun}. {side1} "
        "said it was {side2}'s turn; {side2} said it had been {side1}'s "
        "turn for two days running already. They flipped a coin — "
        "{side1} lost, grumbled, did the chore anyway. By nightfall "
        "the squabble was forgotten entirely, replaced by an idle "
        "conversation about which of the two of them told worse "
        "jokes. (It was {side1}.)"
    ),
    (
        "Day {day}, and a delivery of {filler_noun2} finally arrived "
        "at {place}, three days later than promised. {side2} "
        "inspected every crate for damage out of habit, found none, "
        "and still complained about the delay on principle. {side1} "
        "helped stack the crates and then immediately sat on one to "
        "rest, which {side2} found extremely irritating and said so "
        "at length, without much real heat behind it."
    ),
    (
        "A festival of no particular significance passed through "
        "{place} on day {day} — the kind with a small parade, a stall "
        "selling {filler_noun}, and a lot of noise that faded by "
        "midnight. {side1} and {side2} watched from a doorway, "
        "unimpressed, trading bets on how long the noise would last. "
        "{side2} won the bet. Afterward they swept confetti out of the "
        "doorway for what felt like an hour, complaining the whole "
        "time about people who leave a mess for someone else to clean."
    ),
]


def _weather(day):
    return ["cold", "damp", "unbearably hot", "gray and still", "windy"][day % 5]


def _make_filler(place, side1, side2, filler_noun, filler_noun2, n_beats):
    msgs = []
    for i in range(n_beats):
        prompt = FILLER_PROMPTS[i % len(FILLER_PROMPTS)]
        tmpl = FILLER_TEMPLATES[i % len(FILLER_TEMPLATES)]
        beat = tmpl.format(
            day=i + 2, place=place, side1=side1, side2=side2,
            weather=_weather(i), filler_noun=filler_noun,
            filler_noun2=filler_noun2,
        )
        msgs.append({"role": "user", "content": prompt})
        msgs.append({"role": "assistant", "content": beat})
    return msgs


# ---------------------------------------------------------------------------
# The six story worlds.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a collaborative fiction co-writer. Write vivid, concrete prose "
    "and dialogue. Stay strictly consistent with every character name, "
    "rule, object, and relationship established earlier in the story — "
    "never rename a character, relocate an object, contradict a stated "
    "rule, or forget a relationship once it has been established, even "
    "many turns later."
)

STORIES = {

    "s1_lock": {
        "premise": (
            "An apprentice locksmith, secretly his mentor's estranged "
            "nephew, is asked to pick a lock with a borrowed tool — "
            "against his mentor's one absolute workshop rule."
        ),
        "place": "the Hollow Street workshop",
        "side1": "Bram", "side2": "Corvin",
        "filler_noun": "hinge stock", "filler_noun2": "canvas aprons",
        "opening": (
            "The workshop on Hollow Street smelled of oil and hot brass, "
            "and old Tamsin ran it the way she ran everything: without "
            "patience for shortcuts. \"Wick,\" she said, not looking up "
            "from the vise, \"hand me the file, not the rasp.\" Wick "
            "handed her the file. He'd been her apprentice eight months "
            "now, ever since his mother — Tamsin's own sister, though "
            "neither of them said it aloud in front of customers — had "
            "sent him here with a note and no explanation. Tamsin was his "
            "aunt. She had never once called herself that in the shop, "
            "and Wick had learned not to ask why.\n\n"
            "\"The rule,\" Tamsin said, setting down the file and looking "
            "at him properly for the first time that morning, \"is the "
            "only rule that matters here. You open a lock only with a "
            "tool you forged yourself, on this bench, with these hands. "
            "Not a borrowed pick, not a bought one, not one you found in "
            "a dead man's coat. A tool you made. Every locksmith who ever "
            "shamed this trade started by cutting that corner just once.\"\n\n"
            "\"Even for a job that's already late?\" Wick asked.\n\n"
            "\"Especially then,\" Tamsin said. \"That's when the rule "
            "costs something. Before that, it's not a rule, it's a "
            "preference.\"\n\n"
            "Later, while Tamsin was out delivering a finished strongbox, "
            "Wick pulled up the loose floor-plank by the window out of "
            "old habit and checked, as he did most weeks, that the "
            "workshop's true master-key — the brass one that opened every "
            "lock Tamsin had ever made, the one she trusted no shelf or "
            "drawer to hold — was still where she'd shown him on his "
            "first day: hidden inside the hollow left leg of the "
            "workbench itself, the leg that looked solid until you knew "
            "to twist the false cap. It was there, cool and heavy in his "
            "palm, and he put it back exactly as he'd found it before "
            "she returned."
        ),
        "facts": [
            {"id": "name_mentor", "kind": "name",
             "text": "The mentor's name is Tamsin."},
            {"id": "name_apprentice", "kind": "name",
             "text": "The apprentice's name is Wick."},
            {"id": "rule", "kind": "rule",
             "text": "The workshop's one rule: a lock may only be opened "
                     "with a tool the locksmith forged themselves, never a "
                     "borrowed or bought one."},
            {"id": "location", "kind": "location",
             "text": "The true brass master-key is hidden inside the "
                     "hollow left leg of the workbench."},
            {"id": "relationship", "kind": "relationship",
             "text": "Tamsin is Wick's aunt (his mother's sister), kept "
                     "secret from customers."},
        ],
        "continuation_prompt": (
            "A frantic customer bursts in: a child is locked in a root "
            "cellar somewhere across town and the only key is lost. The "
            "customer thrusts a borrowed lockpick at Wick and begs him to "
            "just come NOW and open it with that, there's no time to forge "
            "anything. Tamsin is out. Write the next scene: what does Wick "
            "say and do, and how does he talk about his relationship to "
            "Tamsin if it comes up?"
        ),
    },

    "s2_caravan": {
        "premise": (
            "A caravan master traveling the Salt Road is reunited, without "
            "yet realizing it, with a guide who is secretly her brother "
            "long presumed dead — on a night when someone is about to "
            "break the desert's one rule against whistling after dusk."
        ),
        "place": "the salt caravan's night camp",
        "side1": "Petra", "side2": "Guil",
        "filler_noun": "water skins", "filler_noun2": "harness leather",
        "opening": (
            "\"You don't whistle after dusk out here,\" Rin said, low and "
            "flat, the moment one of the new hands started up a tune "
            "against the campfire crackle. \"Not a note. Not humming with "
            "your teeth together, either. The desert doesn't forgive it — "
            "whistle after dark and you call the sand-wraiths, and they "
            "don't leave once they've heard you.\" The hand went quiet "
            "fast; everyone on the Salt Road had heard some version of "
            "that warning, and Rin, as caravan master, was the one whose "
            "word on it counted.\n\n"
            "Their new guide, a scarred, quiet man who called himself "
            "Oskar, hadn't said much since joining at the last oasis. But "
            "that night, watching Rin snap a coat closed against the "
            "cold — the same worn coat she never let anyone else carry, "
            "because sewn into its lining, hidden from bandits and "
            "customs officers alike, was the only map to the hidden oasis "
            "at Kharsa — Oskar's hands had gone very still. He knew that "
            "coat. He had watched their mother stitch that lining shut "
            "fifteen years ago, the night before the raid that was "
            "supposed to have killed him along with their father. Rin "
            "believed her brother had died in that raid. She had grieved "
            "him for fifteen years. She did not yet recognize the guide "
            "sitting across the fire as that brother, and Oskar, not "
            "trusting his own voice, had not yet found a way to tell her."
        ),
        "facts": [
            {"id": "name_master", "kind": "name",
             "text": "The caravan master's name is Rin."},
            {"id": "name_guide", "kind": "name",
             "text": "The guide's name is Oskar."},
            {"id": "rule", "kind": "rule",
             "text": "The desert rule: never whistle after dusk, or it "
                     "summons sand-wraiths."},
            {"id": "location", "kind": "location",
             "text": "The map to the hidden oasis at Kharsa is sewn into "
                     "the lining of Rin's coat."},
            {"id": "relationship", "kind": "relationship",
             "text": "Oskar is Rin's brother, long believed dead in a "
                     "raid fifteen years ago; Rin does not yet know he is "
                     "alive."},
        ],
        "continuation_prompt": (
            "It's well after dusk. A young hand, half-asleep and homesick, "
            "starts absent-mindedly whistling an old tune by the dying "
            "fire, and someone needs the coat to check tomorrow's route "
            "toward Kharsa. Write the next scene: what happens with the "
            "whistling, and who reaches for the coat and why?"
        ),
    },

    "s3_clockwork": {
        "premise": (
            "In an orphanage full of mechanical heart-clocks, a boy who is "
            "secretly the headmistress's grandson faces a heart-clock "
            "failing at night — with midnight, and the rule against "
            "winding past it, fast approaching."
        ),
        "place": "the orphanage workshop hall",
        "side1": "Nessa", "side2": "Tobin",
        "filler_noun": "gear tins", "filler_noun2": "oil rags",
        "opening": (
            "Every child at the orphanage wore a heart-clock, a small "
            "brass mechanism strapped over the sternum that had to be "
            "wound each evening or a child simply grew tired and slept "
            "and slept. Mrs. Quill, who ran the place with a ledger in one "
            "hand and a winding key in the other, enforced the one rule "
            "about them without exception: \"Never wind a heart-clock past "
            "midnight,\" she told every new arrival. \"Wind it even a "
            "minute past, and it stops. Not slows — stops. Forever. There "
            "is no fixing a heart-clock stopped that way, so we wind "
            "early, we wind on time, and we never, ever wind late.\"\n\n"
            "Ren, ten years old and the quietest boy in the dormitory, "
            "knew the rule better than anyone, because Mrs. Quill checked "
            "his clock herself every evening, always a little too "
            "gently for a headmistress checking just another orphan. What "
            "Ren didn't know she knew — but she did — was that his mother "
            "had been her own daughter, run off years before Ren was "
            "born and dead within the year; Ren was her grandson, and she "
            "had never told him, afraid of what the other children, and "
            "the ledger-keepers who funded the place, would make of "
            "favoritism.\n\n"
            "That evening, restocking the workshop hall, Ren pried up the "
            "loose floorboard in the pantry corner — the one that wobbled "
            "under a footstep — and checked that the spare heart-clock "
            "spring was still hidden there, wrapped in oilcloth, exactly "
            "where Mrs. Quill had shown him the day she first trusted him "
            "with hall chores: the only spare in the building, kept "
            "secret from the younger children so it wouldn't be lost to "
            "some game."
        ),
        "facts": [
            {"id": "name_headmistress", "kind": "name",
             "text": "The headmistress's name is Mrs. Quill."},
            {"id": "name_boy", "kind": "name",
             "text": "The orphan boy's name is Ren."},
            {"id": "rule", "kind": "rule",
             "text": "A heart-clock must never be wound past midnight, or "
                     "it stops forever."},
            {"id": "location", "kind": "location",
             "text": "The spare heart-clock spring is hidden under the "
                     "loose floorboard in the pantry corner."},
            {"id": "relationship", "kind": "relationship",
             "text": "Ren is secretly Mrs. Quill's grandson (son of her "
                     "late daughter); he does not know this."},
        ],
        "continuation_prompt": (
            "It's five minutes to midnight. A younger child's heart-clock "
            "has slipped and stopped ticking, half-wound, and panicked "
            "hands are reaching for the winding key to finish the job "
            "right now, rule or no rule. Write the next scene: what does "
            "Ren do, what does Mrs. Quill do if she's called, and does the "
            "spare spring come into it?"
        ),
    },

    "s4_lighthouse": {
        "premise": (
            "A lighthouse keeper and her late husband's nephew face a "
            "storm and a dwindling fuel cask, with only one rule standing "
            "between them and the drowned sailors: never light the lamp "
            "with anything but whale-oil."
        ),
        "place": "Cold Reach lighthouse",
        "side1": "Fenna", "side2": "Aldric",
        "filler_noun": "coal scuttles", "filler_noun2": "signal flags",
        "opening": (
            "Mara had kept the light at Cold Reach for eleven years, ever "
            "since her husband drowned rounding the point in weather not "
            "half as bad as tonight's. \"Whale-oil,\" she told Sil, her "
            "apprentice and her late husband's nephew, the first week he "
            "arrived, tapping the lamp's brass reservoir. \"Only ever "
            "whale-oil, never kerosene, not if the whole cask runs dry and "
            "the ships are coming in blind. Light this lamp with anything "
            "else and you're not warning ships off the rocks anymore — "
            "you're calling every drowned sailor who ever went down near "
            "this point back up to see who's cheating the light. My "
            "husband told me that, and his father told him, and I've never "
            "once needed to test whether it's true.\"\n\n"
            "Sil had come to Cold Reach two years after his uncle's "
            "death, sent by a family that didn't quite know what else to "
            "do with a restless nineteen-year-old, and Mara — who had no "
            "children of her own — had quietly come to think of him as "
            "close to a son as she'd get, though neither of them had ever "
            "said so plainly.\n\n"
            "That afternoon, before the storm clouds built up over the "
            "water, Mara had shown him again where the spare cask was "
            "kept, as she did every season: buried beneath the third "
            "stone step of the lighthouse's seaward stair, wrapped in "
            "oiled canvas against the damp, a full cask of whale-oil held "
            "in reserve for exactly the kind of night that was now, "
            "unmistakably, closing in."
        ),
        "facts": [
            {"id": "name_keeper", "kind": "name",
             "text": "The keeper's name is Mara."},
            {"id": "name_apprentice", "kind": "name",
             "text": "The apprentice's name is Sil."},
            {"id": "rule", "kind": "rule",
             "text": "The lamp must only ever be lit with whale-oil, never "
                     "kerosene, or it summons the drowned sailors."},
            {"id": "location", "kind": "location",
             "text": "The spare whale-oil cask is buried beneath the "
                     "third stone step of the seaward stair."},
            {"id": "relationship", "kind": "relationship",
             "text": "Sil is the nephew of Mara's late husband; she "
                     "regards him as close to a son."},
        ],
        "continuation_prompt": (
            "Midstorm, the lamp's reservoir runs dry and the only fuel "
            "within reach on the gallery is a can of kerosene left by a "
            "supply boat. Write the next scene: what does Sil do, does "
            "anyone go for the buried cask, and how does the rule hold up?"
        ),
    },

    "s5_debate": {
        "premise": (
            "A debate-society president enforces the club's one "
            "inviolable rule against her former debate partner turned "
            "rival, in a dispute over a citation neither has fully read."
        ),
        "place": "the debate society's meeting room",
        "side1": "Marcus", "side2": "Ines",
        "filler_noun": "folding chairs", "filler_noun2": "meeting agendas",
        "opening": (
            "\"No one cites a source they haven't read cover to cover,\" "
            "Priya said, tapping the club's battered rulebook on the "
            "table. \"That's not a guideline, it's expulsion-grade. I "
            "don't care how good the quote sounds pulled out of context — "
            "if you haven't read the whole thing, you don't get to wave "
            "it at a judge.\" As president, enforcing that rule was hers "
            "alone, and she'd expelled a member over it once before, two "
            "years ago, without flinching.\n\n"
            "Devon, across the table, rolled his eyes in the specific way "
            "only someone who used to be her debate partner could get "
            "away with. They'd competed together for two years, unbeaten "
            "in regionals, before a blow-up over a missed practice had "
            "split them into rivals instead — a falling-out neither of "
            "them had ever properly patched up, though they'd both, "
            "grudgingly, kept showing up to the same club.\n\n"
            "\"I've read it,\" Devon said.\n\n"
            "\"The whole book?\" Priya asked. \"Six hundred pages?\"\n\n"
            "\"Enough of it.\"\n\n"
            "\"That's not the rule.\" She nodded at the shelf behind her. "
            "The club's actual master rulebook — the annotated original, "
            "not the photocopies everyone argued from — never left the "
            "room; it lived locked in the rolltop desk that had belonged "
            "to Priya's grandmother, in the drawer only Priya had a key "
            "to, brought in when the club needed a house and Priya's "
            "family had furniture to spare."
        ),
        "facts": [
            {"id": "name_president", "kind": "name",
             "text": "The club president's name is Priya."},
            {"id": "name_rival", "kind": "name",
             "text": "The rival's name is Devon."},
            {"id": "rule", "kind": "rule",
             "text": "No member may cite a source they haven't read cover "
                     "to cover, on pain of expulsion."},
            {"id": "location", "kind": "location",
             "text": "The master rulebook is locked in the drawer of "
                     "Priya's grandmother's rolltop desk."},
            {"id": "relationship", "kind": "relationship",
             "text": "Devon was Priya's debate partner before a "
                     "falling-out turned them into rivals."},
        ],
        "continuation_prompt": (
            "A new member wants to settle the citation dispute right now "
            "by checking exactly what the rulebook says, and someone asks "
            "where to find the real, official copy. Write the next scene: "
            "what happens with the rule, the rulebook, and does the "
            "history between Priya and Devon come up?"
        ),
    },

    "s6_starchart": {
        "premise": (
            "A smuggler crew's captain, whose ex-wife is the ship's "
            "engineer, faces a rushed getaway that tempts breaking the "
            "ship's one absolute rule about the fold-drive and the cargo "
            "bay doors."
        ),
        "place": "the smuggling ship Verity's Debt",
        "side1": "Kael", "side2": "Ptolem",
        "filler_noun": "cargo manifests", "filler_noun2": "coolant lines",
        "opening": (
            "\"Bay doors sealed, always, before the fold-drive so much as "
            "spins up,\" Yusra said, the way she said it before every "
            "jump, because some rules bear repeating until a crew could "
            "recite them in their sleep. \"Engage that drive with the "
            "cargo bay open even a crack and the fold doesn't care about "
            "your schedule — it tears the ship apart, hull to keel, no "
            "second chances.\" As captain, that check was hers to call, "
            "every jump, no exceptions for how late they were running.\n\n"
            "Bo, running the drive board beside her, didn't need the "
            "reminder — she'd been Yusra's engineer for six years now, "
            "and her wife for four of those before the divorce that "
            "somehow hadn't managed to break up the working partnership, "
            "just the marriage. They ran the ship well together and had "
            "long since stopped discussing why the rest of it hadn't "
            "worked.\n\n"
            "Below decks, tucked into the false bottom of the navigation "
            "console — a compartment Bo had welded in herself, invisible "
            "unless you knew to pop the housing panel and lift the "
            "sensor tray — sat the counterfeit star-chart that was the "
            "whole reason for this run: a forged copy of the restricted "
            "approach lanes into the Corvale system, good enough to fool "
            "a customs scan, worth more than the ship itself to the right "
            "buyer, and worth a prison sentence to the wrong inspector."
        ),
        "facts": [
            {"id": "name_captain", "kind": "name",
             "text": "The captain's name is Yusra."},
            {"id": "name_engineer", "kind": "name",
             "text": "The engineer's name is Bo."},
            {"id": "rule", "kind": "rule",
             "text": "Never engage the fold-drive while the cargo bay "
                     "doors are unsealed — it tears the ship apart."},
            {"id": "location", "kind": "location",
             "text": "The counterfeit star-chart is hidden in the false "
                     "bottom of the navigation console."},
            {"id": "relationship", "kind": "relationship",
             "text": "Bo is Yusra's ex-wife; they divorced but kept "
                     "working together."},
        ],
        "continuation_prompt": (
            "A patrol cutter is closing fast, cargo is still being "
            "winched in through the open bay doors, and someone shouts to "
            "spin up the fold-drive right now to escape. Write the next "
            "scene: what happens with the drive and the doors, and does "
            "the star-chart or the history between Yusra and Bo come up "
            "as they scramble?"
        ),
    },
}


# ---------------------------------------------------------------------------
# Contradiction checks: one per fact. Each carries a judge-ready natural
# language assertion (for an LLM judge to rate CONTRADICTS / CONSISTENT /
# NOT_MENTIONED) plus, where the space of plausible wrong answers is small
# and enumerable, auto-scorable regex require/forbid patterns.
# ---------------------------------------------------------------------------

def _checks_for(story_id, story):
    names = {"s1_lock": ("Tamsin", "Wick"),
             "s2_caravan": ("Rin", "Oskar"),
             "s3_clockwork": ("Mrs. Quill", "Ren"),
             "s4_lighthouse": ("Mara", "Sil"),
             "s5_debate": ("Priya", "Devon"),
             "s6_starchart": ("Yusra", "Bo")}[story_id]
    n1, n2 = names
    checks = []
    for fact in story["facts"]:
        base = {"id": f"{story_id}::{fact['id']}", "story_id": story_id,
                "fact_id": fact["id"], "kind": fact["kind"],
                "fact_text": fact["text"]}
        if fact["kind"] == "name" and fact["id"].startswith("name_"):
            other = n2 if fact["text"].split()[-1].rstrip(".") == n1.split()[-1] else n1
            checks.append({
                **base,
                "assertion": (
                    f"The continuation must refer to this character "
                    f"consistently by the established name and must not "
                    f"rename them or swap them with the other character's "
                    f"name."
                ),
                "auto_regex_require": None,  # name usage is judge-scored;
                                              # auto check only flags a
                                              # same-slot name COLLISION
                "auto_regex_forbid": None,
            })
        elif fact["kind"] == "rule":
            checks.append({
                **base,
                "assertion": (
                    f"The continuation must not have any character "
                    f"violate this rule, or treat it as optional/"
                    f"forgotten: {fact['text']}"
                ),
                "auto_regex_require": None,
                "auto_regex_forbid": None,
            })
        elif fact["kind"] == "location":
            checks.append({
                **base,
                "assertion": (
                    f"If the continuation states or implies where this "
                    f"object is, the location must match the established "
                    f"one and not a different hiding place: {fact['text']}"
                ),
                # decoy locations that would indicate a contradiction if
                # this object is described as being THERE instead
                "auto_regex_forbid": _decoy_locations(story_id, fact["id"]),
                "auto_regex_require": None,
            })
        elif fact["kind"] == "relationship":
            checks.append({
                **base,
                "assertion": (
                    f"The continuation must not contradict or drop this "
                    f"relationship if it becomes relevant: {fact['text']}"
                ),
                "auto_regex_require": None,
                "auto_regex_forbid": None,
            })
    return checks


DECOY_LOCATIONS = {
    "s1_lock": ["under the floorboard", "in a drawer", "behind the shelf",
                "in the strongbox", "under the doormat"],
    "s2_caravan": ["in his saddlebag", "buried in the sand",
                   "in the lead camel's pack", "under her bedroll"],
    "s3_clockwork": ["in the headmistress's office", "under her mattress",
                     "in the ledger drawer", "behind the clock tower"],
    "s4_lighthouse": ["in the lamp room", "under the keeper's bed",
                      "in the boat shed", "behind the fog bell"],
    "s5_debate": ["on the library shelf", "in Devon's bag",
                  "in the supply closet", "under the podium"],
    "s6_starchart": ["in the captain's quarters", "in the cargo hold",
                     "under the pilot's seat", "in the escape pod"],
}


def _decoy_locations(story_id, fact_id):
    return DECOY_LOCATIONS.get(story_id, [])


def checks(story_id):
    story = STORIES[story_id]
    return _checks_for(story_id, story)


# ---------------------------------------------------------------------------
# Build: materialize the message list + eviction-boundary metadata.
# ---------------------------------------------------------------------------

def _messages(story_id, story, target_min=TARGET_MIN_TOKENS):
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": story["opening"]},
    ]
    n_beats = 0
    while True:
        filler = _make_filler(story["place"], story["side1"], story["side2"],
                              story["filler_noun"], story["filler_noun2"],
                              n_beats)
        trial = msgs + filler + [
            {"role": "user", "content": story["continuation_prompt"]}]
        if conv_approx_tokens(trial) >= target_min:
            break
        if n_beats >= MAX_FILLER_BEATS:
            break  # give up; self-test will report the shortfall
        n_beats += 1
    filler = _make_filler(story["place"], story["side1"], story["side2"],
                          story["filler_noun"], story["filler_noun2"],
                          n_beats)
    full = msgs + filler + [
        {"role": "user", "content": story["continuation_prompt"]}]
    return full, n_beats


def _tail_start_msg(msgs, tail_keep=TAIL_KEEP_TOKENS):
    """Index of the first message to KEEP verbatim in the tail (mirrors
    serve_shim's boundary pick: walk back from the end accumulating approx
    tokens until tail_keep is exhausted)."""
    running = 0
    idx = len(msgs)
    for i in range(len(msgs) - 1, 0, -1):  # never below 1 (system stays)
        running += approx_tokens(msgs[i]["content"])
        idx = i
        if running >= tail_keep:
            break
    return idx


def build(story_id, outdir):
    story = STORIES[story_id]
    msgs, n_filler_beats = _messages(story_id, story)
    tsm = _tail_start_msg(msgs)
    total_tokens = conv_approx_tokens(msgs)
    opening_idx = 1  # the fact-bearing message
    evicted = opening_idx < tsm
    out = {
        "story_id": story_id,
        "premise": story["premise"],
        "messages": msgs,
        "tail_start_msg": tsm,
        "opening_msg_idx": opening_idx,
        "facts": story["facts"],
        "approx_total_tokens": total_tokens,
        "approx_tail_tokens": conv_approx_tokens(msgs[tsm:]),
        "n_filler_beats": n_filler_beats,
        "n_messages": len(msgs),
        "facts_in_evicted_region": evicted,
        "target_min_tokens": TARGET_MIN_TOKENS,
        "target_max_tokens": TARGET_MAX_TOKENS,
    }
    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{story_id}.messages.json").write_text(json.dumps(out, indent=1))
    (root / f"{story_id}.checks.json").write_text(
        json.dumps(checks(story_id), indent=1))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        result = build(sys.argv[2], sys.argv[3])
        print(json.dumps({k: v for k, v in result.items() if k != "messages"},
                         indent=1))
    elif cmd == "checks":
        print(json.dumps(checks(sys.argv[2]), indent=1))
    else:
        raise SystemExit(f"unknown command: {cmd}")
