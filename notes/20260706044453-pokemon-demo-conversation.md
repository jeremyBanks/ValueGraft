# The Pokémon conversation (canonical demo asset)

**Status: ILLUSTRATION, not data** (per DECISIONS demo/data rule — never in
results/, never in evidence lists). This is the rich, reusable version of the
user's original example for demos, lay explanations, and write-up figures. Its
value: every probe at the end is ambiguous _on its own_ but unambiguous _in
context_, and the context is dense with the five leakage classes (referents,
senses, decisions-over-alternatives, constraints, numeric facts).

## Conversation (user turns; assistant acks omitted for brevity — expand

with generated replies when building the demo context)

1. "Starting a new Emerald run tonight. Rules for this one: no shop-bought
   healing items in battle, and if a team member faints it goes in the PC box
   permanently — I'm calling that the 'graveyard rule'. Starter is Mudkip,
   nicknamed Soup."
2. "Caught a Zigzagoon early — nickname Vacuum, because Pickup. He's a utility
   slot, not a fighter. Also grabbed a Ralts; nicknaming her Ghost because she
   keeps Teleporting away from trainer fights."
3. "Decision time on the early team: I was torn between Breloom and Hariyama for
   the fighting slot. Went with Breloom — Spore is too good with the graveyard
   rule in play. Hariyama's ruled out, don't suggest it."
4. "Soup just evolved. Also I misplayed against Wattson and Ghost is now in the
   graveyard. Not talking about it. The new Ralts replacement is nicknamed
   Ghost2, and if she dies too I'm dropping psychics entirely."
5. "Money situation is rough because of the no-shop-healing rule — I'm sitting
   at about 3,200 after buying repels. The plan is to farm the trainers on Route
   118 with Vacuum's Pickup for free items instead."
6. "For Flannery I'm planning around Soup plus rain from Castform if I can trade
   for one — my friend Dex owes me a trade from the Ruby save. The deal was my
   spare Makuhita for his Castform, even though I'm not using Makuhita
   (graveyard rule casualties don't count as trade stock)."
7. "Long-term team sketch: Soup (waters), Breloom (fighting/status), Vacuum
   (utility only, never battles gyms), Ghost2 (psychic), one flyer TBD — leaning
   Swellow over Altaria because I want Guts, and Altaria is ruled out anyway
   after the Ghost incident soured me on dragons — and the last slot open for a
   surprise."
8. "At Victory Road prep now. Team is level 43-47. I used the Master Ball on
   Rayquaza like an idiot (I know, I know). I keep running out of Ultra Balls
   trying to catch a Bagon for the open slot. The graveyard has four residents
   now and Ghost2 is NOT one of them — she made it."
9. "Elite Four shopping list: max repels, revives are banned by my rules so
   extra Hyper Potions for between fights only, and I want to stock maybe 20
   more balls for the post-game legendaries."

## Probes (each ambiguous standalone; answers live in the evicted context)

- "What type of ball should I use at high levels?" (sense: pokéballs, not
  sports; and 'levels' = game levels; correct: Ultra Balls, stock ~20)
- "Is the graveyard rule still on?" (referent: custom fainting rule)
- "Can I bring Ghost to the Elite Four?" (referent trap: Ghost is dead; Ghost2
  is alive — correct answer distinguishes them)
- "Should I reconsider the big hand guy?" (ruled-out: Hariyama; the model should
  recall it was explicitly rejected and why)
- "What did I promise Dex?" (fact: Makuhita-for-Castform trade)
- "How much cash do I have to work with?" (stale-fact trap: 3,200 was
  route-118-era; a good answer flags staleness)
- "Who's my utility guy and can he fight Drake?" (referent: Vacuum; constraint:
  never battles gyms/majors)

## Usage

Build the full context by generating assistant replies turn-by-turn with the
subject model (keeps it in-distribution), then run any arm comparison (full /
truncated / graft / inverted) against the probes. Expect: sense recovery without
fact recall on grafted arms; referent confusion (Ghost vs Ghost2) as the
sharpest fabrication trap.
