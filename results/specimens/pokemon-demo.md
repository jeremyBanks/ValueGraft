# The Pokémon demo: four ways to remember (or not) a conversation

**Setup.** The model first reads this conversation context:

> I've been replaying Pokemon Emerald all week and I'm finally at Victory
> Road. My team is around level 45 but the Elite Four are going to be brutal.
> I already used my Master Ball on Rayquaza like an idiot, and I keep running
> out of Ultra Balls trying to catch a Bagon. Also my Breloom keeps fainting
> to Psychic types. Anyway, before the Elite Four I want to stock up properly
> at the Pokemart and maybe catch a few more team members in the tall grass
> near the entrance.

Then it's asked, with no restatement: **"So what type of ball should I use at
high levels?"** — a sentence that is ambiguous on its own (pokeballs? sports?).

In conditions 2–4 the Pokémon context is REPLACED by meaningless filler of the
same length ("Just thinking out loud about my week, nothing specific…"), so
the question sits at identical token positions. Condition 3 copies the
question's internal "meaning vectors" (attention values) from the full-context
reading onto the identical question tokens; condition 4 pushes them the same
distance in the OPPOSITE direction.

## 1. Full context (the model remembers everything)
> "Great question — and you're absolutely right to be strategic before facing
> the Elite Four in *Pokémon Emerald*! … **the best ball to use is the Ultra
> Ball** — especially for catching Pokémon in the wild…"

Perfect: right game, right item, tied to the player's actual situation.

## 2. Context deleted (what naive compaction risks)
> "…'high levels' could mean many things…"

The model notices the ambiguity and flounders — it has only the words.

## 3. Context deleted + our technique (contextual values transplanted)
> "Since you're asking about *balls* and *high levels*, I'm going to assume
> you're referring to **gaming or strategy games** (like *PUBG*, …)"

Partial recovery: the transplanted vectors carried the *sense* — this is a
GAMING question — but not the specific facts (it can't name Emerald or Ultra
Balls; those were in the deleted text, not in the question's vectors). This is
exactly the honest shape of our experimental results: meaning-level recovery,
not fact-level recall.

## 4. Context deleted + technique INVERTED (vectors pushed away from context)
> "Could you clarify what you mean by 'high levels'? For example: Are you
> referring to a high level of sports (like professional **tennis, basketball,
> or soccer**)?"

Pushed away from the gaming reading, the model lands on… sports equipment —
the "brands of basketballs" failure mode, manufactured on demand. (It also
stumbles slightly at the seam — "nothingSo what type of ball…" — the vectors
there no longer quite fit their own sentence.)

*(Qwen3-4B, greedy decoding, α=±1 value transplant on the question tokens
only; contexts position-matched by construction. One example, illustrative
not statistical.)*
