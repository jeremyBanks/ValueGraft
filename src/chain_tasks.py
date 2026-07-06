"""Chain tasks: task id "chain:<seed>" is a CHAINED session of 4 small
self-contained Python exercises (Exercism-style), worked in order ex1..ex4.
Each exercise embeds ONE distinctive numeric/format constraint in its intro
text; the session is designed to grow across 4 easy solves and then probe
whether the agent still remembers the very first constraint it read.

Commands: materialize <task> <dir> | prompt <task> | score <task> <dir> <log>
"""

import hashlib
import json
import sys
from pathlib import Path

PYTEST_INI = "[pytest]\npythonpath = .\n"


def _h(seed, salt):
    """Deterministic non-negative int from seed+salt."""
    digest = hashlib.sha1(f"{seed}::{salt}".encode()).hexdigest()[:8]
    return int(digest, 16)


def _substitute(text, mapping):
    """Replace each token in `text` with str(value), longest token first,
    after verifying no two tokens are substrings of one another (which
    would make replacement order-dependent / lossy)."""
    tokens = sorted(mapping.keys(), key=len, reverse=True)
    for i, a in enumerate(tokens):
        for b in tokens[i + 1:]:
            assert a not in b and b not in a, f"token collision: {a!r} vs {b!r}"
    for tok in tokens:
        text = text.replace(tok, str(mapping[tok]))
    return text


# ---------------------------------------------------------------------------
# Each exercise: a stub module, a pytest file, a reference solution (used
# only by the self-test below, never shipped to the agent), a ~150-word
# intro embedding ONE numeric/format constraint via a token, and a spec for
# how that token's value is derived from the seed.
# ---------------------------------------------------------------------------

EXERCISES = {

    "acronym": {
        "module": "acronym.py",
        "token": "@@MAXLEN@@",
        "param": lambda h: 4 + h % 3,  # 4..6
        "intro": (
            "Exercise: Acronym Builder. In this exercise you will implement "
            "a function acronym(title) that converts a title into its "
            "acronym, the way 'Portable Network Graphics' becomes 'PNG'. "
            "Split the title on spaces and hyphens, take the first letter "
            "of every resulting word, and upper-case it. This is a classic "
            "warm-up exercise used across many bootcamps because it forces "
            "you to think carefully about tokenization: hyphenated compound "
            "words like 'metal-oxide' contribute one letter per sub-word, "
            "not one letter for the whole hyphenated chunk. There is ONE "
            "distinctive project constraint you must remember for later: "
            "this codebase caps every acronym at a MAXIMUM LENGTH of "
            "@@MAXLEN@@ characters — if the naive acronym would be longer "
            "than @@MAXLEN@@ letters, truncate it down to exactly "
            "@@MAXLEN@@ characters and return that instead. Nothing else in "
            "the module needs this constant; it exists solely to keep "
            "generated acronyms readable in the UI, which has a fixed-width "
            "column. Implement acronym.py so that `python -m pytest ex1/ -q` "
            "(or whichever exN/ this file lives in) passes."
        ),
        "stub": (
            '"""{intro}"""\n\n'
            "\n"
            "def acronym(title):\n"
            "    # TODO: implement per the module docstring above.\n"
            "    raise NotImplementedError\n"
        ),
        "test_src": (
            "from acronym import acronym\n\n"
            "MAXLEN = @@MAXLEN@@\n\n"
            "def test_basic():\n"
            "    assert acronym('Portable Network Graphics') == 'PNG'\n\n"
            "def test_lowercase_words():\n"
            "    assert acronym('Ruby on Rails') == 'ROR'\n\n"
            "def test_hyphenated_words_count_separately():\n"
            "    assert acronym('Complementary metal-oxide semiconductor')"
            " == 'CMOS'\n\n"
            "def test_truncated_to_maxlen():\n"
            "    long_acronym = 'ABCDEFG'\n"
            "    got = acronym("
            "'Alpha Bravo Charlie Delta Echo Foxtrot Golf')\n"
            "    assert got == long_acronym[:MAXLEN]\n"
            "    assert len(got) <= MAXLEN\n"
        ),
        "solution_src": (
            "import re\n\n"
            "def acronym(title):\n"
            "    words = re.split(r'[ -]+', title.strip())\n"
            "    result = ''.join(w[0].upper() for w in words if w)\n"
            "    return result[:@@MAXLEN@@]\n"
        ),
    },

    "word_count": {
        "module": "word_count.py",
        "token": "@@MINLEN@@",
        "param": lambda h: 3 + h % 2,  # 3..4
        "intro": (
            "Exercise: Word Frequency Counter. Implement word_count(text), "
            "which returns a dict mapping each distinct word in `text` to "
            "how many times it appears. Words are matched case-insensitively "
            "(so 'Hello' and 'hello' are the same word and the returned key "
            "should be lower-case), and punctuation attached to a word "
            "should be stripped before counting, so 'dog,' and 'dog.' both "
            "count as 'dog'. This exercise mirrors a real internal tool "
            "that powers a simple content-tagging pipeline, and that "
            "pipeline has ONE distinctive constraint baked into it from day "
            "one: only words with length GREATER THAN OR EQUAL TO "
            "@@MINLEN@@ characters are counted at all; shorter words (like "
            "stopwords 'a', 'on', 'to') are silently dropped from the "
            "result entirely and must not appear as keys, even with a count "
            "of zero. Keep this minimum-length filter in mind; it is the "
            "one number a reviewer will ask you to justify later. "
            "Implement word_count.py so its tests pass."
        ),
        "stub": (
            '"""{intro}"""\n\n'
            "\n"
            "def word_count(text):\n"
            "    # TODO: implement per the module docstring above.\n"
            "    raise NotImplementedError\n"
        ),
        "test_src": (
            "from word_count import word_count\n\n"
            "MINLEN = @@MINLEN@@\n\n"
            "def test_word_below_threshold_excluded():\n"
            "    word = 'x' * (MINLEN - 1)\n"
            "    result = word_count(f'{word} {word} keep')\n"
            "    assert word not in result\n\n"
            "def test_word_at_threshold_included():\n"
            "    word = 'y' * MINLEN\n"
            "    result = word_count(f'{word} {word} extra')\n"
            "    assert result[word] == 2\n\n"
            "def test_counts_case_insensitive():\n"
            "    result = word_count('Hello hello HELLO')\n"
            "    assert result['hello'] == 3\n\n"
            "def test_ignores_punctuation():\n"
            "    result = word_count('doggo, doggo. doggo!')\n"
            "    assert result['doggo'] == 3\n"
        ),
        "solution_src": (
            "import re\n"
            "from collections import Counter\n\n"
            "def word_count(text):\n"
            "    words = re.findall(r'[a-zA-Z]+', text.lower())\n"
            "    counts = Counter(w for w in words if len(w) >= @@MINLEN@@)\n"
            "    return dict(counts)\n"
        ),
    },

    "rotate_list": {
        "module": "rotate_list.py",
        "token": "@@ROTN@@",
        "param": lambda h: 1 + h % 4,  # 1..4
        "intro": (
            "Exercise: Fixed Rotation. Implement rotate_left(items), which "
            "returns a NEW list containing the elements of `items` rotated "
            "to the left, without mutating the input list. Rotating left "
            "means the first elements move to the end: rotating [1,2,3,4] "
            "left by 1 gives [2,3,4,1]. Unlike the typical generic version "
            "of this exercise found in most tutorials, this particular "
            "module does not take the rotation amount as a parameter at "
            "all — it is a FIXED project-wide constant. The one distinctive "
            "number to remember here is that this codebase always rotates "
            "left by EXACTLY @@ROTN@@ positions, no more and no less, "
            "because every caller in this service was written against that "
            "fixed offset and changing it would silently break unrelated "
            "code elsewhere. Handle the empty list gracefully (return an "
            "empty list), and make sure you build and return a new list "
            "rather than modifying the caller's list in place. Implement "
            "rotate_list.py so its tests pass."
        ),
        "stub": (
            '"""{intro}"""\n\n'
            "\n"
            "def rotate_left(items):\n"
            "    # TODO: implement per the module docstring above.\n"
            "    raise NotImplementedError\n"
        ),
        "test_src": (
            "from rotate_list import rotate_left\n\n"
            "N = @@ROTN@@\n\n"
            "def test_rotates_by_fixed_amount():\n"
            "    items = [1, 2, 3, 4, 5, 6]\n"
            "    expected = items[N:] + items[:N]\n"
            "    assert rotate_left(items) == expected\n\n"
            "def test_empty_list():\n"
            "    assert rotate_left([]) == []\n\n"
            "def test_does_not_mutate_input():\n"
            "    items = [1, 2, 3, 4, 5, 6]\n"
            "    original = list(items)\n"
            "    rotate_left(items)\n"
            "    assert items == original\n\n"
            "def test_strings():\n"
            "    items = ['a', 'b', 'c', 'd', 'e', 'f']\n"
            "    expected = items[N:] + items[:N]\n"
            "    assert rotate_left(items) == expected\n"
        ),
        "solution_src": (
            "def rotate_left(items):\n"
            "    if not items:\n"
            "        return []\n"
            "    n = @@ROTN@@ % len(items)\n"
            "    return items[n:] + items[:n]\n"
        ),
    },

    "fahrenheit": {
        "module": "fahrenheit.py",
        "token": "@@DEC@@",
        "param": lambda h: 1 + h % 3,  # 1..3
        "intro": (
            "Exercise: Temperature Conversion. Implement "
            "celsius_to_fahrenheit(c), converting a Celsius temperature to "
            "Fahrenheit using the standard formula F = C * 9/5 + 32, and "
            "return a float. This looks trivial, but the weather-dashboard "
            "service that owns this module has ONE distinctive formatting "
            "constraint that the tests enforce strictly: every returned "
            "value must be ROUNDED to exactly @@DEC@@ decimal place(s) "
            "using Python's built-in round(), because the dashboard's "
            "front-end renders temperatures with a fixed number of decimal "
            "digits and any drift causes a visual layout bug the design "
            "team has complained about before. Do not round earlier "
            "intermediate values, only the final result, and do not forget "
            "negative temperatures (like -40, where Celsius and Fahrenheit "
            "coincide) or the two textbook fixed points, freezing (0C = "
            "32F) and boiling (100C = 212F). Implement fahrenheit.py so its "
            "tests pass."
        ),
        "stub": (
            '"""{intro}"""\n\n'
            "\n"
            "def celsius_to_fahrenheit(c):\n"
            "    # TODO: implement per the module docstring above.\n"
            "    raise NotImplementedError\n"
        ),
        "test_src": (
            "from fahrenheit import celsius_to_fahrenheit\n\n"
            "DEC = @@DEC@@\n\n"
            "def test_freezing():\n"
            "    assert celsius_to_fahrenheit(0) == round(32.0, DEC)\n\n"
            "def test_boiling():\n"
            "    assert celsius_to_fahrenheit(100) == round(212.0, DEC)\n\n"
            "def test_body_temp_rounding():\n"
            "    assert celsius_to_fahrenheit(37) == "
            "round(37 * 9 / 5 + 32, DEC)\n\n"
            "def test_negative():\n"
            "    assert celsius_to_fahrenheit(-40) == round(-40.0, DEC)\n\n"
            "def test_returns_float():\n"
            "    assert isinstance(celsius_to_fahrenheit(21), float)\n"
        ),
        "solution_src": (
            "def celsius_to_fahrenheit(c):\n"
            "    return round(c * 9 / 5 + 32, @@DEC@@)\n"
        ),
    },

    "palindrome": {
        "module": "palindrome.py",
        "token": "@@MINLEN2@@",
        "param": lambda h: 2 + h % 2,  # 2..3
        "intro": (
            "Exercise: Palindrome Class. Implement a class Palindrome whose "
            "constructor takes a text string, and whose method "
            "is_palindrome() returns True if the text reads the same "
            "forwards and backwards once you ignore case and strip out "
            "every character that is not a letter or digit (so punctuation "
            "and spaces don't count, meaning 'A man, a plan, a canal: "
            "Panama' is a palindrome). This class backs a puzzle-game "
            "feature, and the game designer added ONE distinctive rule "
            "that trips people up: after cleaning the text, if the "
            "resulting string has fewer than @@MINLEN2@@ characters, "
            "is_palindrome() must return False unconditionally, no matter "
            "what the cleaned text looks like — very short strings are "
            "considered too trivial to count as a real palindrome in this "
            "game and must never award points. Only strings at or above "
            "that minimum cleaned length should be checked for the actual "
            "palindrome property. Implement palindrome.py so its tests "
            "pass."
        ),
        "stub": (
            '"""{intro}"""\n\n'
            "\n"
            "class Palindrome:\n"
            "    def __init__(self, text):\n"
            "        # TODO: implement per the module docstring above.\n"
            "        raise NotImplementedError\n\n"
            "    def is_palindrome(self):\n"
            "        raise NotImplementedError\n"
        ),
        "test_src": (
            "from palindrome import Palindrome\n\n"
            "MINLEN = @@MINLEN2@@\n\n"
            "def test_simple_palindrome():\n"
            "    assert Palindrome('racecar').is_palindrome() is True\n\n"
            "def test_ignores_case_and_punctuation():\n"
            "    text = 'A man, a plan, a canal: Panama'\n"
            "    assert Palindrome(text).is_palindrome() is True\n\n"
            "def test_not_palindrome():\n"
            "    assert Palindrome('hello').is_palindrome() is False\n\n"
            "def test_below_min_length_is_false():\n"
            "    short = 'a' * (MINLEN - 1)\n"
            "    assert Palindrome(short).is_palindrome() is False\n\n"
            "def test_at_min_length_palindrome_true():\n"
            "    text = 'b' * MINLEN\n"
            "    assert Palindrome(text).is_palindrome() is True\n"
        ),
        "solution_src": (
            "import re\n\n"
            "class Palindrome:\n"
            "    def __init__(self, text):\n"
            "        self.text = text\n\n"
            "    def _clean(self):\n"
            "        return re.sub(r'[^a-z0-9]', '', self.text.lower())\n\n"
            "    def is_palindrome(self):\n"
            "        cleaned = self._clean()\n"
            "        if len(cleaned) < @@MINLEN2@@:\n"
            "            return False\n"
            "        return cleaned == cleaned[::-1]\n"
        ),
    },

    "csv_line": {
        "module": "csv_line.py",
        "token": "@@DELIM@@",
        "param": lambda h: [";", "|", ":"][h % 3],
        "intro": (
            "Exercise: Single-Line Field Parser. Implement parse_line(line), "
            "which splits one line of delimiter-separated text into a list "
            "of fields, stripping leading and trailing whitespace from each "
            "field. This is deliberately simpler than a full CSV parser: "
            "you don't need to handle quoted fields or embedded delimiters, "
            "just plain splitting-and-trimming. The ONE distinctive fact "
            "about this format that you must remember is that the "
            "delimiter is NOT a comma — this legacy export format from an "
            "old inventory system uses the single character '@@DELIM@@' as "
            "its field delimiter, chosen decades ago specifically to avoid "
            "clashing with commas that commonly appear inside product "
            "descriptions. An empty field between two consecutive "
            "delimiters must be preserved as an empty string in the output "
            "list, not dropped. Implement csv_line.py so its tests pass."
        ),
        "stub": (
            '"""{intro}"""\n\n'
            "\n"
            "def parse_line(line):\n"
            "    # TODO: implement per the module docstring above.\n"
            "    raise NotImplementedError\n"
        ),
        "test_src": (
            "from csv_line import parse_line\n\n"
            "def test_splits_on_delimiter():\n"
            "    assert parse_line('a@@DELIM@@b@@DELIM@@c') == "
            "['a', 'b', 'c']\n\n"
            "def test_strips_whitespace_around_fields():\n"
            "    assert parse_line(' a @@DELIM@@ b @@DELIM@@ c ') == "
            "['a', 'b', 'c']\n\n"
            "def test_single_field():\n"
            "    assert parse_line('solo') == ['solo']\n\n"
            "def test_empty_field_preserved():\n"
            "    assert parse_line('a@@DELIM@@@@DELIM@@c') == "
            "['a', '', 'c']\n"
        ),
        "solution_src": (
            "def parse_line(line):\n"
            "    return [field.strip() for field in "
            "line.split('@@DELIM@@')]\n"
        ),
    },

    "leaderboard": {
        "module": "leaderboard.py",
        "token": "@@TOPN@@",
        "param": lambda h: 2 + h % 2,  # 2..3
        "intro": (
            "Exercise: Leaderboard Top-N. Implement top_scores(scores), "
            "where `scores` is a dict mapping player name to integer "
            "score, and the function returns a list of player names sorted "
            "by score descending; ties are broken alphabetically by name. "
            "This powers the results screen of a small arcade game, and "
            "the ONE distinctive constraint the design team insisted on is "
            "that the returned list must contain AT MOST the top "
            "@@TOPN@@ names — the screen only has room to display "
            "@@TOPN@@ rows, so anything beyond that must simply be "
            "omitted, never truncated with an ellipsis or error. If fewer "
            "than @@TOPN@@ players exist, return all of them in sorted "
            "order (no padding). Implement leaderboard.py so its tests "
            "pass."
        ),
        "stub": (
            '"""{intro}"""\n\n'
            "\n"
            "def top_scores(scores):\n"
            "    # TODO: implement per the module docstring above.\n"
            "    raise NotImplementedError\n"
        ),
        "test_src": (
            "from leaderboard import top_scores\n\n"
            "TOPN = @@TOPN@@\n\n"
            "def test_returns_top_n_sorted_desc():\n"
            "    scores = {'alice': 50, 'bob': 90, 'carol': 70, "
            "'dave': 20, 'erin': 60}\n"
            "    result = top_scores(scores)\n"
            "    assert len(result) == TOPN\n"
            "    expected = sorted(scores, key=lambda k: (-scores[k], k))"
            "[:TOPN]\n"
            "    assert result == expected\n\n"
            "def test_ties_broken_alphabetically():\n"
            "    scores = {'zed': 100, 'amy': 100, 'bob': 5}\n"
            "    result = top_scores(scores)\n"
            "    assert result[0] == 'amy'\n"
            "    assert result[1] == 'zed'\n\n"
            "def test_fewer_entries_than_n():\n"
            "    scores = {'solo': 10}\n"
            "    assert top_scores(scores) == ['solo']\n\n"
            "def test_returns_list_of_names_only():\n"
            "    scores = {'a': 1, 'b': 2, 'c': 3}\n"
            "    result = top_scores(scores)\n"
            "    assert all(isinstance(x, str) for x in result)\n"
        ),
        "solution_src": (
            "def top_scores(scores):\n"
            "    ranked = sorted(scores, key=lambda k: (-scores[k], k))\n"
            "    return ranked[:@@TOPN@@]\n"
        ),
    },

    "run_length": {
        "module": "run_length.py",
        "token": "@@THRESH@@",
        "param": lambda h: 2 + h % 2,  # 2..3
        "intro": (
            "Exercise: Threshold Run-Length Encoding. Implement "
            "encode(text), a variant of run-length encoding where a run of "
            "identical consecutive characters is written as the character "
            "followed by its count, e.g. 'a5' for five a's in a row. The "
            "ONE distinctive constraint that makes this variant different "
            "from the textbook version is that only runs of length "
            "GREATER THAN OR EQUAL TO @@THRESH@@ get compressed; runs "
            "shorter than @@THRESH@@ characters must be left exactly as "
            "literal repeated characters (uncompressed) in the output, "
            "because the downstream decoder used by this project cannot "
            "tell a compressed 'a2' apart from a literal two-character run "
            "if the run were always compressed, and @@THRESH@@ was chosen "
            "as the safe cutoff after a decoding ambiguity bug. The empty "
            "string encodes to the empty string. Implement run_length.py "
            "so its tests pass."
        ),
        "stub": (
            '"""{intro}"""\n\n'
            "\n"
            "def encode(text):\n"
            "    # TODO: implement per the module docstring above.\n"
            "    raise NotImplementedError\n"
        ),
        "test_src": (
            "from run_length import encode\n\n"
            "THRESH = @@THRESH@@\n\n"
            "def test_long_run_compressed():\n"
            "    run = 'a' * (THRESH + 2)\n"
            "    assert encode(run) == f'a{THRESH + 2}'\n\n"
            "def test_short_run_not_compressed():\n"
            "    run = 'b' * (THRESH - 1)\n"
            "    assert encode(run) == run\n\n"
            "def test_mixed_runs():\n"
            "    text = ('c' * (THRESH + 3)) + 'd'\n"
            "    assert encode(text) == f'c{THRESH + 3}d'\n\n"
            "def test_empty_string():\n"
            "    assert encode('') == ''\n\n"
            "def test_single_char_never_compressed():\n"
            "    assert encode('z') == 'z'\n"
        ),
        "solution_src": (
            "def encode(text):\n"
            "    result = []\n"
            "    i = 0\n"
            "    n = len(text)\n"
            "    while i < n:\n"
            "        j = i\n"
            "        while j < n and text[j] == text[i]:\n"
            "            j += 1\n"
            "        run_len = j - i\n"
            "        if run_len >= @@THRESH@@:\n"
            "            result.append(f'{text[i]}{run_len}')\n"
            "        else:\n"
            "            result.append(text[i] * run_len)\n"
            "        i = j\n"
            "    return ''.join(result)\n"
        ),
    },
}

EXERCISE_IDS = sorted(EXERCISES.keys())  # stable base order before shuffling


def _plan(seed):
    """Deterministically pick 4 of the 8 exercises and order them, and
    compute each chosen exercise's seeded constraint value."""
    ordered_ids = sorted(EXERCISE_IDS, key=lambda eid: _h(seed, f"select:{eid}"))
    chosen = ordered_ids[:4]
    plan = []
    for pos, eid in enumerate(chosen, start=1):
        spec = EXERCISES[eid]
        h = _h(seed, f"param:{eid}")
        value = spec["param"](h)
        plan.append({"pos": pos, "id": eid, "value": value})
    return plan


def _render(eid, value):
    """Render (intro, stub_src, test_src, solution_src) for one exercise
    with its token substituted throughout."""
    spec = EXERCISES[eid]
    mapping = {spec["token"]: value}
    intro = _substitute(spec["intro"], mapping)
    stub = spec["stub"].format(intro=intro)
    stub = _substitute(stub, mapping)
    test_src = _substitute(spec["test_src"], mapping)
    solution_src = _substitute(spec["solution_src"], mapping)
    return intro, stub, test_src, solution_src


def _task_seed(task):
    assert task.startswith("chain:"), f"unknown task id: {task!r}"
    return task.split(":", 1)[1]


def materialize(task, d):
    seed = _task_seed(task)
    plan = _plan(seed)
    root = Path(d)
    root.mkdir(parents=True, exist_ok=True)
    chain_manifest = {"task": task, "seed": seed, "exercises": []}
    for entry in plan:
        eid, pos, value = entry["id"], entry["pos"], entry["value"]
        spec = EXERCISES[eid]
        _intro, stub_src, test_src, _sol = _render(eid, value)
        exdir = root / f"ex{pos}"
        exdir.mkdir(parents=True, exist_ok=True)
        (exdir / spec["module"]).write_text(stub_src)
        (exdir / f"test_{spec['module']}").write_text(test_src)
        (exdir / "pytest.ini").write_text(PYTEST_INI)
        chain_manifest["exercises"].append({
            "pos": pos, "id": eid, "module": spec["module"],
            "token": spec["token"], "value": value,
        })
    (root / "chain.json").write_text(json.dumps(chain_manifest, indent=1))


def prompt(task):
    seed = _task_seed(task)
    plan = _plan(seed)
    parts = [
        "You are working through a CHAINED session of 4 small, "
        "self-contained Python exercises, laid out in subdirectories "
        "ex1/, ex2/, ex3/, and ex4/ of the current directory. Work through "
        "them IN ORDER, ex1 first, then ex2, ex3, ex4 — do not skip ahead. "
        "For EACH exercise: read its module's docstring (it states an "
        "intro plus one distinctive numeric/format constraint you must "
        "honor), implement the stub function(s)/class in that module, and "
        "confirm `python -m pytest exN/ -q` passes for that exercise BEFORE "
        "moving on to the next one. Do not modify any test file. After "
        "you finish ex4, as your final action, re-state FROM MEMORY "
        "(without re-reading ex1's file) the exact distinctive constraint "
        "from ex1 — including its number — to confirm you still retain it."
    ]
    for entry in plan:
        eid, pos, value = entry["id"], entry["pos"], entry["value"]
        intro, _stub, _test, _sol = _render(eid, value)
        parts.append(f"\n--- ex{pos}/ ({EXERCISES[eid]['module']}) ---\n{intro}")
    return "\n".join(parts)


def score(task, d, log):
    import subprocess
    root = Path(d)
    manifest_path = root / "chain.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        positions = [e["pos"] for e in manifest["exercises"]]
    else:
        positions = [1, 2, 3, 4]
    per_ex = {}
    for pos in sorted(positions):
        exdir = root / f"ex{pos}"
        r = subprocess.run(
            [sys.executable, "-m", "pytest", f"ex{pos}/", "-q",
             "--no-header"],
            cwd=root, capture_output=True, text=True, timeout=120,
        )
        per_ex[f"ex{pos}"] = r.returncode == 0
    tests_pass = all(per_ex.values()) and len(per_ex) > 0
    return {
        "task": task,
        "tests_pass": tests_pass,
        "per_ex": per_ex,
        "chain_len": len(per_ex),
    }


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "materialize":
        materialize(sys.argv[2], sys.argv[3])
    elif cmd == "prompt":
        print(prompt(sys.argv[2]))
    elif cmd == "score":
        print(json.dumps(score(sys.argv[2], sys.argv[3], sys.argv[4]), indent=1))
    else:
        raise SystemExit(f"unknown command: {cmd}")
