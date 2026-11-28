# Curriculum seed prompt — paste into ChatGPT (or Claude), one conversation, daily drip

Companion to `intake.py`: save each reply as `batchN.json`, validate, promote.

---

## First message (starts the curriculum conversation)

```
You are co-authoring problems for an execution-verified Python coding benchmark.
Your problems will be run by a harness: your `ref` solution is executed, then your
`tests` asserts are executed against it. A problem is accepted only if the asserts
pass against the ref — so wrong expected values are fatal, and you must verify every
assert by careful mental execution before returning it.

RULES
1. Each problem asks the solver to implement exactly ONE function (or class) in
   Python 3, standard library only.
2. Fully deterministic: no randomness, time, filesystem, network, locale, or
   version-specific behavior. Small inputs only (n <= 1000).
3. `tests` is a string of `assert` statements (helper `def`s allowed). 4-7 asserts:
   first is the canonical example, the rest probe edges (empty input, ties,
   boundaries, off-by-one). Edges must actually distinguish a correct solution from
   plausible near-misses.
4. Non-unique answers (any valid order, any valid window, ...) must be normalized in
   the tests (sorted) or verified by a checker function — never force one arbitrary
   answer.
5. Exact expected values only. Hand-verify every number. No float equality unless
   exactly representable; prefer ints; if unavoidable use round(x, 6) == ...
6. The prompt must spell out every tie-break and edge semantic ("on ties the smallest
   index wins", "empty input returns 0", what errors are raised on invalid input).
7. Difficulty comes from edge-case discipline, not obscurity. Solvable in <= 40 lines.
8. TIER 2 = classic algorithm problem with at least one sharp edge. TIER 3 = harder:
   multi-stage state (parser with escapes, DP that must reconstruct, heap/stream
   invariant, graph property checked by a checker function) — still <= 40 lines,
   still pure-assert testable.

OUTPUT FORMAT
Return ONLY a JSON array (no markdown fences, no commentary). Each element:
{"id": "unique_snake_case", "tier": 2, "tag": "dp|graphs|parsing|strings|algorithms|data-structures|matrix|backtracking|math|lists|dicts", "title": "...", "prompt": "...", "tests": "assert ...\nassert ...", "ref": "def ...: ..."}
CRITICAL: `ref` and `tests` must carry real Python indentation INSIDE the JSON
strings — after every \n write the exact literal spaces (4 per nesting level),
e.g. "def f(xs):\n    total = 0\n    for x in xs:\n        if x > 0:\n            total += x\n    return total"
A ref whose lines all start with a single space is silently broken and will be
rejected by the harness without review.
FLATTEN-SAFE: transport sometimes strips leading spaces, so keep indented lines
to a minimum — prefer one-line compounds (`if not nums: return 0`), one-line
try/except pairs (`try: r = f([[3, 2]])` then a new line `except ValueError: r = "err"`
at column 0, then `assert r == "err"`), and comprehensions over append-loops.
Where a block body is unavoidable, indent with exactly 4 spaces per level.

EXAMPLES (note how eval_expr pins division semantics, and course_order uses a checker
because the answer is non-unique):
{
 "id": "eval_expr",
 "tier": 2,
 "tag": "parsing",
 "title": "Expression evaluator",
 "prompt": "Implement `eval_expr(s)` evaluating an arithmetic expression of\nnon-negative integers with `+ - * /` and parentheses, standard precedence,\noptional spaces. Division truncates toward zero and applies to its operand\nsubexpression results. No unary minus appears in input.",
 "tests": "assert eval_expr(\"3+2*2\") == 7\nassert eval_expr(\"3/2\") == 1\nassert eval_expr(\"(1+(4+5+2)-3)\") == 9\nassert eval_expr(\"2*(5+5*2)/3+(6/2+8)\") == 21\nassert eval_expr(\"14-3*2\") == 8\nassert eval_expr(\"10/3*3\") == 9",
 "ref": "def eval_expr(s):\n    s = s.replace(\" \", \"\")\n    pos = 0\n\n    def expr():\n        nonlocal pos\n        v = term()\n        while pos < len(s) and s[pos] in \"+-\":\n            op = s[pos]\n            pos += 1\n            r = term()\n            v = v + r if op == \"+\" else v - r\n        return v\n\n    def term():\n        nonlocal pos\n        v = atom()\n        while pos < len(s) and s[pos] in \"*/\":\n            op = s[pos]\n            pos += 1\n            r = atom()\n            if op == \"*\":\n                v *= r\n            else:\n                q = abs(v) // abs(r)\n                v = q if (v >= 0) == (r >= 0) else -q\n        return v\n\n    def atom():\n        nonlocal pos\n        if s[pos] == \"(\":\n            pos += 1\n            v = expr()\n            pos += 1\n            return v\n        j = pos\n        while j < len(s) and s[j].isdigit():\n            j += 1\n        v = int(s[pos:j])\n        pos = j\n        return v\n\n    return expr()"
}
{
 "id": "course_order",
 "tier": 2,
 "tag": "graphs",
 "title": "Course ordering",
 "prompt": "Implement `course_order(n, prereqs)` where prereqs is a list of\n`[course, prerequisite]` pairs; return ANY valid study order as a list of all\nn courses, or [] if impossible. There may be many valid orders.",
 "tests": "def check(order, n, pairs):\n    assert isinstance(order, list) and sorted(order) == list(range(n))\n    pos = {c: i for i, c in enumerate(order)}\n    for c, p in pairs:\n        assert pos[p] < pos[c]\n\no = course_order(4, [[1, 0], [2, 0], [3, 1], [3, 2]])\ncheck(o, 4, [[1, 0], [2, 0], [3, 1], [3, 2]])\nassert course_order(2, [[1, 0], [0, 1]]) == []\nassert course_order(1, []) == [0]\no2 = course_order(3, [[2, 1]])\ncheck(o2, 3, [[2, 1]])",
 "ref": "def course_order(n, prereqs):\n    from collections import deque\n    adj, indeg = [[] for _ in range(n)], [0] * n\n    for c, p in prereqs:\n        adj[p].append(c)\n        indeg[c] += 1\n    q = deque(i for i in range(n) if indeg[i] == 0)\n    order = []\n    while q:\n        u = q.popleft()\n        order.append(u)\n        for v in adj[u]:\n            indeg[v] -= 1\n            if indeg[v] == 0:\n                q.append(v)\n    return order if len(order) == n else []"
}

AVOID — these ids already exist, no duplicates or near-duplicates:
anagrams, assign_rooms, balanced, balanced_code, base_neg2_conversion, bipartition_coloring, canonical_fraction, case_convert, closest_sum_pair, coin_change, combo_sum_once, count_coprime_inclusion_exclusion, count_digit_occurrences, count_divisible_subsets, count_domino_tilings, count_islands, count_n_queens, course_order, csv_line, daily_temperatures, deadline_dispatch, decode_escaped_fields, decode_nested_repeat, deduplicate_stable, directed_euler_walk, edit_distance, egyptian_greedy, eulerian_trail, eval_expr, excel_column_title, expand_counted_pattern, flatten, fraction_add, glob_full_match, grid_wall_escape, histogram, interval_overlap_count, knapsack_reconstruct, kth_largest_stream, largest_ones_square, lcs_string, lis_reconstruct, longest_bounded_window, longest_consec, longest_k_distinct, longest_unique_window, lru, matrix_chain_multiplication, matrix_diagonal_sums, max_product_after_cut, max_product_slice, merge_intervals, merge_sorted_streams, min_assignment_cost, min_eating_speed, min_join_cost, min_path_effort_grid, min_ragged_wrap, min_stack, min_station_refuels, min_window, minimize_largest_block_sum, next_greater_steps, next_perm, normalize_identifier, normalize_path, palindrome_sub, parse_duration, parse_kv_escaped, predict_the_winner, render_escaped_template, repeating_decimal, restore_ips, retry_decorator, rle, roman, rotate_matrix, rotated_search, running_median, set_zeroes, shortest_cycle_length, shortest_path_with_count, shortest_threshold_slice, solve_linear_congruence, spiral_order, stable_frequency_sort, streaming_kth_largest, subarrays_sum_equals_k, summary_ranges, top_k_words, trap_rain_water, tsp_exact_bitmask, weighted_interval_schedule, word_break, word_pattern_bijection.

TASK: 10 problems as one JSON array. At least 6 distinct tags. 6 tier-2, 4 tier-3.
Before returning, mentally execute EVERY assert against its ref; drop any pair you
are not 100% certain passes and replace it until all 10 are certain.
```

## Teacher-specific task lines

Keep everything above byte-identical across teachers (the contract is what
makes first-pass validation rates comparable — that IS the teacher ranking).
Swap only the TASK paragraph at the end:

- **Claude** (best first-pass correctness): default task line — broad coverage,
  6 tier-2 + 4 tier-3. Spend its scarce flagship messages on tier-3 authoring.
- **ChatGPT** (good designs, sloppy expected values): default task line; its
  broken asserts are usually provable arithmetic — hand-fix or drop, don't correct.
- **Kimi K3.x** (competition/math strength): replace the TASK paragraph with:

```
TASK: 10 problems as one JSON array. At least 6 distinct tags. 3 tier-2, 7 tier-3.
Lean into mathematically flavored algorithms with EXACT INTEGER answers:
modular arithmetic and counting, greedy algorithms with a provable exchange
argument, binary search over the answer with a monotone predicate, small-n
bitmask DP, game/turn-based invariants, positional/base arithmetic, shortest
paths on small explicit grids. Integer expected values make every assert
hand-verifiable and dodge tie-break ambiguity entirely. Write all output in
English. Before returning, mentally execute EVERY assert against its ref;
drop any pair you are not 100% certain passes and replace it until all 10
are certain.
```

Integer-answer problems are deliberately requested: they eliminate the
float/tie-break bug classes that cost the most curation time in batches 1-2.

## Correction message (exception path — forward-only is the default)

The seed message costs the same paste-effort and returns ~10 fresh problems;
a correction round recovers 1-3, and intake's near-dup filter rejects many of
them. With 3 teachers the suite grows faster than the harvest loop can absorb,
so **skip corrections by default**: broken items are hand-fixed only when the
expected value is provably wrong, otherwise dropped. Tomorrow's batch replaces
them.

Spend a correction message only when the BROKEN block contains a problem whose
**mechanism no current suite problem covers** — one message per batch max, and
never a second round in the same chat. When you do, run
`python coding_eval/intake.py batchN.json` and paste:

```
These problems you authored failed execution verification. For each, decide whether
the assert or the ref is wrong, fix it, and return ONLY the corrected elements in the
same JSON array format:

<paste the BROKEN block printed by intake.py>
```

## Local loop

```
python coding_eval/intake.py /tmp/batch1.json            # validate (no writes)
python coding_eval/intake.py /tmp/batch1.json --promote  # append survivors to
                                                         # problems.py + references.py
git -C /Volumes/NVME/ling-3.2-air diff coding_eval/problems.py   # eyeball before commit
./run3.sh                                                # re-eval all packs on the bigger suite
```

Notes
- Free-tier flagship message budget: spend it ALL on the FIRST message's
  10-problem batches (forward-only); corrections are the rare exception above.
  The mini fallback is fine for retagging or reformatting but writes too many
  broken asserts for authoring.
- Claude free tier can run the same first message in parallel for teacher diversity;
  intake dedupes by id and the harness decides the rest.
- After promoting, ids in the AVOID list auto-update: intake.py reads them from
  problems.py, but re-paste the new list into the teacher chat weekly.
