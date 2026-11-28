"""Original coding problems with executable asserts (stdlib-only, deterministic).

Each problem: id, title, tag, prompt (what the model sees), tests (executed
against the model's solution module `sol`).
"""

# Held-out test split — never in LoRA training data; run_eval --harvest
# excludes these by default. Carved 2026-11-27 before distillation began:
# tier mix + the shared failure ceiling (parse_kv_escaped, lis_reconstruct).
# The scoreboard's honest generalization number = held-out subset only.
# (Kept above the list: intake.py --promote requires PROBLEMS at EOF.)
HOLDOUT_IDS = {
    "eval_expr", "running_median", "histogram", "word_break",
    "min_window", "egyptian_greedy", "combo_sum_once", "kth_largest_stream",
    "canonical_fraction", "parse_kv_escaped", "lis_reconstruct",
    "shortest_cycle_length",
}

PROBLEMS = [
    dict(
        id="merge_intervals", tag="lists",
        title="Merge intervals",
        prompt="""Implement `merge_intervals(intervals)` that takes a list of `[start, end]`
pairs and returns a new list of non-overlapping `[start, end]` pairs covering
the same points, sorted by start. Touching intervals (end == next start) merge.""",
        tests="""
assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
assert merge_intervals([]) == []
assert merge_intervals([[5,5]]) == [[5,5]]
assert merge_intervals([[1,4],[4,5]]) == [[1,5]]
assert merge_intervals([[1,10],[2,3],[4,5]]) == [[1,10]]
""",
    ),
    dict(
        id="top_k_words", tag="dicts",
        title="Top-k frequent words",
        prompt="""Implement `top_k_words(words, k)` returning the k most frequent words,
ordered by frequency descending then alphabetically ascending for ties.""",
        tests="""
assert top_k_words(["i","it","is","it","i"], 2) == ["i","it"]
assert top_k_words(["b","a","b","c","c"], 2) == ["b","c"]
assert top_k_words(["x"], 1) == ["x"]
assert top_k_words(["a","a","b","b","c"], 3) == ["a","b","c"]
""",
    ),
    dict(
        id="balanced", tag="strings",
        title="Balanced brackets",
        prompt="""Implement `is_balanced(s)` for a string containing `()[]{}` and other
characters; return True iff every bracket is closed by the right type in order.""",
        tests="""
assert is_balanced("([{}])") is True
assert is_balanced("(]") is False
assert is_balanced("func(a[0], {b: 1}) ok") is True
assert is_balanced("((())") is False
assert is_balanced("") is True
""",
    ),
    dict(
        id="rle", tag="strings",
        title="Run-length encode/decode",
        prompt="""Implement `rle_encode(s)` mapping "aaabcc" to "a3b1c2" (char followed by its
run length), and `rle_decode(t)` inverting it. Input chars are lowercase letters.""",
        tests="""
assert rle_encode("aaabcc") == "a3b1c2"
assert rle_encode("") == ""
assert rle_decode("a3b1c2") == "aaabcc"
assert rle_decode(rle_encode("zzzaaabbczz")) == "zzzaaabbczz"
assert rle_encode("abc") == "a1b1c1"
""",
    ),
    dict(
        id="flatten", tag="dicts",
        title="Flatten nested dict",
        prompt="""Implement `flatten(d)` turning nested dicts/lists into a single-level dict
with dot-joined keys; list indices appear as segments, e.g. {"a": {"b": [7]}}
becomes {"a.b.0": 7}. Empty containers contribute no keys.""",
        tests="""
assert flatten({"a": {"b": 1}}) == {"a.b": 1}
assert flatten({"a": {"b": [7, 8]}}) == {"a.b.0": 7, "a.b.1": 8}
assert flatten({}) == {}
assert flatten({"x": {"y": {"z": 3}}, "w": 1}) == {"x.y.z": 3, "w": 1}
assert flatten({"a": []}) == {}
""",
    ),
    dict(
        id="closest_sum_pair", tag="algorithms",
        title="Closest two-sum pair",
        prompt="""Implement `closest_pair(nums, target)` returning the indices (i, j), i < j,
of the two elements whose sum is closest to target; on equal closeness return
the pair found first in index order (smallest i, then smallest j).""",
        tests="""
assert closest_pair([2, 5, 1, 8], 10) == (0, 3)
assert closest_pair([1, 2, 3], 100) == (1, 2)
assert closest_pair([4, 4], 8) == (0, 1)
assert closest_pair([1, 5, 9], 6) == (0, 1)
""",
    ),
    dict(
        id="roman", tag="strings",
        title="Roman to integer",
        prompt="""Implement `roman_to_int(s)` for standard Roman numerals up to thousands
(I=1, V=5, X=10, L=50, C=100, D=500, M=1000), subtractive notation included.""",
        tests="""
assert roman_to_int("III") == 3
assert roman_to_int("LVIII") == 58
assert roman_to_int("MCMXCIV") == 1994
assert roman_to_int("XL") == 40
assert roman_to_int("MMXXVI") == 2026
""",
    ),
    dict(
        id="lru", tag="data-structures",
        title="LRU cache",
        prompt="""Implement class `LRU` with `__init__(capacity)`, `get(key)` (returns value or
-1, marks most-recent) and `put(key, value)` (evicts least-recently-used when
over capacity). Both operations must be O(1) amortized.""",
        tests="""
c = LRU(2)
c.put("a", 1); c.put("b", 2)
assert c.get("a") == 1
c.put("c", 3)          # evicts b (a was touched after b)
assert c.get("b") == -1
assert c.get("c") == 3
c.put("a", 9)          # update existing
assert c.get("a") == 9 and c.get("c") == 3 and c.get("b") == -1
""",
    ),
    dict(
        id="csv_line", tag="parsing",
        title="CSV line parser",
        prompt="""Implement `parse_csv(line, delim=',')` splitting one CSV line into fields,
honoring double-quoted sections that may contain the delimiter; a doubled
quote `""` inside quotes is one literal quote. No newlines appear in input.""",
        tests="""
assert parse_csv('a,b,"c,d"') == ["a", "b", "c,d"]
assert parse_csv('x,"y""z",1') == ["x", 'y"z', "1"]
assert parse_csv("") == [""]
assert parse_csv('"only"') == ["only"]
assert parse_csv("a;b;c", delim=';') == ["a", "b", "c"]
""",
    ),
    dict(
        id="rotated_search", tag="algorithms",
        title="Search in rotated array",
        prompt="""Implement `search_rotated(nums, target)` returning the index of target in a
sorted-then-rotated array of distinct integers, or -1. Aim for O(log n).""",
        tests="""
assert search_rotated([4,5,6,7,0,1,2], 0) == 4
assert search_rotated([4,5,6,7,0,1,2], 3) == -1
assert search_rotated([1], 1) == 0
assert search_rotated([6,7,0,1,2,4,5], 4) == 5
assert search_rotated([1,2,3,4,5], 1) == 0
""",
    ),
    dict(
        id="rotate_matrix", tag="matrix",
        title="Rotate 90 degrees clockwise",
        prompt="""Implement `rotate(m)` returning a new n x n matrix rotated 90 degrees
clockwise (input may be any square matrix of ints).""",
        tests="""
assert rotate([[1,2],[3,4]]) == [[3,1],[4,2]]
assert rotate([[1,2,3],[4,5,6],[7,8,9]]) == [[7,4,1],[8,5,2],[9,6,3]]
assert rotate([[5]]) == [[5]]
""",
    ),
    dict(
        id="palindrome_sub", tag="strings",
        title="Longest palindromic substring",
        prompt="""Implement `longest_palindrome(s)` returning any longest contiguous
palindromic substring of s (length >= 1 for non-empty s; return "" for "").""",
        tests="""
assert len(longest_palindrome("babad")) == 3
assert longest_palindrome("cbbd") in ("bb",)
assert longest_palindrome("a") == "a"
assert longest_palindrome("") == ""
assert len(longest_palindrome("forgeeforg")) == 2
""",
    ),
    dict(
        id="retry_decorator", tag="python",
        title="Retry decorator",
        prompt="""Implement decorator `retry(attempts)` wrapping a function so it is called
at most `attempts` times total, returning the first success and re-raising the
last ValueError if all fail. Only ValueError is retried.""",
        tests="""
calls = {"n": 0}

@retry(3)
def flaky():
    calls["n"] += 1
    if calls["n"] < 3:
        raise ValueError("nope")
    return calls["n"]

assert flaky() == 3 and calls["n"] == 3

calls2 = {"n": 0}

@retry(2)
def always():
    calls2["n"] += 1
    raise ValueError("x")

try:
    always()
    assert False
except ValueError:
    assert calls2["n"] == 2

@retry(2)
def other():
    raise KeyError("k")

try:
    other()
    assert False
except KeyError:
    pass
""",
    ),
    dict(
        id="anagrams", tag="dicts",
        title="Group anagrams",
        prompt="""Implement `group_anagrams(words)` returning groups (lists) of anagrams;
groups may be in any order, as may words within a group.""",
        tests="""
res = group_anagrams(["eat","tea","tan","ate","nat","bat"])
norm = sorted(sorted(g) for g in res)
assert norm == [["ate","eat","tea"], ["bat"], ["nat","tan"]]
assert group_anagrams([""]) == [[""]]
""",
    ),
    dict(
        id="min_stack", tag="data-structures",
        title="Min stack",
        prompt="""Implement class `MinStack` with `push(x)`, `pop()`, `top()` and `get_min()`,
all O(1). pop/top/get_min are only called when non-empty.""",
        tests="""
s = MinStack()
s.push(-2); s.push(0); s.push(-3)
assert s.get_min() == -3
s.pop()
assert s.top() == 0 and s.get_min() == -2
s.push(-4)
assert s.get_min() == -4
""",
    ),
    dict(
        id="case_convert", tag="strings",
        title="snake_case <-> camelCase",
        prompt="""Implement `snake_to_camel(s)` (first word lowercase, later words
capitalized; e.g. "get_http_code" -> "getHttpCode") and `camel_to_snake(s)`
(the inverse; a run of capitals that precedes a lowercase word is one
acronym: "HTTPCode" -> "http_code"; single words stay lowercase).""",
        tests="""
assert snake_to_camel("get_http_code") == "getHttpCode"
assert snake_to_camel("one") == "one"
assert camel_to_snake("getHttpCode") == "get_http_code"
assert camel_to_snake("getHTTPCode") == "get_http_code"
assert camel_to_snake("one") == "one"
assert camel_to_snake("HTTPCode") == "http_code"
""",
    ),

    # ---- tier 2: harder edge-case discipline (added 2026-11-27) ----
    dict(
        id="next_perm", tier=2, tag="algorithms",
        title="Next permutation",
        prompt="""Implement `next_perm(nums)` returning the next lexicographic
permutation of the list (duplicates allowed) as a NEW list; if the input is
already the largest permutation, return the smallest (ascending).""",
        tests="""
assert next_perm([1, 2, 3]) == [1, 3, 2]
assert next_perm([3, 2, 1]) == [1, 2, 3]
assert next_perm([1, 5, 1]) == [5, 1, 1]
assert next_perm([1, 1, 5]) == [1, 5, 1]
assert next_perm([5]) == [5]
""",
    ),
    dict(
        id="word_break", tier=2, tag="algorithms",
        title="Word break",
        prompt="""Implement `word_break(s, words)` returning True iff s can be
segmented into a sequence of dictionary words (words reusable). The empty
string is considered segmentable.""",
        tests="""
assert word_break("leetcode", ["leet", "code"]) is True
assert word_break("catsandog", ["cats", "dog", "sand", "and", "cat"]) is False
assert word_break("", ["a"]) is True
assert word_break("aaaa", ["aa"]) is True
assert word_break("ab", ["a"]) is False
""",
    ),
    dict(
        id="edit_distance", tier=2, tag="dp",
        title="Edit distance",
        prompt="""Implement `edit_distance(a, b)` returning the minimum number of
single-character insertions, deletions, or substitutions turning a into b.""",
        tests="""
assert edit_distance("kitten", "sitting") == 3
assert edit_distance("horse", "ros") == 3
assert edit_distance("abc", "abc") == 0
assert edit_distance("", "abc") == 3
assert edit_distance("intention", "execution") == 5
""",
    ),
    dict(
        id="histogram", tier=2, tag="algorithms",
        title="Largest rectangle in histogram",
        prompt="""Implement `largest_rectangle(heights)` returning the area of the
largest axis-aligned rectangle inside the histogram of bar heights (widths 1).
Return 0 for an empty list.""",
        tests="""
assert largest_rectangle([2, 1, 5, 6, 2, 3]) == 10
assert largest_rectangle([2, 4]) == 4
assert largest_rectangle([]) == 0
assert largest_rectangle([3, 3, 3]) == 9
assert largest_rectangle([1]) == 1
""",
    ),
    dict(
        id="course_order", tier=2, tag="graphs",
        title="Course ordering",
        prompt="""Implement `course_order(n, prereqs)` where prereqs is a list of
`[course, prerequisite]` pairs; return ANY valid study order as a list of all
n courses, or [] if impossible. There may be many valid orders.""",
        tests="""
def check(order, n, pairs):
    assert isinstance(order, list) and sorted(order) == list(range(n))
    pos = {c: i for i, c in enumerate(order)}
    for c, p in pairs:
        assert pos[p] < pos[c]

o = course_order(4, [[1, 0], [2, 0], [3, 1], [3, 2]])
check(o, 4, [[1, 0], [2, 0], [3, 1], [3, 2]])
assert course_order(2, [[1, 0], [0, 1]]) == []
assert course_order(1, []) == [0]
o2 = course_order(3, [[2, 1]])
check(o2, 3, [[2, 1]])
""",
    ),
    dict(
        id="running_median", tier=2, tag="data-structures",
        title="Running median",
        prompt="""Implement class `RunningMedian` with `add(x)` and `median()`
returning the median of all values so far as a float (mean of the two middle
values when the count is even). median() is only called when non-empty.""",
        tests="""
m = RunningMedian()
m.add(5)
assert m.median() == 5.0
m.add(15)
assert m.median() == 10.0
m.add(1)
assert m.median() == 5.0
m.add(3)
assert m.median() == 4.0
m.add(100)
assert m.median() == 5.0
""",
    ),
    dict(
        id="coin_change", tier=2, tag="dp",
        title="Coin change",
        prompt="""Implement `coin_change(coins, amount)` returning the fewest coins
to make amount (coins unlimited supply) or -1 if impossible.""",
        tests="""
assert coin_change([1, 2, 5], 11) == 3
assert coin_change([2], 3) == -1
assert coin_change([], 0) == 0
assert coin_change([1], 0) == 0
assert coin_change([186, 419, 83, 408], 6249) == 20
""",
    ),
    dict(
        id="balanced_code", tier=2, tag="parsing",
        title="Balanced brackets with strings",
        prompt="""Implement `is_balanced_code(s)` for pseudo-code text: brackets
`()[]{}` must balance, but brackets inside single- or double-quoted string
literals do not count. Inside a literal a backslash escapes the next char;
an unterminated literal swallows the rest of the input.""",
        tests="""
assert is_balanced_code('f("([)]")') is True
assert is_balanced_code('f("[{]")') is True
assert is_balanced_code('a[{]') is False
assert is_balanced_code('if (a[0] == "]") { b(); }') is True
assert is_balanced_code("f('a)', b") is False
assert is_balanced_code("'()") is True
assert is_balanced_code("f('\\\\'')") is True
""",
    ),
    dict(
        id="eval_expr", tier=2, tag="parsing",
        title="Expression evaluator",
        prompt="""Implement `eval_expr(s)` evaluating an arithmetic expression of
non-negative integers with `+ - * /` and parentheses, standard precedence,
optional spaces. Division truncates toward zero and applies to its operand
subexpression results. No unary minus appears in input.""",
        tests="""
assert eval_expr("3+2*2") == 7
assert eval_expr("3/2") == 1
assert eval_expr("(1+(4+5+2)-3)") == 9
assert eval_expr("2*(5+5*2)/3+(6/2+8)") == 21
assert eval_expr("14-3*2") == 8
assert eval_expr("10/3*3") == 9
""",
    ),
    dict(
        id="longest_consec", tier=2, tag="algorithms",
        title="Longest consecutive run",
        prompt="""Implement `longest_consecutive(nums)` returning the length of the
longest run of consecutive integers present in nums (duplicates ignored), 0
for an empty list. O(n) expected.""",
        tests="""
assert longest_consecutive([100, 4, 200, 1, 3, 2]) == 4
assert longest_consecutive([]) == 0
assert longest_consecutive([1, 1]) == 1
assert longest_consecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]) == 9
""",
    ),
    dict(
        id="parse_duration", tier=2, tag="parsing",
        title="Duration parser",
        prompt="""Implement `parse_duration(s)` returning seconds for durations made
of optional `Nh`, `Nm`, `Ns` parts IN THAT ORDER (e.g. "2h5m10s"); "" means 0.
Anything else (wrong order, unknown unit, empty unit, stray digits) raises
ValueError.""",
        tests="""
assert parse_duration("1h") == 3600
assert parse_duration("2h5m10s") == 7510
assert parse_duration("90s") == 90
assert parse_duration("") == 0
assert parse_duration("45m") == 2700
for bad in ["1d", "h1", "1h1m1", "1m2h", "1hh"]:
    try:
        parse_duration(bad)
        assert False, bad
    except ValueError:
        pass
""",
    ),
    dict(
        id="restore_ips", tier=2, tag="backtracking",
        title="Restore IP addresses",
        prompt="""Implement `restore_ips(s)` returning the SORTED list of all valid
IPv4 addresses obtainable by inserting 4-part dots into the digit string s
(no leading zeros in a part, parts 0-255, all digits used).""",
        tests="""
assert restore_ips("25525511135") == ["255.255.11.135", "255.255.111.35"]
assert restore_ips("0000") == ["0.0.0.0"]
assert restore_ips("10203") == ["1.0.20.3", "10.2.0.3"]
assert restore_ips("11121") == ["1.1.1.21", "1.1.12.1", "1.11.2.1", "11.1.2.1"]
assert restore_ips("12345678901234") == []
""",
    ),
    dict(
        id="spiral_order", tier=2, tag="matrix",
        title="Spiral order",
        prompt="""Implement `spiral_order(m)` returning the elements of the (possibly
non-square) matrix m in clockwise spiral order starting top-left.""",
        tests="""
assert spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [1, 2, 3, 6, 9, 8, 7, 4, 5]
assert spiral_order([[1, 2, 3], [4, 5, 6]]) == [1, 2, 3, 6, 5, 4]
assert spiral_order([[7]]) == [7]
assert spiral_order([[1, 2], [3, 4], [5, 6]]) == [1, 2, 4, 6, 5, 3]
""",
    ),
    dict(
        id="min_window", tier=2, tag="algorithms",
        title="Minimum window length",
        prompt="""Implement `min_window_len(s, t)` returning the LENGTH of the shortest
substring of s containing every character of t with at least its multiplicity,
or -1 if none exists. (Only the length is required, not the window itself.)""",
        tests="""
assert min_window_len("ADOBECODEBANC", "ABC") == 4
assert min_window_len("a", "aa") == -1
assert min_window_len("aa", "a") == 1
assert min_window_len("aab", "aba") == 3
assert min_window_len("", "a") == -1
""",
    ),

    dict(
        id='interval_overlap_count', tier=2, intaked=True, tag='algorithms',
        title='Maximum interval overlap',
        prompt="""Implement `interval_overlap_count(intervals)` where intervals is a list of closed integer intervals `[start, end]`. Return the maximum number of intervals containing the same integer point. Intervals with equal endpoints overlap. An empty list returns 0. Raise ValueError if any interval has start > end. The result counts duplicate intervals separately.""",
        tests="""
assert interval_overlap_count([[1, 5], [2, 6], [4, 8]]) == 3
assert interval_overlap_count([]) == 0
assert interval_overlap_count([[1, 2], [2, 3]]) == 2
assert interval_overlap_count([[1, 1], [1, 1], [2, 2]]) == 2
assert interval_overlap_count([[-5, -1], [-3, 0], [1, 4]]) == 2
try:
 interval_overlap_count([[3, 2]])
 assert False
except ValueError:
 pass
""",
    ),
    dict(
        id='canonical_fraction', tier=2, intaked=True, tag='math',
        title='Canonical fraction',
        prompt="""Implement `canonical_fraction(n, d)` returning a pair `[numerator, denominator]` representing the reduced fraction n/d. The denominator must always be positive. Divide both values by their greatest common divisor. Zero is represented as [0, 1]. Raise ZeroDivisionError when d is zero. Inputs are integers.""",
        tests="""
assert canonical_fraction(6, 8) == [3, 4]
assert canonical_fraction(0, 7) == [0, 1]
assert canonical_fraction(-6, 8) == [-3, 4]
assert canonical_fraction(6, -8) == [-3, 4]
assert canonical_fraction(-12, -18) == [2, 3]
assert canonical_fraction(7, 1) == [7, 1]
try:
 canonical_fraction(1, 0)
 assert False
except ZeroDivisionError:
 pass
""",
    ),
    dict(
        id='deduplicate_stable', tier=2, intaked=True, tag='lists',
        title='Stable deduplication',
        prompt="""Implement `deduplicate_stable(items)` returning a new list containing each distinct item exactly once, preserving the order of its first occurrence. Items are hashable. An empty input returns an empty list. The original list must not be modified.""",
        tests="""
assert deduplicate_stable([3, 1, 3, 2, 1]) == [3, 1, 2]
assert deduplicate_stable([]) == []
assert deduplicate_stable([1, 1, 1]) == [1]
assert deduplicate_stable([0, False, 1, True]) == [0, 1]
assert deduplicate_stable(["a", "b", "a", "c"]) == ["a", "b", "c"]
x = [2, 1, 2]
y = deduplicate_stable(x)
assert x == [2, 1, 2] and y == [2, 1]
""",
    ),
    dict(
        id='lis_reconstruct', tier=3, intaked=True, tag='dp',
        title='Reconstruct a longest increasing subsequence',
        prompt="""Implement `lis_reconstruct(nums)` returning one longest strictly increasing subsequence as a list of values. The selected indices must be increasing. If multiple longest subsequences exist, return the lexicographically smallest value sequence. An empty input returns []. Values are integers. The input is not modified.""",
        tests="""
assert lis_reconstruct([3, 1, 2, 5, 4]) == [1, 2, 4]
assert lis_reconstruct([]) == []
assert lis_reconstruct([4, 3, 2, 1]) == [1]
assert lis_reconstruct([1, 2, 2, 3]) == [1, 2, 3]
assert lis_reconstruct([2, 1, 3]) == [1, 3]
assert lis_reconstruct([3, 1, 2, 4, 0, 5]) == [1, 2, 4, 5]
""",
    ),

    dict(
        id='longest_k_distinct', tier=2, intaked=True, tag='strings',
        title='Longest substring with at most K distinct characters',
        prompt="""Implement longest_k_distinct(s, k) returning the length of the longest contiguous substring of s containing at most k distinct characters. If k <= 0 or s is empty, return 0.""",
        tests="""
assert longest_k_distinct('eceba', 2) == 3
assert longest_k_distinct('', 3) == 0
assert longest_k_distinct('aa', 1) == 2
assert longest_k_distinct('abc', 0) == 0
assert longest_k_distinct('abaccc', 2) == 4
assert longest_k_distinct('abc', 5) == 3
assert longest_k_distinct('aabbcc', 1) == 2
""",
    ),
    dict(
        id='egyptian_greedy', tier=2, intaked=True, tag='math',
        title='Greedy Egyptian fraction',
        prompt="""Implement egyptian(num, den) returning the list of denominators of the greedy Egyptian-fraction expansion of num/den (a sum of distinct unit fractions, always taking the largest unit fraction that does not exceed the remaining value). Input need not be reduced. Use exact integer arithmetic. If num == 0 return []. Raise ValueError if den <= 0, num < 0, or num >= den.""",
        tests="""
def raises(*a):
    try:
        egyptian(*a)
    except ValueError:
        return True
    return False
assert egyptian(2, 3) == [2, 6]
assert egyptian(4, 13) == [4, 18, 468]
assert egyptian(6, 14) == [3, 11, 231]
assert egyptian(1, 5) == [5]
assert egyptian(0, 5) == []
assert egyptian(3, 4) == [2, 4]
assert raises(5, 5) and raises(1, 0) and raises(-1, 3)
""",
    ),
    dict(
        id='combo_sum_once', tier=2, intaked=True, tag='backtracking',
        title='Combination sum, each element once',
        prompt="""Implement combo_sum_once(nums, target) where nums is a list of positive integers (may contain duplicates) and target is a non-negative integer. Return all UNIQUE combinations (as lists sorted ascending) whose elements sum to target, using each element of nums at most once. Two combinations are the same if they contain the same multiset of values. Return the list of combinations sorted lexicographically. If target == 0 the empty combination counts, so return [[]]. If no combination exists return [].""",
        tests="""
assert combo_sum_once([10, 1, 2, 7, 6, 1, 5], 8) == [[1, 1, 6], [1, 2, 5], [1, 7], [2, 6]]
assert combo_sum_once([2, 5, 2, 1, 2], 5) == [[1, 2, 2], [5]]
assert combo_sum_once([1], 0) == [[]]
assert combo_sum_once([3], 2) == []
assert combo_sum_once([1, 1, 1], 2) == [[1, 1]]
assert combo_sum_once([], 4) == []
""",
    ),
    dict(
        id='set_zeroes', tier=2, intaked=True, tag='matrix',
        title='Zero out rows and columns',
        prompt="""Implement set_zeroes(matrix) where matrix is a list of equal-length lists of ints. Return a NEW matrix in which every row and every column that contains a 0 in the original is entirely 0. The input must not be mutated. An empty matrix returns [].""",
        tests="""
assert set_zeroes([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) == [[1, 0, 1], [0, 0, 0], [1, 0, 1]]
assert set_zeroes([[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]]) == [[0, 0, 0, 0], [0, 4, 5, 0], [0, 3, 1, 0]]
assert set_zeroes([]) == []
assert set_zeroes([[5]]) == [[5]]
assert set_zeroes([[0]]) == [[0]]
m = [[0, 1], [1, 1]]
assert set_zeroes(m) == [[0, 0], [0, 1]]
assert m == [[0, 1], [1, 1]]
""",
    ),
    dict(
        id='kth_largest_stream', tier=2, intaked=True, tag='data-structures',
        title='Kth largest after each element',
        prompt="""Implement kth_largest_stream(k, nums) that processes nums left to right and returns a list with one entry per element: the k-th largest value among the elements seen so far (duplicates count separately, so [5, 5] has 2nd largest 5). If fewer than k elements have been seen, the entry is None. Raise ValueError if k <= 0. An empty nums returns [].""",
        tests="""
def raises(*a):
    try:
        kth_largest_stream(*a)
    except ValueError:
        return True
    return False
assert kth_largest_stream(3, [4, 5, 8, 2, 3, 5, 10, 9, 10]) == [None, None, 4, 4, 4, 5, 5, 8, 9]
assert kth_largest_stream(1, [3, 1, 4]) == [3, 3, 4]
assert kth_largest_stream(2, [-1, -5, -3]) == [None, -5, -3]
assert kth_largest_stream(2, []) == []
assert kth_largest_stream(2, [5, 5, 5]) == [None, 5, 5]
assert raises(0, [1])
""",
    ),
    dict(
        id='lcs_string', tier=3, intaked=True, tag='dp',
        title='Longest common subsequence (reconstruct)',
        prompt="""Implement lcs(a, b) returning a longest common subsequence of strings a and b as a string (characters in order, not necessarily contiguous). If several longest common subsequences exist, ANY one is accepted. If there is none, return ''.""",
        tests="""
def is_sub(t, s):
 it = iter(s)
 return all(c in it for c in t)
o = lcs('ABCBDAB', 'BDCABA')
assert len(o) == 4 and is_sub(o, 'ABCBDAB') and is_sub(o, 'BDCABA')
assert lcs('AGGTAB', 'GXTXAYB') == 'GTAB'
assert lcs('abc', 'def') == ''
assert lcs('', 'abc') == ''
assert lcs('abc', 'abc') == 'abc'
assert lcs('aXbXc', 'abc') == 'abc'
""",
    ),
    dict(
        id='assign_rooms', tier=3, intaked=True, tag='data-structures',
        title='Meeting room assignment',
        prompt="""Implement assign_rooms(meetings) where meetings is a list of [start, end] pairs with start < end, treated as half-open [start, end) (a room is free again at time end, so a meeting starting exactly at end may reuse it). Process meetings in order of (start, original index). Each meeting is given the free room with the SMALLEST room number; if no room is free, a new room with the next unused number (0, 1, 2, ...) is opened. Return a list where entry i is the room number assigned to meetings[i]. Empty input returns [].""",
        tests="""
assert assign_rooms([[0, 30], [5, 10], [15, 20]]) == [0, 1, 1]
assert assign_rooms([[1, 3], [3, 5]]) == [0, 0]
assert assign_rooms([]) == []
assert assign_rooms([[1, 5], [1, 5], [2, 3]]) == [0, 1, 2]
assert assign_rooms([[2, 4], [0, 2], [2, 6]]) == [0, 0, 1]
assert assign_rooms([[0, 10], [1, 2], [3, 4], [5, 6]]) == [0, 1, 1, 1]
""",
    ),
    dict(
        id='bipartition_coloring', tier=3, intaked=True, tag='graphs',
        title='Two-coloring a graph',
        prompt="""Implement bipartition(n, edges) for an undirected graph with nodes 0..n-1 and edges given as a list of [u, v] pairs (the graph may be disconnected, may contain duplicate edges, and may contain self-loops). If the graph is bipartite return a list of length n whose entries are 0 or 1 such that every edge connects nodes of different colors; any valid coloring is accepted. If it is not bipartite (including any self-loop) return None. n == 0 returns [].""",
        tests="""
def check(col, n, edges):
    assert isinstance(col, list) and len(col) == n
    assert all(c in (0, 1) for c in col)
    for u, v in edges:
        assert col[u] != col[v]

e1 = [[0, 1], [1, 2], [2, 3], [3, 0]]
check(bipartition(4, e1), 4, e1)
assert bipartition(3, [[0, 1], [1, 2], [2, 0]]) is None
assert bipartition(0, []) == []
assert bipartition(2, [[1, 1]]) is None
e2 = [[0, 1], [2, 3], [0, 1]]
check(bipartition(5, e2), 5, e2)
assert bipartition(5, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]) is None
""",
    ),

    dict(
        id='decode_escaped_fields', tier=3, intaked=True, tag='parsing',
        title='Decode escaped delimited fields',
        prompt="""Implement `decode_escaped_fields(s)` parsing a single line into fields separated by unescaped commas. Backslash escapes the next character, regardless of what it is. An escaped comma belongs to the current field, and an escaped backslash becomes one literal backslash. Empty fields are preserved, including leading, trailing, and consecutive empty fields. An unescaped comma terminates a field. A trailing unmatched backslash raises ValueError. Empty input represents one empty field and returns [""].""",
        tests="""
assert decode_escaped_fields("a,b,c") == ["a", "b", "c"]
assert decode_escaped_fields("") == [""]
assert decode_escaped_fields(",,") == ["", "", ""]
assert decode_escaped_fields("a\\\\,b,c") == ["a,b", "c"]
assert decode_escaped_fields("x\\\\\\\\y,z") == ["x\\\\y", "z"]
assert decode_escaped_fields("\\\\,a,") == [",a", ""]
try:
 decode_escaped_fields("abc\\\\")
 assert False
except ValueError:
 pass
""",
    ),

    dict(
        id='shortest_cycle_length', tier=3, intaked=True, tag='graphs',
        title='Shortest cycle in an undirected graph',
        prompt="""Implement `shortest_cycle_length(n, edges)` for an undirected graph with vertices numbered 0 through n-1. Each edge is a pair [u, v]. Return the number of edges in the shortest simple cycle, or -1 if no cycle exists. A self-loop is a cycle of length 1. Duplicate edges are ignored. Raise ValueError if an endpoint is outside the valid vertex range. The graph may be disconnected.""",
        tests="""
assert shortest_cycle_length(3, [[0, 1], [1, 2], [2, 0]]) == 3
assert shortest_cycle_length(1, []) == -1
assert shortest_cycle_length(2, [[0, 1], [0, 1]]) == -1
assert shortest_cycle_length(3, [[0, 0], [0, 1]]) == 1
assert shortest_cycle_length(5, [[0, 1], [1, 2], [2, 0], [2, 3], [3, 4], [4, 2]]) == 3
assert shortest_cycle_length(4, [[0, 1], [1, 2], [2, 3]]) == -1
try: r = shortest_cycle_length(2, [[0, 2]])
except ValueError: r = "err"
assert r == "err"
""",
    ),

    dict(
        id='parse_kv_escaped', tier=3, intaked=True, tag='parsing',
        title='Escaped key-value parser',
        prompt="""Implement parse_kv(s) that parses a string of entries separated by ';' into a dict of str to str. Rules: within an entry, the FIRST unescaped '=' separates key from value; any later unescaped '=' is an ordinary character of the value. A backslash escapes the next character (any character), which is taken literally and has no special meaning, and the backslash itself is dropped. An entry with zero characters (e.g. from ';;' or a leading/trailing ';') is skipped. An entry with no unescaped '=' maps its key to ''. An entry like '=v' has key ''. If the same key appears again, the later value overwrites. A backslash as the very last character of s raises ValueError. An empty string returns {}.""",
        tests="""
BS = chr(92)
def raises(f, *a):
    try:
        f(*a)
    except ValueError:
        return True
    return False
assert parse_kv('a=1;b=2') == {'a': '1', 'b': '2'}
assert parse_kv('a=x' + BS + ';y;b=p' + BS + '=q=r') == {'a': 'x;y', 'b': 'p=q=r'}
assert parse_kv('') == {}
assert parse_kv(';;a;;') == {'a': ''}
assert parse_kv('k=1;k=2') == {'k': '2'}
assert parse_kv('p' + BS + BS + 'q=' + BS + BS) == {'p' + BS + 'q': BS}
assert parse_kv('a==b') == {'a': '=b'}
assert parse_kv('=v') == {'': 'v'}
assert raises(parse_kv, 'a=b' + BS)
""",
    ),

    dict(
        id='longest_bounded_window', tier=2, intaked=True, tag='algorithms',
        title='Longest bounded-range window',
        prompt="""Implement `longest_bounded_window(nums, limit)` returning the length of the longest contiguous subarray whose maximum minus minimum is at most limit. An empty input returns 0. If limit is negative, return 0. Only the length is returned, so ties need no special handling. All values and limit are integers.""",
        tests="""
assert longest_bounded_window([8, 2, 4, 7], 4) == 2
assert longest_bounded_window([], 3) == 0
assert longest_bounded_window([1, 1, 1], 0) == 3
assert longest_bounded_window([1, 5, 9], -1) == 0
assert longest_bounded_window([10, 1, 2, 4, 7, 2], 5) == 4
assert longest_bounded_window([1, 9, 2, 8], 7) == 3
""",
    ),
    dict(
        id='matrix_diagonal_sums', tier=2, intaked=True, tag='matrix',
        title='Matrix diagonal sums',
        prompt="""Implement `matrix_diagonal_sums(matrix)` for an n by n square matrix of integers. Return `[primary_sum, secondary_sum]`. The primary diagonal runs from top-left to bottom-right; the secondary diagonal runs from top-right to bottom-left. For a 1 by 1 matrix, the same element contributes to both sums. An empty matrix returns [0, 0]. Raise ValueError if the matrix is not square.""",
        tests="""
assert matrix_diagonal_sums([[1, 2], [3, 4]]) == [5, 5]
assert matrix_diagonal_sums([]) == [0, 0]
assert matrix_diagonal_sums([[7]]) == [7, 7]
assert matrix_diagonal_sums([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [15, 15]
assert matrix_diagonal_sums([[1, 0, 0], [0, 2, 0], [0, 0, 3]]) == [6, 2]
try: r = matrix_diagonal_sums([[1, 2], [3]])
except ValueError: r = "err"
assert r == "err"
""",
    ),
    dict(
        id='normalize_identifier', tier=2, intaked=True, tag='strings',
        title='Strict identifier normalization',
        prompt="""Implement `normalize_identifier(s)`. First strip all leading and trailing ASCII space characters. Then convert ASCII uppercase letters A-Z to lowercase. Replace each maximal run of remaining internal ASCII spaces with one underscore. Tabs and all other characters are ordinary characters and are preserved unchanged. Empty or whitespace-only ASCII-space input returns an empty string.""",
        tests="""
assert normalize_identifier(" Hello WORLD ") == "hello_world"
assert normalize_identifier("") == ""
assert normalize_identifier(" ") == ""
assert normalize_identifier("A\\t B") == "a\\t_b"
assert normalize_identifier("A B C") == "a_b_c"
assert normalize_identifier("BC ") == "bc"
""",
    ),

    dict(
        id='streaming_kth_largest', tier=3, intaked=True, tag='data-structures',
        title='Streaming kth largest',
        prompt="""Implement `streaming_kth_largest(nums, k)` returning a list containing the kth largest value after each insertion, starting with the kth insertion. Duplicates count as separate values. If fewer than k values have been inserted, emit nothing. Raise ValueError if k is less than 1. The input is processed in its given order.""",
        tests="""
assert streaming_kth_largest([3, 1, 5, 12, 2, 11], 3) == [1, 3, 3, 5]
assert streaming_kth_largest([], 1) == []
assert streaming_kth_largest([4, 4, 4], 2) == [4, 4]
assert streaming_kth_largest([1, 2, 3, 4], 1) == [1, 2, 3, 4]
assert streaming_kth_largest([9, 8, 7], 4) == []
try: r = streaming_kth_largest([1], 0)
except ValueError: r = "err"
assert r == "err"
""",
    ),

    dict(
        id='fraction_add', tier=2, intaked=True, tag='math',
        title='Add and normalize fractions',
        prompt="""Implement `fraction_add(a, b)` where each argument is a pair `(numerator, denominator)` of integers representing a rational number. Return the sum as a reduced pair `(numerator, denominator)`. The denominator must be positive, and numerator and denominator must be coprime. Return (0, 1) for a zero result. Raise ValueError if either denominator is zero. Use exact integer arithmetic only.""",
        tests="""
assert fraction_add((1, 2), (1, 3)) == (5, 6)
assert fraction_add((1, 2), (-1, 2)) == (0, 1)
assert fraction_add((1, -2), (1, 3)) == (-1, 6)
assert fraction_add((2, 4), (2, 4)) == (1, 1)
assert fraction_add((0, 7), (3, 9)) == (1, 3)
try:
    fraction_add((1, 0), (1, 2))
    assert False
except ValueError:
    pass
""",
    ),
    dict(
        id='stable_frequency_sort', tier=2, intaked=True, tag='lists',
        title='Stable frequency ordering',
        prompt="""Implement `stable_frequency_sort(nums)` returning the elements sorted by descending frequency. Elements with equal frequencies must retain the order in which their values first appeared in the input. Repeated occurrences of each value must remain grouped together. The input contains integers and may be empty. Return a new list without modifying the input.""",
        tests="""
assert stable_frequency_sort([4, 6, 4, 6, 4, 2]) == [4, 4, 4, 6, 6, 2]
assert stable_frequency_sort([]) == []
assert stable_frequency_sort([3, 2, 1]) == [3, 2, 1]
assert stable_frequency_sort([2, 2, 1, 1, 3, 3]) == [2, 2, 1, 1, 3, 3]
assert stable_frequency_sort([5, 5, 4, 4, 4, 5]) == [5, 5, 5, 4, 4, 4]
assert stable_frequency_sort([-1, 0, -1, 0, -1]) == [-1, -1, -1, 0, 0]
""",
    ),
    dict(
        id='merge_sorted_streams', tier=2, intaked=True, tag='data-structures',
        title='Merge sorted streams',
        prompt="""Implement `merge_sorted_streams(streams)` where streams is a list of individually nondecreasing integer lists. Return one globally nondecreasing list containing every element. Duplicate values must be preserved. Empty streams are allowed. The input may be empty, and the result must not modify the input.""",
        tests="""
assert merge_sorted_streams([[1, 4, 7], [2, 5], [3, 6]]) == [1, 2, 3, 4, 5, 6, 7]
assert merge_sorted_streams([]) == []
assert merge_sorted_streams([[], []]) == []
assert merge_sorted_streams([[1, 1], [1], [1, 1]]) == [1, 1, 1, 1, 1]
assert merge_sorted_streams([[5], [-1, 0], [2, 9]]) == [-1, 0, 2, 5, 9]
assert merge_sorted_streams([[1, 3, 3], [2, 3, 4]]) == [1, 2, 3, 3, 3, 4]
""",
    ),
    dict(
        id='weighted_interval_schedule', tier=3, intaked=True, tag='dp',
        title='Weighted interval scheduling with reconstruction',
        prompt="""Implement `weighted_interval_schedule(intervals)` where each interval is [start, end, weight], with integer start <= end and integer weight. Return a list of original interval indices forming a non-overlapping schedule with maximum total weight. Intervals are half-open [start, end), so touching endpoints are compatible. Empty intervals (start == end) are allowed and are compatible with every interval. The returned indices must be in chronological order by start, then end, then original index. On equal maximum total weight, choose the schedule with fewer intervals; if still tied, choose the lexicographically smallest list of original indices after sorting by the required chronological order. Empty input returns [].""",
        tests="""
def check_schedule(intervals, indices):
    assert indices == sorted(indices, key=lambda i: (intervals[i][0], intervals[i][1], i))
    total = sum(intervals[i][2] for i in indices)
    for p in range(len(indices)):
        for q in range(p + 1, len(indices)):
            a, b = intervals[indices[p]], intervals[indices[q]]
            assert a[1] <= b[0] or b[1] <= a[0]
    return total

x = [[1, 3, 5], [3, 5, 6], [1, 5, 10]]
r = weighted_interval_schedule(x)
assert check_schedule(x, r) == 11
assert weighted_interval_schedule([]) == []
x = [[0, 2, 5], [2, 4, 5], [0, 4, 10]]
assert weighted_interval_schedule(x) == [2]
x = [[0, 1, 0], [1, 2, 0]]
assert weighted_interval_schedule(x) == []
x = [[0, 2, 5], [2, 4, 5], [4, 6, 10]]
r = weighted_interval_schedule(x)
assert r == [0, 1, 2] and check_schedule(x, r) == 20
""",
    ),
    dict(
        id='shortest_path_with_count', tier=3, intaked=True, tag='graphs',
        title='Shortest path and number of shortest paths',
        prompt="""Implement `shortest_path_with_count(n, edges, source, target)` for an undirected, unweighted graph with vertices 0 through n-1. Edges are pairs [u, v]; duplicate edges are ignored. Return `(distance, count)` where distance is the number of edges in a shortest path and count is the number of distinct shortest vertex sequences from source to target. If target is unreachable, return (-1, 0). If source == target, return (0, 1). Vertices and endpoints are guaranteed valid.""",
        tests="""
assert shortest_path_with_count(5, [[0, 1], [1, 4], [0, 2], [2, 4], [0, 3]], 0, 4) == (2, 2)
assert shortest_path_with_count(1, [], 0, 0) == (0, 1)
assert shortest_path_with_count(3, [[0, 1]], 0, 2) == (-1, 0)
assert shortest_path_with_count(4, [[0, 1], [1, 2], [2, 3], [0, 2]], 0, 3) == (2, 1)
assert shortest_path_with_count(4, [[0, 1], [0, 1], [1, 2], [0, 2]], 0, 2) == (1, 1)
assert shortest_path_with_count(6, [[0, 1], [0, 2], [1, 3], [2, 3], [1, 4], [2, 4], [3, 5], [4, 5]], 0, 5) == (3, 4)
""",
    ),
    dict(
        id='normalize_path', tier=3, intaked=True, tag='strings',
        title='Normalize a Unix-style path',
        prompt="""Implement `normalize_path(path)` for an absolute Unix-style path. The input starts with '/'. Collapse repeated slashes, remove '.' components, and resolve '..' by removing the preceding ordinary component when one exists. Attempts to move above the root leave the path at root. Return '/' for an empty resulting component list. Do not interpret backslashes as separators. No filesystem access is allowed.""",
        tests="""
assert normalize_path("/a/b/../c") == "/a/c"
assert normalize_path("/") == "/"
assert normalize_path("/../../a") == "/a"
assert normalize_path("/a//b/./c/../") == "/a/b"
assert normalize_path("/a/../../..") == "/"
assert normalize_path("/a\\\\b/../c") == "/c"
""",
    ),

    dict(
        id='longest_unique_window', tier=2, intaked=True, tag='strings',
        title='Longest unique-character window',
        prompt="""Implement `longest_unique_window(s)` returning a pair `(start, length)` describing the longest contiguous substring containing no repeated characters. Indices are zero-based. If multiple windows have the same maximum length, return the one with the smallest start index. For an empty string return (0, 0). Characters are compared exactly, without normalization.""",
        tests="""
assert longest_unique_window("abcabcbb") == (0, 3)
assert longest_unique_window("") == (0, 0)
assert longest_unique_window("aaaa") == (0, 1)
assert longest_unique_window("abcaef") == (1, 5)
assert longest_unique_window("abba") == (0, 2)
assert longest_unique_window("abcdef") == (0, 6)
""",
    ),

    dict(
        id='count_coprime_inclusion_exclusion', tier=3, intaked=True, tag='math',
        title='Coprime counter via inclusion-exclusion',
        prompt="""Implement count_coprime(n, m): return how many integers in [1, n] are coprime to m. Use the inclusion-exclusion principle over the distinct prime factors of m (n <= 1012 in principle, but keep inputs small). Edge semantics: count_coprime(0, m) == 0; m == 1 means every integer in [1, n] counts (1 is coprime to 1).""",
        tests="""
assert count_coprime(10, 12) == 3
assert count_coprime(1, 1) == 1
assert count_coprime(0, 5) == 0
assert count_coprime(20, 1) == 20
assert count_coprime(12, 6) == 4
assert count_coprime(36, 30) == 9
""",
    ),
    dict(
        id='min_eating_speed', tier=2, intaked=True, tag='algorithms',
        title="Binary search on the answer: Koko's eating speed",
        prompt="""Implement min_eating_speed(piles, h): piles is a list of positive integers, h >= len(piles). Koko eats at integer speed k (k >= 1); each pile takes ceil(pile / k) hours and piles are eaten one at a time. Return the minimum k such that total hours <= h. The predicate is monotone in k, so binary search over [1, max(piles)]. Empty piles is not allowed (h >= len(piles) >= 1 guaranteed).""",
        tests="""
assert min_eating_speed([3, 6, 7, 11], 8) == 4
assert min_eating_speed([30, 11, 23, 4, 20], 6) == 23
assert min_eating_speed([7], 5) == 2
assert min_eating_speed([2, 2], 2) == 2
assert min_eating_speed([1000000], 1000000) == 1
""",
    ),
    dict(
        id='tsp_exact_bitmask', tier=3, intaked=True, tag='dp',
        title='Exact TSP tour length with bitmask DP',
        prompt="""Implement tsp_length(dist): dist is an n x n symmetric matrix of non-negative integers (n <= 10) where dist[i][i] == 0. Return the minimum total length of a tour that starts at city 0, visits every city exactly once, and returns to city 0, using the classic bitmask DP over subsets. For n == 1 the answer is 0 (stay at city 0). The answer is unique because it is a minimum.""",
        tests="""
assert tsp_length([[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]) == 80
assert tsp_length([[0]]) == 0
assert tsp_length([[0, 5], [5, 0]]) == 10
assert tsp_length([[0, 1, 100], [1, 0, 1], [100, 1, 0]]) == 102
assert tsp_length([[0, 3, 3, 3], [3, 0, 3, 3], [3, 3, 0, 3], [3, 3, 3, 0]]) == 12
""",
    ),
    dict(
        id='predict_the_winner', tier=2, intaked=True, tag='dp',
        title='Turn-based game: Predict the Winner',
        prompt="""Implement predict_winner(nums): two players take turns picking a number from either end of the list nums (non-empty list of non-negative integers). Player 1 moves first. Both play optimally to maximize their own total minus the opponent's. Return True if Player 1 can tie or win (i.e. Player 1's final total >= Player 2's total), else False. Use interval DP where dp of a range is the best score difference the mover can achieve.""",
        tests="""
assert predict_winner([1, 5, 2]) == False
assert predict_winner([1, 5, 233, 7]) == True
assert predict_winner([0]) == True
assert predict_winner([1, 1]) == True
assert predict_winner([7, 1, 1, 7]) == True
""",
    ),
    dict(
        id='base_neg2_conversion', tier=2, intaked=True, tag='math',
        title='Convert integer to base -2',
        prompt="""Implement base_neg2(n): return the representation of the integer n (n may be negative, zero, or positive; |n| <= 109) in base -2 as a string of '0'/'1' digits with no leading zeros, except the representation of 0 is exactly "0". The digit loop is: r = n mod 2 in {0, 1}; emit r; n = (n - r) // -2; repeat while n != 0, then reverse the digits.""",
        tests="""
assert base_neg2(2) == "110"
assert base_neg2(3) == "111"
assert base_neg2(-1) == "11"
assert base_neg2(0) == "0"
assert base_neg2(6) == "11010"
assert base_neg2(-3) == "1101"
assert base_neg2(1) == "1"
""",
    ),
    dict(
        id='min_path_effort_grid', tier=3, intaked=True, tag='graphs',
        title='Path with minimum effort on a grid (Dijkstra)',
        prompt="""Implement min_effort(heights): heights is a rectangular grid of non-negative integers, at least 1x1. Starting at the top-left cell and ending at the bottom-right cell, you may move up, down, left, or right. The effort of a path is the maximum absolute height difference between consecutive cells along it. Return the minimum possible effort. A 1x1 grid has effort 0. Use Dijkstra where the edge weight between adjacent cells is their absolute difference and the path cost is the max.""",
        tests="""
assert min_effort([[1, 2, 2], [3, 8, 2], [5, 3, 5]]) == 2
assert min_effort([[1, 2, 3], [3, 8, 4], [5, 3, 5]]) == 1
assert min_effort([[5]]) == 0
assert min_effort([[1, 10]]) == 9
assert min_effort([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) == 0
""",
    ),
    dict(
        id='matrix_chain_multiplication', tier=3, intaked=True, tag='dp',
        title='Matrix chain multiplication minimum scalar multiplications',
        prompt="""Implement matrix_chain(dims): dims is a list of d+1 positive integers describing d matrices A_1..A_d where A_i has shape dims[i-1] x dims[i]. Return the minimum total number of scalar multiplications needed to fully multiply the chain, computed by interval DP. If d == 1 (len(dims) == 2) the answer is 0. All intermediate values are exact integers.""",
        tests="""
assert matrix_chain([40, 20, 30, 10, 30]) == 26000
assert matrix_chain([10, 20, 30]) == 6000
assert matrix_chain([10, 20, 30, 40, 30]) == 30000
assert matrix_chain([5, 10]) == 0
assert matrix_chain([1, 2, 3, 4]) == 18
""",
    ),
    dict(
        id='subarrays_sum_equals_k', tier=2, intaked=True, tag='lists',
        title='Count subarrays with sum exactly k',
        prompt="""Implement count_subarrays_k(nums, k): count contiguous subarrays of the integer list nums whose elements sum to exactly k. Integers may be negative or zero, so sliding window does not work; use prefix sums with a hash map counting occurrences of each prefix sum. Edge semantics: an empty list returns 0; an empty subarray is never counted (subarrays are non-empty).""",
        tests="""
assert count_subarrays_k([1, 1, 1], 2) == 2
assert count_subarrays_k([1, 2, 3], 3) == 2
assert count_subarrays_k([3, 4, 7, 2, -3, 1, 4, 2], 7) == 4
assert count_subarrays_k([], 0) == 0
assert count_subarrays_k([0, 0], 0) == 3
assert count_subarrays_k([5], 3) == 0
""",
    ),

    dict(
        id='min_station_refuels', tier=3, intaked=True, tag='data-structures',
        title='Minimum refueling stops',
        prompt="""Implement min_refuels(start_fuel, target, stations): you start with start_fuel liters, drive along a line toward position target (start is position 0), and stations is a list of [position, capacity] pairs. When at a station you may add up to its capacity to your tank (assume unlimited tank size). Return the minimum number of stops needed to reach target, or -1 if impossible. Greedy: drive as far as possible, then retroactively 'use' the largest-capacity passed station via a max-heap. If start_fuel >= target, return 0 even with no stations. Stations may be given in any order; positions are strictly positive.""",
        tests="""
assert min_refuels(10, 100, [[10, 60], [20, 30], [30, 30], [60, 40]]) == 2
assert min_refuels(1, 1, []) == 0
assert min_refuels(1, 100, [[10, 1]]) == -1
assert min_refuels(1, 100, []) == -1
assert min_refuels(50, 100, [[20, 100]]) == 1
""",
    ),
    dict(
        id='render_escaped_template', tier=3, intaked=True, tag='parsing',
        title='Template renderer with tilde escapes',
        prompt="""Implement render(tpl, env): tpl is a string, env is a dict mapping field names to strings. Scan left to right. A backslash-like escape character '' escapes the next single character literally (so '' yields one '', '{' yields '{'). The sequence '{{' begins a field which must be closed by the nearest '}}'; its content is the field name, looked up in env and replaced by its string value. Rules: if '' is the last character of tpl (nothing to escape), raise ValueError. If a field is never closed, raise ValueError. If a field name is missing from env, raise KeyError. A single '{' or '}' (not doubled) is a literal character. Note '{{' inside a field's name text is not special; the first '}}' after the opening '{{' closes the field. Return the rendered string.""",
        tests="""
assert render("Hi {{name}}!", {"name": "Bob"}) == "Hi Bob!"
assert render("{{name}}", {"name": "Bob"}) == "Bob"
assert render("x", {}) == "x"
assert render("{{a}}{{b}}", {"a": "1", "b": "2"}) == "12"
try:
 render("{{z}}", {})
 r = "noerr"
except KeyError:
 r = "err"
assert r == "err"
try:
 render("abc~", {})
 r = "noerr"
except ValueError:
 r = "err"
assert r == "err"
assert render("a{b}c", {}) == "a{b}c"
""",
    ),

    dict(
        id='solve_linear_congruence', tier=2, intaked=True, tag='math',
        title='Smallest solution of a linear congruence',
        prompt="""Implement `solve_linear_congruence(a, b, m)` for integers a and b and a positive integer m. Return the smallest nonnegative integer x satisfying a*x ≡ b (mod m), or -1 if no solution exists. When m = 1 every integer is a solution, so the answer is 0. Inputs satisfy m <= 1000; behavior outside the stated constraints is unspecified.""",
        tests="""
assert solve_linear_congruence(3, 4, 7) == 6
assert solve_linear_congruence(4, 8, 12) == 2
assert solve_linear_congruence(6, 4, 9) == -1
assert solve_linear_congruence(5, 0, 10) == 0
assert solve_linear_congruence(7, 3, 1) == 0
assert solve_linear_congruence(-3, 4, 7) == 1
assert solve_linear_congruence(6, 3, 9) == 2
""",
    ),
    dict(
        id='max_product_after_cut', tier=2, intaked=True, tag='dp',
        title='Maximum product after cutting an integer',
        prompt="""Implement `max_product_after_cut(n)` for an integer n with 2 <= n <= 1000. Split n into at least two positive integers whose sum is n, and return the maximum possible product of those integers. The number and sizes of the parts are otherwise unrestricted.""",
        tests="""
assert max_product_after_cut(10) == 36
assert max_product_after_cut(2) == 1
assert max_product_after_cut(3) == 2
assert max_product_after_cut(4) == 4
assert max_product_after_cut(8) == 18
assert max_product_after_cut(12) == 81
""",
    ),
    dict(
        id='count_digit_occurrences', tier=2, intaked=True, tag='math',
        title='Count a nonzero digit up to n',
        prompt="""Implement `count_digit_occurrences(n, d)` for 0 <= n <= 1000 and a decimal digit d with 1 <= d <= 9. Return the total number of occurrences of d in the ordinary decimal representations of all integers from 0 through n inclusive. There are no leading zeros, and 0 is represented as '0'.""",
        tests="""
assert count_digit_occurrences(12, 1) == 5
assert count_digit_occurrences(0, 1) == 0
assert count_digit_occurrences(9, 1) == 1
assert count_digit_occurrences(99, 1) == 20
assert count_digit_occurrences(100, 1) == 21
assert count_digit_occurrences(1000, 9) == 300
assert count_digit_occurrences(22, 2) == 6
""",
    ),
    dict(
        id='count_divisible_subsets', tier=3, intaked=True, tag='dp',
        title='Count subsets by modular sum',
        prompt="""Implement `count_divisible_subsets(nums, m, target)`, where nums contains at most 1000 integers, m is positive, and 0 <= target < m. Count subsets whose element sum is congruent to target modulo m, returning the count modulo 1000000007. Include the empty subset. Elements at different positions are distinct even when their values are equal, and negative values may occur.""",
        tests="""
assert count_divisible_subsets([1, 2, 3], 3, 0) == 4
assert count_divisible_subsets([1, 1], 2, 0) == 2
assert count_divisible_subsets([], 5, 0) == 1
assert count_divisible_subsets([], 5, 3) == 0
assert count_divisible_subsets([2, 4, 6], 2, 1) == 0
assert count_divisible_subsets([0, 0], 3, 0) == 4
assert count_divisible_subsets([5, 7, 9], 3, 0) == 4
""",
    ),
    dict(
        id='min_assignment_cost', tier=3, intaked=True, tag='dp',
        title='Minimum square assignment cost',
        prompt="""Implement `min_assignment_cost(cost)` for an n by n matrix of integers with 0 <= n <= 8. Choose exactly one entry from each row so that no column is chosen twice, and return the minimum possible sum of the chosen entries. Costs may be negative. For the empty 0 by 0 matrix, return 0; if several assignments attain the minimum, their common sum is the answer.""",
        tests="""
assert min_assignment_cost([[9, 2, 7], [6, 4, 3], [5, 8, 1]]) == 9
assert min_assignment_cost([]) == 0
assert min_assignment_cost([[5]]) == 5
assert min_assignment_cost([[1, 2], [3, 4]]) == 5
assert min_assignment_cost([[0, -1], [-2, 3]]) == -3
assert min_assignment_cost([[1, 100, 100], [100, 1, 100], [100, 100, 1]]) == 3
""",
    ),
    dict(
        id='minimize_largest_block_sum', tier=3, intaked=True, tag='algorithms',
        title='Minimize the largest contiguous block sum',
        prompt="""Implement `minimize_largest_block_sum(nums, k)` for a nonempty list of at most 1000 positive integers and 1 <= k <= len(nums). Partition nums into exactly k nonempty contiguous blocks, preserving order, and return the minimum possible value of the largest block sum. Block boundaries do not otherwise affect the answer.""",
        tests="""
assert minimize_largest_block_sum([7, 2, 5, 10, 8], 2) == 18
assert minimize_largest_block_sum([1, 2, 3], 3) == 3
assert minimize_largest_block_sum([1, 2, 3], 1) == 6
assert minimize_largest_block_sum([5], 1) == 5
assert minimize_largest_block_sum([1, 1, 1, 1], 2) == 2
assert minimize_largest_block_sum([2, 1, 1, 3], 2) == 4
assert minimize_largest_block_sum([4, 2, 4, 1], 3) == 5
""",
    ),
    dict(
        id='count_n_queens', tier=3, intaked=True, tag='backtracking',
        title='Count n-queens arrangements',
        prompt="""Implement `count_n_queens(n)` for 0 <= n <= 10. Return the number of ways to place n queens on an n by n chessboard so that no two queens share a row, column, or diagonal. Reflections and rotations count as different arrangements. By convention, the empty 0 by 0 board has exactly one arrangement.""",
        tests="""
assert count_n_queens(4) == 2
assert count_n_queens(0) == 1
assert count_n_queens(1) == 1
assert count_n_queens(2) == 0
assert count_n_queens(5) == 10
assert count_n_queens(6) == 4
assert count_n_queens(8) == 92
""",
    ),
    dict(
        id='grid_wall_escape', tier=3, intaked=True, tag='graphs',
        title='Shortest grid path with limited wall entries',
        prompt="""Implement `grid_wall_escape(grid, k)` for a nonempty rectangular grid of strings containing only '.' and '#', with nonnegative k. Start at the top-left cell and finish at the bottom-right cell; both are guaranteed open. Moving orthogonally to an adjacent cell costs one step, and a '#' cell may be entered only if doing so does not make the total number of entered wall cells exceed k. Return the shortest path length in steps, or -1 if no permitted path exists. Entering the same wall cell on separate visits counts separately, and staying at the start when it is also the finish takes 0 steps.""",
        tests="""
assert grid_wall_escape(["..#", "##.", "..."], 1) == 4
assert grid_wall_escape(["..#", "##.", "..."], 0) == -1
assert grid_wall_escape(["."], 3) == 0
assert grid_wall_escape(["..."], 0) == 2
assert grid_wall_escape([".#", "#."], 1) == 2
assert grid_wall_escape([".#", "#."], 0) == -1
assert grid_wall_escape(["...", "###", "..."], 1) == 4
""",
    ),
    dict(
        id='min_join_cost', tier=3, intaked=True, tag='data-structures',
        title='Minimum cost to join all lengths',
        prompt="""Implement `min_join_cost(lengths)` for a list of at most 1000 positive integers. Repeatedly choose any two current lengths and replace them by their sum, paying a cost equal to that sum, until one length remains. Return the minimum possible total cost. An empty list or a list containing one length requires no joins and returns 0.""",
        tests="""
assert min_join_cost([4, 3, 2, 6]) == 29
assert min_join_cost([]) == 0
assert min_join_cost([7]) == 0
assert min_join_cost([5, 5, 5]) == 25
assert min_join_cost([1, 1, 1, 1]) == 8
assert min_join_cost([3, 1, 4, 1, 5]) == 30
assert min_join_cost([2, 8]) == 10
""",
    ),
    dict(
        id='count_domino_tilings', tier=3, intaked=True, tag='matrix',
        title='Count domino tilings of a small board',
        prompt="""Implement `count_domino_tilings(rows, cols)` for nonnegative integers whose product is at most 16. Return the number of ways to cover a rows by cols rectangle exactly with 2 by 1 or 1 by 2 dominoes, without overlaps or overhang. Dominoes are indistinguishable, and different placements count separately. If either dimension is 0, the board has no cells and exactly one empty tiling.""",
        tests="""
assert count_domino_tilings(2, 3) == 3
assert count_domino_tilings(2, 2) == 2
assert count_domino_tilings(1, 1) == 0
assert count_domino_tilings(1, 2) == 1
assert count_domino_tilings(3, 4) == 11
assert count_domino_tilings(4, 4) == 36
assert count_domino_tilings(0, 5) == 1
""",
    ),

    dict(
        id='excel_column_title', tier=2, intaked=True, tag='math',
        title='Spreadsheet column label',
        prompt="""Implement excel_column_title(n) that returns the spreadsheet-style column label for the
1-based column number n (an int): 1 -> "A", 2 -> "B", ..., 26 -> "Z", 27 -> "AA",
28 -> "AB", and so on (bijective base-26 using uppercase A-Z; there is no zero digit).
Raise ValueError if n < 1.""",
        tests="""
assert excel_column_title(28) == "AB"
assert excel_column_title(1) == "A"
assert excel_column_title(26) == "Z"
assert excel_column_title(52) == "AZ"
assert excel_column_title(702) == "ZZ"
assert excel_column_title(703) == "AAA"
try: r = excel_column_title(0)
except ValueError: r = "err"
assert r == "err"
""",
    ),
    dict(
        id='trap_rain_water', tier=2, intaked=True, tag='algorithms',
        title='Trapped rain water',
        prompt="""Implement trap_rain_water(heights) where heights is a list of non-negative ints giving
the heights of adjacent bars of width 1. Return the total units of rain water trapped.
The water level above bar i is min(max(heights[:i+1]), max(heights[i:])); water above
that level spills off either end. An empty list returns 0.""",
        tests="""
assert trap_rain_water([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
assert trap_rain_water([]) == 0
assert trap_rain_water([4, 2, 0, 3, 2, 5]) == 9
assert trap_rain_water([5, 4, 3, 2, 1]) == 0
assert trap_rain_water([2, 0, 2]) == 2
assert trap_rain_water([5, 0, 0, 1, 0, 6]) == 19
""",
    ),
    dict(
        id='daily_temperatures', tier=2, intaked=True, tag='data-structures',
        title='Days until warmer',
        prompt="""Implement daily_temperatures(temps) (a list of ints). Return a list of the same length
where entry i is j - i for the smallest j > i with temps[j] STRICTLY greater than temps[i],
or 0 if no such j exists. Equal temperatures do not count as warmer. Empty input returns [].""",
        tests="""
assert daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]) == [1, 1, 4, 2, 1, 1, 0, 0]
assert daily_temperatures([]) == []
assert daily_temperatures([70, 70, 70]) == [0, 0, 0]
assert daily_temperatures([70, 70, 71]) == [2, 1, 0]
assert daily_temperatures([80, 60, 70, 90]) == [3, 1, 1, 0]
assert daily_temperatures([5, 3, 5, 3, 6]) == [4, 1, 2, 1, 0]
""",
    ),
    dict(
        id='word_pattern_bijection', tier=2, intaked=True, tag='dicts',
        title='Word pattern bijection',
        prompt="""Implement word_pattern_bijection(pattern, s) returning True iff the words of s follow
pattern via a bijection, else False (return a bool). Words are obtained with s.split()
(runs of whitespace separate words; leading/trailing whitespace is ignored). The number of
words must equal len(pattern); each pattern character must always map to the same word and
no two distinct characters may map to the same word. Matching is case-sensitive.
word_pattern_bijection("", "") is True.""",
        tests="""
assert word_pattern_bijection("abba", "dog cat cat dog") is True
assert word_pattern_bijection("abba", "dog dog dog dog") is False
assert word_pattern_bijection("aaaa", "dog cat cat dog") is False
assert word_pattern_bijection("ab", "dog cat cat") is False
assert word_pattern_bijection("", "") is True
assert word_pattern_bijection("abc", " b c a ") is True
assert word_pattern_bijection("aA", "x x") is False
""",
    ),
    dict(
        id='count_islands', tier=2, intaked=True, tag='matrix',
        title='Count islands (4-connected)',
        prompt="""Implement count_islands(grid) where grid is a rectangular list of rows of 0/1 ints.
Return the number of islands: maximal groups of 1-cells connected horizontally or
vertically. Diagonal neighbours are NOT connected. [] and [[]] contain 0 islands.
The input grid must not be modified.""",
        tests="""
assert count_islands([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1]]) == 3
g = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
assert count_islands(g) == 5 and g == [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
assert count_islands([]) == 0
assert count_islands([[]]) == 0
assert count_islands([[1, 1, 1], [0, 1, 0], [1, 1, 1]]) == 1
assert count_islands([[1, 0, 1, 0, 1]]) == 3
""",
    ),
    dict(
        id='summary_ranges', tier=2, intaked=True, tag='lists',
        title='Summarize consecutive runs',
        prompt="""Implement summary_ranges(nums) where nums is a list of ints sorted in non-decreasing
order; duplicates may appear and count as a single value. Return a list of strings, in
ascending order, describing the maximal runs of consecutive integers: a run from a to b
with b > a is written "a->b", a single value as "a". Negative numbers keep their minus
sign (e.g. "-3->-1"). Empty input returns [].""",
        tests="""
assert summary_ranges([0, 1, 2, 4, 5, 7]) == ["0->2", "4->5", "7"]
assert summary_ranges([]) == []
assert summary_ranges([-3, -2, -1, 1]) == ["-3->-1", "1"]
assert summary_ranges([1, 1, 2, 2, 4, 4]) == ["1->2", "4"]
assert summary_ranges([5]) == ["5"]
assert summary_ranges([0, 2, 4]) == ["0", "2", "4"]
assert summary_ranges([3, 3]) == ["3"]
""",
    ),
    dict(
        id='knapsack_reconstruct', tier=3, intaked=True, tag='dp',
        title='0/1 knapsack with item reconstruction',
        prompt="""Implement knapsack_reconstruct(weights, values, capacity) for the 0/1 knapsack: item i has
weight weights[i] and value values[i] (non-negative ints) and may be taken at most once.
Return a tuple (best, indices) where best is the maximum total value with total weight
<= capacity, and indices is an ascending list of distinct item indices whose values sum to
exactly best and whose weights sum to at most capacity. If several selections achieve best,
any one is accepted (zero-value items may be included or omitted). Empty item lists return
(0, []). Raise ValueError if len(weights) != len(values), capacity < 0, or any weight or
value is negative.""",
        tests="""
def check(res, w, v, cap, best):
 assert isinstance(res, tuple) and len(res) == 2
 total, idx = res
 assert total == best and idx == sorted(set(idx)) and all(0 <= i < len(w) for i in idx)
 assert sum(w[i] for i in idx) <= cap and sum(v[i] for i in idx) == best

check(knapsack_reconstruct([1, 3, 4, 5], [1, 4, 5, 7], 7), [1, 3, 4, 5], [1, 4, 5, 7], 7, 9)
assert knapsack_reconstruct([], [], 10) == (0, [])
assert knapsack_reconstruct([2, 3], [3, 4], 4) == (4, [1])
assert knapsack_reconstruct([0, 1], [4, 9], 0) == (4, [0])
check(knapsack_reconstruct([2, 2, 3], [3, 3, 1], 2), [2, 2, 3], [3, 3, 1], 2, 3)
try: r = knapsack_reconstruct([1], [1, 2], 5)
except ValueError: r = "err"
assert r == "err"
try: r = knapsack_reconstruct([1], [1], -1)
except ValueError: r = "err"
assert r == "err"
""",
    ),
    dict(
        id='eulerian_trail', tier=3, intaked=True, tag='graphs',
        title='Directed Eulerian trail',
        prompt="""Implement eulerian_trail(edges) for a directed multigraph given as a list of [u, v]
edges (nodes are hashable, e.g. ints or strings; self-loops and parallel edges are allowed).
Return a list of nodes [x0, x1, ..., xm] with m == len(edges) such that the consecutive
pairs (x_i, x{i+1}) use every edge exactly once (as a multiset). Return [] if no such trail
exists (in/out-degree imbalance, or edges that cannot all be reached in one trail) and also
when edges is empty. If several trails exist, any one is accepted.""",
        tests="""
from collections import Counter

def check(path, edges):
 assert isinstance(path, list) and len(path) == len(edges) + 1
 assert Counter(zip(path, path[1:])) == Counter(map(tuple, edges))

e1 = [[2, 3], [1, 2], [3, 1], [1, 4], [4, 2]]
check(eulerian_trail(e1), e1)
e2 = [[0, 1], [1, 2], [2, 0], [1, 3], [3, 1]]
check(eulerian_trail(e2), e2)
assert eulerian_trail([]) == []
assert eulerian_trail([[1, 2], [2, 1], [3, 4], [4, 3]]) == []
assert eulerian_trail([[1, 2], [1, 3]]) == []
e3 = [["a", "a"], ["a", "b"], ["b", "a"], ["a", "b"]]
check(eulerian_trail(e3), e3)
""",
    ),

    dict(
        id='decode_nested_repeat', tier=3, intaked=True, tag='parsing',
        title='Nested repeat decoder with escapes',
        prompt="""Implement decode_nested_repeat(s) that expands repeat groups. A count is one or more ASCII
digits read as a decimal integer (0 is allowed, leading zeros are allowed) and MUST be
immediately followed by [; k[X] expands to the decoded X repeated k times, and groups
nest. A backslash makes the next character literal, so \\\\[, \\\\], \\\\\\\\ and \\\\7 produce
[, ], \\\\ and 7. Every other character is literal. Raise ValueError on: a [ not
immediately preceded by a count, a count followed by anything other than [ (including a
backslash or the end of input), an unmatched ], an unclosed [, or a backslash as the
last character. Empty input returns "".""",
        tests="""
def bad(s):
 try: decode_nested_repeat(s)
 except ValueError: return True
 return False

assert decode_nested_repeat("3[a2[c]]") == "accaccacc"
assert decode_nested_repeat("ab2[cd]e10[x]") == "abcdcde" + "x" * 10
assert decode_nested_repeat("2[a0[b]c]03[z]") == "acaczzz"
assert decode_nested_repeat(r"2[\\[\\]]\\3\\\\") == "[][]3\\\\"
assert decode_nested_repeat("") == ""
assert all(bad(s) for s in ["3[a", "a]", "[a]", "3a", "2[a]3", "ab\\\\", "2\\\\[a]"])
""",
    ),

    dict(
        id='shortest_threshold_slice', tier=2, intaked=True, tag='algorithms',
        title='Shortest slice reaching a signed threshold',
        prompt="""Implement shortest_threshold_slice(nums, threshold). Return the length of the shortest nonempty contiguous slice of nums whose sum is at least threshold. Return 0 if no such slice exists, including when nums is empty. Values and the threshold may be negative or zero; an empty slice never qualifies, even when the threshold is nonpositive. Inputs are guaranteed to be a list of at most 1000 integers and an integer threshold. Do not modify the input. Use O(n) time and O(n) auxiliary space.""",
        tests="""
assert shortest_threshold_slice([2, -1, 2], 3) == 3
assert shortest_threshold_slice([], 0) == 0
assert shortest_threshold_slice([1, 2], 4) == 0
assert shortest_threshold_slice([1, 2, 3], 3) == 1
assert shortest_threshold_slice([84, -37, 32, 40, 95], 167) == 3
assert shortest_threshold_slice([-5], -5) == 1
assert shortest_threshold_slice([-2, -3], 0) == 0
""",
    ),
    dict(
        id='glob_full_match', tier=2, intaked=True, tag='strings',
        title='Whole-string wildcard matching',
        prompt="""Implement glob_full_match(text, pattern) returning a bool indicating whether the entire text matches the entire pattern. Text contains only lowercase ASCII letters. A pattern contains lowercase ASCII letters, '?' matching exactly one letter, and '*' matching any sequence of letters, including the empty sequence. Consecutive stars are allowed and have the same effect as one star. There are no escapes. Both inputs may be empty and have length at most 1000. Thus two empty inputs match, and an empty text matches a nonempty pattern exactly when every pattern character is '*'. Inputs are guaranteed valid; do not perform substring matching.""",
        tests="""
assert glob_full_match("adceb", "*a*b") is True
assert glob_full_match("", "") is True
assert glob_full_match("", "**") is True
assert glob_full_match("", "?*") is False
assert glob_full_match("abc", "a?") is False
assert glob_full_match("acdcb", "a*c?b") is False
assert glob_full_match("ab", "*?*?*") is True
""",
    ),
    dict(
        id='max_product_slice', tier=2, intaked=True, tag='dp',
        title='Maximum-product slice with precise ties',
        prompt="""Implement max_product_slice(nums). For a nonempty list of integers, return (product, start, end) describing a nonempty contiguous slice with the largest possible product; start is inclusive and end is exclusive, using zero-based indices. Among slices with equal products, choose the shortest slice, then the one with the smallest start index. Negative numbers, zeros, and repeated values are allowed. Return None for an empty list. Inputs are guaranteed valid with at most 1000 elements. Do not modify the input. Use one O(n) scan and O(1) auxiliary storage, treating integer arithmetic as a single operation.""",
        tests="""
assert max_product_slice([2, 3, -2, 4]) == (6, 0, 2)
assert max_product_slice([]) is None
assert max_product_slice([-2, 3, -4]) == (24, 0, 3)
assert max_product_slice([0, -2, 0]) == (0, 0, 1)
assert max_product_slice([1, 2, 1]) == (2, 1, 2)
assert max_product_slice([-1, -1, -1]) == (1, 0, 2)
assert max_product_slice([-5]) == (-5, 0, 1)
""",
    ),
    dict(
        id='largest_ones_square', tier=2, intaked=True, tag='matrix',
        title='Largest solid square with a location',
        prompt="""Implement largest_ones_square(grid). The input is a list of rows containing only integers 0 and 1, with at most 1000 rows and at most 1000 total entries. Return (side, top, left) for the largest axis-aligned square containing only 1s; side is its side length and top, left are its zero-based top-left coordinates. Among equally large squares choose the smallest top coordinate, then the smallest left coordinate. If there are no 1s, including an empty grid or rows of width zero, return (0, -1, -1). Raise ValueError if row lengths differ. Other input types and values are guaranteed valid. Do not modify grid.""",
        tests="""
assert largest_ones_square([[1, 0, 1, 1, 1], [0, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]]) == (3, 0, 2)
assert largest_ones_square([]) == (0, -1, -1)
assert largest_ones_square([[], []]) == (0, -1, -1)
assert largest_ones_square([[0, 0], [0, 0]]) == (0, -1, -1)
assert largest_ones_square([[1, 1, 0, 1, 1], [1, 1, 0, 1, 1]]) == (2, 0, 0)
assert largest_ones_square([[0, 1], [1, 1], [1, 1]]) == (2, 1, 0)
def rejects_ragged():
    try: largest_ones_square([[], [1]])
    except ValueError: return True
    return False
assert rejects_ragged()
""",
    ),
    dict(
        id='next_greater_steps', tier=2, intaked=True, tag='lists',
        title='Circular distances to strictly greater values',
        prompt="""Implement next_greater_steps(nums) for a list of at most 1000 integers. For each index i, find the smallest distance d satisfying 1 <= d < len(nums) and nums[(i + d) % len(nums)] > nums[i]. Return these distances as a list in index order, using 0 whenever no such distance exists. Equal values do not qualify. Thus a single-element list returns [0], and an empty list returns []. Inputs are guaranteed valid. Do not modify nums. Use O(n) time and O(n) auxiliary space.""",
        tests="""
assert next_greater_steps([2, 1, 2, 4, 3]) == [3, 1, 1, 0, 4]
assert next_greater_steps([]) == []
assert next_greater_steps([7]) == [0]
assert next_greater_steps([5, 5, 5]) == [0, 0, 0]
assert next_greater_steps([5, 4, 3, 2, 1]) == [0, 4, 3, 2, 1]
assert next_greater_steps([-2, -1, -2]) == [1, 0, 2]
assert next_greater_steps([1, 3, 1, 3]) == [1, 0, 1, 0]
""",
    ),
    dict(
        id='repeating_decimal', tier=2, intaked=True, tag='math',
        title='Exact decimal expansion with repeating digits',
        prompt="""Implement repeating_decimal(numerator, denominator) for integers whose absolute values are at most 1000. Return the exact base-ten representation of their quotient as a string. Use a leading '-' exactly when the nonzero quotient is negative; do not use '+'. Zero is '0'. An integer quotient has no decimal point. A terminating fractional expansion has no parentheses or trailing fractional zeros. Otherwise, write the shortest nonrepeating fractional prefix followed by the shortest repeating block in parentheses, preserving all necessary zero digits; for example, 1/6 becomes '0.1(6)' and 1/90 becomes '0.0(1)'. The integer part has no unnecessary leading zeros. Raise ZeroDivisionError if the denominator is zero. Other inputs are guaranteed valid; use exact integer arithmetic.""",
        tests="""
assert repeating_decimal(1, 6) == "0.1(6)"
assert repeating_decimal(1, 90) == "0.0(1)"
assert repeating_decimal(-22, 7) == "-3.(142857)"
assert repeating_decimal(-50, -8) == "6.25"
assert repeating_decimal(0, -7) == "0"
assert repeating_decimal(4, 2) == "2"
def rejects_zero_denominator():
    try: repeating_decimal(1, 0)
    except ZeroDivisionError: return True
    return False
assert rejects_zero_denominator()
""",
    ),
    dict(
        id='directed_euler_walk', tier=3, intaked=True, tag='graphs',
        title='A walk using every directed edge',
        prompt="""Implement directed_euler_walk(n, edges). Vertices are the integers 0 through n-1, and each (u, v) in edges is a directed edge from u to v. Parallel edges and self-loops are allowed, and each occurrence is a separate edge. Return any list of vertices describing one continuous walk that traverses every edge occurrence exactly once, or None if impossible. The walk may revisit vertices. For nonempty edges it must contain len(edges)+1 vertices; isolated vertices need not occur. Any valid start and edge order are acceptable. If edges is empty, return [0] when n>0 and [] when n=0. Inputs are guaranteed valid: 0<=n<=1000, at most 1000 edges, and every endpoint is in range. Do not modify edges.""",
        tests="""
def valid_euler_walk(walk, n, edges):
    from collections import Counter
    if not edges: return walk == ([0] if n else [])
    if not isinstance(walk, list) or len(walk) != len(edges) + 1: return False
    if any(not isinstance(v, int) or not 0 <= v < n for v in walk): return False
    return Counter(zip(walk, walk[1:])) == Counter(map(tuple, edges))
assert valid_euler_walk(directed_euler_walk(5, [(0, 1), (1, 2), (2, 0), (0, 3)]), 5, [(0, 1), (1, 2), (2, 0), (0, 3)])
assert directed_euler_walk(0, []) == []
assert directed_euler_walk(3, []) == [0]
assert valid_euler_walk(directed_euler_walk(3, [(1, 1), (1, 2), (1, 2), (2, 1), (2, 1)]), 3, [(1, 1), (1, 2), (1, 2), (2, 1), (2, 1)])
assert directed_euler_walk(3, [(0, 1), (0, 2)]) is None
assert directed_euler_walk(4, [(0, 1), (1, 0), (2, 3), (3, 2)]) is None
assert directed_euler_walk(4, [(0, 1), (2, 3), (3, 2)]) is None
""",
    ),
    dict(
        id='min_ragged_wrap', tier=3, intaked=True, tag='dp',
        title='Optimal word wrapping with earliest breaks',
        prompt="""Implement min_ragged_wrap(words, width), returning a list of strings that wraps the words without changing their order. Each output line contains consecutive words separated by exactly one ASCII space, with no leading or trailing spaces, and has length at most width. The cost is the sum of (width - len(line)) ** 2 over every line except the final line, whose cost is always zero. Minimize this cost. On equal costs, choose the lexicographically smallest list of 1-based indices of the last words on the lines; equivalently, prefer the earliest first differing line break. Return [] for an empty word list when width is positive. Raise ValueError if width is not positive or any word is longer than width, including checking width for an empty list. Input types are guaranteed: words is a list of nonempty lowercase ASCII strings, width is an integer, and there are at most 1000 words and 1000 total letters. Do not modify words.""",
        tests="""
def bad_wrap(words, width):
    try: min_ragged_wrap(words, width)
    except ValueError: return True
    return False
assert min_ragged_wrap(['aaa', 'bb', 'cc', 'ddddd'], 6) == ['aaa', 'bb cc', 'ddddd']
assert min_ragged_wrap([], 4) == []
assert min_ragged_wrap(['aa', 'b', 'cc', 'ddd'], 4) == ['aa', 'b cc', 'ddd']
assert min_ragged_wrap(['aaa', 'bb', 'cc'], 6) == ['aaa bb', 'cc']
assert min_ragged_wrap(['ab', 'cd'], 5) == ['ab cd']
assert min_ragged_wrap(['a', 'b', 'c'], 1) == ['a', 'b', 'c']
assert bad_wrap(['abc'], 2) and bad_wrap([], 0)
""",
    ),
    dict(
        id='deadline_dispatch', tier=3, intaked=True, tag='data-structures',
        title='Deadline queue with cancellation and replacement',
        prompt="""Implement deadline_dispatch(events), processing event tuples from left to right. ('set', task, deadline) creates or replaces that task's pending schedule. ('cancel', task) removes its pending schedule if present, otherwise does nothing. ('run', threshold) executes and removes every currently pending task whose deadline is at most threshold, returning its task ID in that run's result. Execution order is increasing deadline, with ties ordered by the positions of the tasks' most recent 'set' events, earlier first. Replacing a task resets its tie priority even if the deadline stays the same. Return a list containing one result list for each 'run', including empty results; with no runs return []. Tasks remaining pending at the end produce no output. An executed or cancelled ID may be scheduled again. Run thresholds need not increase: a run examines only tasks currently pending, and earlier runs never consume newly scheduled tasks. Raise ValueError for an unknown operation, an incorrect tuple length, or a negative deadline or threshold. Input is a list of at most 1000 tuples. Nonempty tuples start with a string; for correctly shaped recognized operations, task IDs are nonempty lowercase ASCII strings and timestamps are integers. Do not modify events. Target O(m log m) time and O(m) space for m events.""",
        tests="""
def bad_events(events):
    try: deadline_dispatch(events)
    except ValueError: return True
    return False
assert deadline_dispatch([('set', 'b', 4), ('set', 'a', 4), ('set', 'c', 2), ('run', 3), ('run', 4)]) == [['c'], ['b', 'a']]
assert deadline_dispatch([]) == [] and deadline_dispatch([('set', 'a', 3)]) == []
assert deadline_dispatch([('cancel', 'ghost'), ('run', 0), ('run', 0)]) == [[], []]
assert deadline_dispatch([('set', 'a', 2), ('set', 'a', 5), ('set', 'b', 3), ('run', 3), ('run', 5), ('run', 9)]) == [['b'], ['a'], []]
assert deadline_dispatch([('set', 'a', 8), ('set', 'a', 2), ('set', 'b', 2), ('set', 'a', 2), ('set', 'c', 2), ('set', 'd', 2), ('cancel', 'c'), ('set', 'c', 2), ('run', 2), ('run', 8)]) == [['b', 'a', 'd', 'c'], []]
assert deadline_dispatch([('set', 'x', 0), ('run', 0), ('set', 'x', 0), ('run', 0), ('set', 'z', 9), ('run', 8), ('cancel', 'z'), ('run', 9), ('set', 'x', 1), ('run', 1)]) == [['x'], ['x'], [], [], ['x']]
assert all(bad_events(events) for events in [[('set', 'a', -1)], [('run', -1)], [('noop',)], [('cancel',)], [()], [('set', 'a')], [('run', 1, 2)]])
""",
    ),

    dict(
        id='expand_counted_pattern', tier=3, intaked=True, tag='parsing',
        title='Nested repetition with literal escapes',
        prompt="""Implement expand_counted_pattern(s). Parse a sequence of literals and repeats of the form count[sequence], allowing nested repeats. A count is one or more ASCII decimal digits representing a nonnegative integer; leading zeros are allowed. Backslash quotes exactly the following character, so an escaped digit or bracket is literal and two backslashes produce one. Every unescaped character other than a digit, bracket, or backslash is a literal, including spaces. An unescaped digit run must be immediately followed by '['; a bare '[' is invalid. Empty sequences and zero repetitions are allowed, and empty input returns ''. Raise ValueError for any malformed syntax, including unmatched brackets and a trailing backslash. Syntax inside a zero repetition must still be validated. Inputs are ASCII strings of length at most 1000; digit runs have values at most 1000, and every expanded nested sequence and the final output of any syntactically valid input have length at most 1000.""",
        tests="""
assert expand_counted_pattern("2[a3[b]]") == "abbbabbb"
assert expand_counted_pattern("") == ""
assert expand_counted_pattern("0[abc]2[]x") == "x"
assert expand_counted_pattern(r"2[a\\]b]\\3\\\\") == "a]ba]b3\\\\"
assert expand_counted_pattern("02[a] 1[b]") == "aa b"
assert expand_counted_pattern("10[z]") == "zzzzzzzzzz"
def rejects_pattern(s):
    try: expand_counted_pattern(s)
    except ValueError: return True
    return False
assert all(rejects_pattern(s) for s in ("[a]", "2a", "1[a", "a]", "\\\\", "0[2a]"))
""",
    ),

    dict(
        id='min_capacity_for_days', tier=2, intaked=True, tag='math',
        title='Minimum shipping capacity',
        prompt="""Implement min_capacity_for_days(weights, days). The nonempty list weights contains at most 1000 positive integers, and days is between 1 and len(weights). Items must be shipped in their given order, and each day ships one nonempty contiguous group whose total weight does not exceed the capacity. Return the minimum integer capacity that ships every item within days days.""",
        tests="""
assert min_capacity_for_days([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5) == 15
assert min_capacity_for_days([3, 2, 2, 4, 1, 4], 3) == 6
assert min_capacity_for_days([7], 1) == 7
assert min_capacity_for_days([2, 2, 2], 3) == 2
assert min_capacity_for_days([5, 1, 5], 2) == 6
assert min_capacity_for_days([1, 1, 1, 1], 2) == 2
""",
    ),
    dict(
        id='smallest_fixed_digit_sum', tier=2, intaked=True, tag='math',
        title='Smallest integer with fixed length and digit sum',
        prompt="""Implement smallest_fixed_digit_sum(d, s). Return the smallest nonnegative integer whose decimal representation has exactly d digits and whose digits sum to s, or -1 if no such integer exists. Here 1 <= d <= 1000 and s >= 0. Except for the one-digit integer 0, leading zeros are forbidden. Thus d=1 and s=0 returns 0, while d>1 and s=0 returns -1.""",
        tests="""
assert smallest_fixed_digit_sum(3, 15) == 159
assert smallest_fixed_digit_sum(2, 10) == 19
assert smallest_fixed_digit_sum(1, 0) == 0
assert smallest_fixed_digit_sum(2, 0) == -1
assert smallest_fixed_digit_sum(3, 28) == -1
assert smallest_fixed_digit_sum(4, 10) == 1009
""",
    ),
    dict(
        id='assignment_min_cost_bitmask', tier=3, intaked=True, tag='dp',
        title='Minimum-cost square assignment',
        prompt="""Implement assignment_min_cost_bitmask(cost), where cost is an n by n matrix, 0 <= n <= 8, and every entry is an integer from 0 through 1000000. Assign each row to a distinct column, using every column exactly once, so that the sum of the selected entries is minimized. Return only that minimum sum. The empty 0 by 0 matrix returns 0.""",
        tests="""
assert assignment_min_cost_bitmask([[9, 2, 7, 8], [6, 4, 3, 7], [5, 8, 1, 8], [7, 6, 9, 4]]) == 13
assert assignment_min_cost_bitmask([[4, 1, 3], [2, 0, 5], [3, 2, 2]]) == 5
assert assignment_min_cost_bitmask([]) == 0
assert assignment_min_cost_bitmask([[7]]) == 7
assert assignment_min_cost_bitmask([[5, 1], [1, 5]]) == 2
assert assignment_min_cost_bitmask([[0, 0], [0, 0]]) == 0
""",
    ),
    dict(
        id='grid_keys_shortest_path', tier=3, intaked=True, tag='graphs',
        title='Shortest grid path with keys and doors',
        prompt="""Implement grid_keys_shortest_path(grid). The nonempty rectangular grid contains exactly one S and one T; other cells are . open space, # wall, keys a through c, or doors A through C. Moving orthogonally to an adjacent cell costs 1. Stepping on a key collects it permanently, and each key opens every door of the corresponding uppercase letter. A door cannot be entered before its key is collected. Return the minimum number of moves from S to T, or -1 if T is unreachable. The grid has at most 1000 cells.""",
        tests="""
assert grid_keys_shortest_path(["S.a.A.T"]) == 6
assert grid_keys_shortest_path(["ST"]) == 1
assert grid_keys_shortest_path(["SAT"]) == -1
assert grid_keys_shortest_path(["SAT", "a.."]) == 4
assert grid_keys_shortest_path(["SaAAT"]) == 4
assert grid_keys_shortest_path(["S#", "#T"]) == -1
""",
    ),
    dict(
        id='count_grid_independent_sets', tier=3, intaked=True, tag='dp',
        title='Count independent sets on a grid',
        prompt="""Implement count_grid_independent_sets(rows, cols), where 0 <= rows, cols <= 8. Consider the rows by cols rectangular grid graph, whose edges join cells sharing a side. Return the number of subsets of cells containing no edge's two endpoints, modulo 1000000007. Different subsets are counted separately, and the empty subset counts. If either dimension is 0, there are no cells and the answer is 1.""",
        tests="""
assert count_grid_independent_sets(2, 2) == 7
assert count_grid_independent_sets(1, 1) == 2
assert count_grid_independent_sets(1, 3) == 5
assert count_grid_independent_sets(0, 4) == 1
assert count_grid_independent_sets(3, 0) == 1
assert count_grid_independent_sets(2, 3) == 17
""",
    ),
    dict(
        id='first_winning_subtraction_move', tier=3, intaked=True, tag='algorithms',
        title='First winning move in a subtraction game',
        prompt="""Implement first_winning_subtraction_move(stones, moves), where 0 <= stones <= 1000 and moves is a nonempty list of distinct positive integers. Players alternate removing exactly one listed number of stones, provided that many remain; a player unable to move loses. Assuming optimal play, return the smallest first move that guarantees a win for the initial player. Return -1 if the initial position is losing, including when stones is 0. The input order of moves does not affect the answer.""",
        tests="""
assert first_winning_subtraction_move(10, [1, 3, 4]) == 1
assert first_winning_subtraction_move(7, [1, 3, 4]) == -1
assert first_winning_subtraction_move(2, [1, 3, 4]) == -1
assert first_winning_subtraction_move(3, [4, 1, 3]) == 1
assert first_winning_subtraction_move(0, [1, 3, 4]) == -1
assert first_winning_subtraction_move(4, [2]) == -1
assert first_winning_subtraction_move(6, [2]) == 2
""",
    ),
    dict(
        id='minimum_max_lateness', tier=3, intaked=True, tag='algorithms',
        title='Minimize maximum job lateness',
        prompt="""Implement minimum_max_lateness(jobs). Each of at most 1000 jobs is a (duration, deadline) pair with positive integer duration and an integer deadline. All jobs are available at time 0 and must run without overlap on one machine; idle time is allowed. A job finishing at time t has lateness t-deadline, which may be negative. Return the minimum possible value of the maximum lateness over all jobs. For an empty job list, return 0.""",
        tests="""
assert minimum_max_lateness([(3, 6), (2, 8), (1, 9), (4, 9), (3, 14), (2, 15)]) == 1
assert minimum_max_lateness([]) == 0
assert minimum_max_lateness([(5, 3)]) == 2
assert minimum_max_lateness([(1, 10), (1, 20)]) == -9
assert minimum_max_lateness([(4, 100), (1, 1)]) == 0
assert minimum_max_lateness([(2, 5), (3, 5)]) == 0
""",
    ),
    dict(
        id='repeating_positional_cycle_length', tier=3, intaked=True, tag='math',
        title='Cycle length of a positional fraction',
        prompt="""Implement repeating_positional_cycle_length(numerator, denominator, base), with 0 <= numerator <= 1000, 1 <= denominator <= 1000, and 2 <= base <= 36. Write the rational numerator/denominator in the given positional base. Return the length of the repeating cycle in its fractional part after any nonrepeating prefix. Return 0 if the expansion terminates or the numerator is 0. The integer part does not affect the answer.""",
        tests="""
assert repeating_positional_cycle_length(1, 6, 10) == 1
assert repeating_positional_cycle_length(1, 7, 10) == 6
assert repeating_positional_cycle_length(1, 8, 10) == 0
assert repeating_positional_cycle_length(1, 3, 2) == 2
assert repeating_positional_cycle_length(2, 4, 10) == 0
assert repeating_positional_cycle_length(5, 5, 10) == 0
assert repeating_positional_cycle_length(1, 12, 10) == 1
""",
    ),

    dict(
        id='min_swaps_to_sorted', tier=2, intaked=True, tag='lists',
        title='Swaps to sort a permutation',
        prompt="""Implement min_swaps_to_sorted(nums), where nums is a list of distinct integers with length at most 1000. In one swap, any two positions may be exchanged. Return the minimum number of swaps needed to sort the list in ascending order. The empty permutation requires 0 swaps.""",
        tests="""
assert min_swaps_to_sorted([4, 3, 2, 1]) == 2
assert min_swaps_to_sorted([1, 0, 2, 3, 4]) == 1
assert min_swaps_to_sorted([]) == 0
assert min_swaps_to_sorted([0]) == 0
assert min_swaps_to_sorted([2, 0, 1]) == 2
assert min_swaps_to_sorted([1, 2, 0, 4, 3]) == 3
""",
    ),
]
