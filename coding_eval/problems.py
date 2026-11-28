"""Original coding problems with executable asserts (stdlib-only, deterministic).

Each problem: id, title, tag, prompt (what the model sees), tests (executed
against the model's solution module `sol`).
"""

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
]
