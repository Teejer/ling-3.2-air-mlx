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
]
