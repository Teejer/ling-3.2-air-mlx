"""Reference solutions for tier-2 problems (used only by validate_problems.py
to self-test the asserts — never shown to any model)."""
import re


def next_perm(nums):
    a = nums[:]
    i = len(a) - 2
    while i >= 0 and a[i] >= a[i + 1]:
        i -= 1
    if i < 0:
        return sorted(a)
    j = len(a) - 1
    while a[j] <= a[i]:
        j -= 1
    a[i], a[j] = a[j], a[i]
    a[i + 1:] = reversed(a[i + 1:])
    return a


def word_break(s, words):
    w = set(words)
    ok = [True]
    for i in range(1, len(s) + 1):
        ok.append(any(ok[j] and s[j:i] in w for j in range(i)))
    return ok[len(s)]


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[len(b)]


def largest_rectangle(heights):
    best, stack = 0, []  # stack of indices, increasing heights
    for i, h in enumerate(heights + [0]):
        start = i
        while stack and stack[-1][1] > h:
            j, hj = stack.pop()
            best = max(best, hj * (i - j))
            start = j
        stack.append((start, h))
    return best


def course_order(n, prereqs):
    from collections import deque
    adj, indeg = [[] for _ in range(n)], [0] * n
    for c, p in prereqs:
        adj[p].append(c)
        indeg[c] += 1
    q = deque(i for i in range(n) if indeg[i] == 0)
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order if len(order) == n else []


class RunningMedian:
    def __init__(self):
        self.vals = []

    def add(self, x):
        import bisect
        bisect.insort(self.vals, x)

    def median(self):
        n = len(self.vals)
        return float(self.vals[n // 2]) if n % 2 else \
            (self.vals[n // 2 - 1] + self.vals[n // 2]) / 2


def coin_change(coins, amount):
    dp = [0] + [float("inf")] * amount
    for v in range(1, amount + 1):
        dp[v] = min((dp[v - c] + 1 for c in coins if c <= v and dp[v - c] < float("inf")),
                    default=float("inf"))
    return dp[amount] if dp[amount] < float("inf") else -1


def is_balanced_code(s):
    pairs, stack, quote, i = {")": "(", "]": "[", "}": "{"}, [], None, 0
    while i < len(s):
        c = s[i]
        i += 1
        if quote:
            if c == "\\":
                i += 1  # escaped char
            elif c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c in "([{":
            stack.append(c)
        elif c in ")]}":
            if not stack or stack.pop() != pairs[c]:
                return False
    return not stack


def eval_expr(s):
    s = s.replace(" ", "")
    pos = 0

    def expr():
        nonlocal pos
        v = term()
        while pos < len(s) and s[pos] in "+-":
            op = s[pos]
            pos += 1
            r = term()
            v = v + r if op == "+" else v - r
        return v

    def term():
        nonlocal pos
        v = atom()
        while pos < len(s) and s[pos] in "*/":
            op = s[pos]
            pos += 1
            r = atom()
            if op == "*":
                v *= r
            else:
                q = abs(v) // abs(r)
                v = q if (v >= 0) == (r >= 0) else -q
        return v

    def atom():
        nonlocal pos
        if s[pos] == "(":
            pos += 1
            v = expr()
            pos += 1
            return v
        j = pos
        while j < len(s) and s[j].isdigit():
            j += 1
        v = int(s[pos:j])
        pos = j
        return v

    return expr()


def longest_consecutive(nums):
    nums, best = set(nums), 0
    for n in nums:
        if n - 1 not in nums:
            k = n
            while k + 1 in nums:
                k += 1
            best = max(best, k - n + 1)
    return best


def parse_duration(s):
    if s == "":
        return 0
    m = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", s)
    if not m or not any(m.groups()):
        raise ValueError(f"bad duration: {s!r}")
    h, mi, sec = (int(g) if g else 0 for g in m.groups())
    return h * 3600 + mi * 60 + sec


def restore_ips(s):
    out = []

    def rec(start, cur):
        if len(cur) == 4:
            if start == len(s):
                out.append(".".join(cur))
            return
        for ln in range(1, 4):
            seg = s[start:start + ln]
            if len(seg) < ln:
                break
            if len(seg) > 1 and seg[0] == "0":
                break
            if int(seg) > 255:
                continue
            rec(start + ln, cur + [seg])

    rec(0, [])
    return sorted(out)


def spiral_order(m):
    out = []
    top, bot, left, right = 0, len(m) - 1, 0, len(m[0]) - 1
    while top <= bot and left <= right:
        for c in range(left, right + 1):
            out.append(m[top][c])
        top += 1
        for r in range(top, bot + 1):
            out.append(m[r][right])
        right -= 1
        if top <= bot:
            for c in range(right, left - 1, -1):
                out.append(m[bot][c])
            bot -= 1
        if left <= right:
            for r in range(bot, top - 1, -1):
                out.append(m[r][left])
            left += 1
    return out


def min_window_len(s, t):
    from collections import Counter
    need = Counter(t)
    missing = len(t)
    best, i = len(s) + 1, 0
    for j, c in enumerate(s):
        if need[c] > 0:
            missing -= 1
        need[c] -= 1
        while missing == 0:
            best = min(best, j - i + 1)
            need[s[i]] += 1
            if need[s[i]] > 0:
                missing += 1
            i += 1
    return best if best <= len(s) else -1
