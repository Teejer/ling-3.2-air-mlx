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


# --- interval_overlap_count (intake) ---
def interval_overlap_count(intervals):
    events = []
    for a, b in intervals:
        if a > b:
            raise ValueError
        events.append((a, 1))
        events.append((b, -1))
        events.sort(key=lambda x: (x[0], -x[1]))
    active = best = 0
    for _, delta in events:
        active += delta
        best = max(best, active)
    return best

# --- canonical_fraction (intake) ---
def canonical_fraction(n, d):
    import math
    if d == 0:
        raise ZeroDivisionError
        if n == 0:
            return [0, 1]
    g = math.gcd(n, d)
    n //= g
    d //= g
    if d < 0:
        n, d = -n, -d
    return [n, d]

# --- deduplicate_stable (intake) ---
def deduplicate_stable(items):
    seen = set()
    result = []
    for x in items:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result

# --- lis_reconstruct (intake) ---
def lis_reconstruct(nums):
    n = len(nums)
    if not n:
        return []
    dp = [1] * n
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            if nums[j] > nums[i]:
                dp[i] = max(dp[i], 1 + dp[j])
    remaining = max(dp)
    result = []
    start = 0
    last = None
    while remaining:
        candidates = [i for i in range(start, n) if dp[i] >= remaining and (last is None or nums[i] > last)]
        best_value = min(nums[i] for i in candidates)
        chosen = next(i for i in candidates if nums[i] == best_value)
        result.append(nums[chosen])
        last = nums[chosen]
        start = chosen + 1
        remaining -= 1
    return result


# --- longest_k_distinct (intake) ---
def longest_k_distinct(s, k):
    if k <= 0:
        return 0
    cnt = {}
    l = 0
    best = 0
    for r, c in enumerate(s):
        cnt[c] = cnt.get(c, 0) + 1
        while len(cnt) > k:
            cnt[s[l]] -= 1
            if cnt[s[l]] == 0:
                del cnt[s[l]]
            l += 1
        best = max(best, r - l + 1)
    return best

# --- egyptian_greedy (intake) ---
def egyptian(num, den):
    from math import gcd
    if den <= 0 or num < 0 or num >= den:
        raise ValueError('invalid')
    out = []
    while num > 0:
        d = -(-den // num)
        out.append(d)
        num, den = num * d - den, den * d
        if num:
            g = gcd(num, den)
            num //= g
            den //= g
    return out

# --- combo_sum_once (intake) ---
def combo_sum_once(nums, target):
    nums = sorted(nums)
    out = []
    def go(i, rem, cur):
        if rem == 0:
            out.append(cur[:])
            return
        for j in range(i, len(nums)):
            if j > i and nums[j] == nums[j - 1]:
                continue
            if nums[j] > rem:
                break
            cur.append(nums[j])
            go(j + 1, rem - nums[j], cur)
            cur.pop()
    go(0, target, [])
    return out

# --- set_zeroes (intake) ---
def set_zeroes(matrix):
 rows = {i for i, r in enumerate(matrix) for v in r if v == 0}
 cols = {j for r in matrix for j, v in enumerate(r) if v == 0}
 return [[0 if i in rows or j in cols else v for j, v in enumerate(r)] for i, r in enumerate(matrix)]

# --- kth_largest_stream (intake) ---
def kth_largest_stream(k, nums):
    import heapq
    if k <= 0:
        raise ValueError('k must be positive')
    h = []
    out = []
    for x in nums:
        if len(h) < k:
            heapq.heappush(h, x)
        elif x > h[0]:
            heapq.heapreplace(h, x)
        out.append(h[0] if len(h) == k else None)
    return out

# --- lcs_string (intake) ---
def lcs(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    i = j = 0
    out = []
    while i < n and j < m:
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return ''.join(out)

# --- assign_rooms (intake) ---
def assign_rooms(meetings):
    import heapq
    order = sorted(range(len(meetings)), key=lambda i: (meetings[i][0], i))
    free = []
    busy = []
    nxt = 0
    res = [0] * len(meetings)
    for i in order:
        s, e = meetings[i]
        while busy and busy[0][0] <= s:
            heapq.heappush(free, heapq.heappop(busy)[1])
        if free:
            r = heapq.heappop(free)
        else:
            r = nxt
            nxt += 1
        res[i] = r
        heapq.heappush(busy, (e, r))
    return res

# --- bipartition_coloring (intake) ---
def bipartition(n, edges):
    from collections import deque
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    col = [-1] * n
    for s in range(n):
        if col[s] != -1:
            continue
        col[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if col[v] == -1:
                    col[v] = 1 - col[u]
                    q.append(v)
                elif col[v] == col[u]:
                    return None
    return col


# --- decode_escaped_fields (intake) ---
def decode_escaped_fields(s):
    fields = []
    current = []
    escaped = False
    for ch in s:
        if escaped:
            current.append(ch)
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == ",":
            fields.append("".join(current))
            current = []
        else:
            current.append(ch)
    if escaped:
        raise ValueError
    fields.append("".join(current))
    return fields


# --- shortest_cycle_length (intake) ---
def shortest_cycle_length(n, edges):
    from collections import deque
    if any(not (0 <= u < n and 0 <= v < n) for u, v in edges): raise ValueError
    if any(u == v for u, v in edges): return 1
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    best = float('inf')
    for root in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[root] = 0
        q = deque([root])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif parent[u] != v:
                    best = min(best, dist[u] + dist[v] + 1)
    return -1 if best == float('inf') else best


# --- parse_kv_escaped (intake) ---
def parse_kv(s):
    BS = chr(92)
    out = {}
    key, val, in_val, seen = [], [], False, False

    def flush():
        nonlocal key, val, in_val, seen
        if seen:
            out[''.join(key)] = ''.join(val)
        key, val, in_val, seen = [], [], False, False

    i = 0
    while i < len(s):
        c = s[i]
        if c == BS:
            if i + 1 >= len(s):
                raise ValueError('dangling escape')
            (val if in_val else key).append(s[i + 1])
            seen = True
            i += 2
            continue
        if c == ';':
            flush()
        elif c == '=' and not in_val:
            in_val = True
            seen = True
        else:
            (val if in_val else key).append(c)
            seen = True
        i += 1
    flush()
    return out
