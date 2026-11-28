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


# --- longest_bounded_window (intake) ---
def longest_bounded_window(nums, limit):
    if limit < 0: return 0
    from collections import deque
    mx, mn, left, best = deque(), deque(), 0, 0
    for right, x in enumerate(nums):
        while mx and nums[mx[-1]] <= x: mx.pop()
        mx.append(right)
        while mn and nums[mn[-1]] >= x: mn.pop()
        mn.append(right)
        while nums[mx[0]] - nums[mn[0]] > limit:
            left += 1
            if mx[0] < left: mx.popleft()
            if mn[0] < left: mn.popleft()
        best = max(best, right - left + 1)
    return best

# --- matrix_diagonal_sums (intake) ---
def matrix_diagonal_sums(matrix):
    n = len(matrix)
    if n == 0: return [0, 0]
    for row in matrix:
        if len(row) != n: raise ValueError("not square")
    return [sum(matrix[i][i] for i in range(n)), sum(matrix[i][n - 1 - i] for i in range(n))]

# --- normalize_identifier (intake) ---
def normalize_identifier(s):
    import re
    return re.sub(" +", "_", s.strip(" ").lower())


# --- streaming_kth_largest (intake) ---
def streaming_kth_largest(nums, k):
    if k < 1: raise ValueError("k must be >= 1")
    import heapq
    heap, out = [], []
    for x in nums:
        heapq.heappush(heap, x)
        if len(heap) > k: heapq.heappop(heap)
        if len(heap) == k: out.append(heap[0])
    return out


# --- fraction_add (intake) ---
def fraction_add(a, b):
    from math import gcd
    n1, d1 = a
    n2, d2 = b
    if d1 == 0 or d2 == 0:
        raise ValueError("zero denominator")
    n = n1 * d2 + n2 * d1
    d = d1 * d2
    if d < 0:
        n, d = -n, -d
    g = gcd(abs(n), d)
    return n // g, d // g

# --- stable_frequency_sort (intake) ---
def stable_frequency_sort(nums):
    counts = {}
    first = {}
    for i, x in enumerate(nums):
        counts[x] = counts.get(x, 0) + 1
        if x not in first:
            first[x] = i
    result = []
    for x in sorted(counts, key=lambda x: (-counts[x], first[x])):
        result.extend([x] * counts[x])
    return result

# --- merge_sorted_streams (intake) ---
def merge_sorted_streams(streams):
    import heapq
    heap = []
    for i, stream in enumerate(streams):
        if stream:
            heapq.heappush(heap, (stream[0], i, 0))
    result = []
    while heap:
        value, i, j = heapq.heappop(heap)
        result.append(value)
        j += 1
        if j < len(streams[i]):
            heapq.heappush(heap, (streams[i][j], i, j))
    return result

# --- weighted_interval_schedule (intake) ---
def weighted_interval_schedule(intervals):
    from functools import cmp_to_key
    items = sorted([(e, s, i, w) for i, (s, e, w) in enumerate(intervals)])
    states = [(0, 0, ())]
    for e, s, i, w in items:
        compatible = [j for j, (end, _, _, _) in enumerate(items[:len(states)-1]) if end <= s]
        base = states[max(compatible) + 1] if compatible else states[0]
        candidate = (base[0] + w, base[1] + 1, base[2] + (i,))
        previous = states[-1]
        if candidate[0] > previous[0] or (candidate[0] == previous[0] and (candidate[1], tuple(sorted(candidate[2]))) < (previous[1], tuple(sorted(previous[2])))):
            states.append(candidate)
        else:
            states.append(previous)
    chosen = list(states[-1][2])
    return sorted(chosen, key=lambda i: (intervals[i][0], intervals[i][1], i))

# --- shortest_path_with_count (intake) ---
def shortest_path_with_count(n, edges, source, target):
    from collections import deque
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    dist = [-1] * n
    count = [0] * n
    dist[source] = 0
    count[source] = 1
    q = deque([source])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                count[v] = count[u]
                q.append(v)
            elif dist[v] == dist[u] + 1:
                count[v] += count[u]
    return (dist[target], count[target]) if dist[target] != -1 else (-1, 0)

# --- normalize_path (intake) ---
def normalize_path(path):
    stack = []
    for part in path.split("/"):
        if part == "" or part == ".":
            continue
        if part == "..":
            if stack:
                stack.pop()
        else:
            stack.append(part)
    return "/" + "/".join(stack)


# --- longest_unique_window (intake) ---
def longest_unique_window(s):
    last = {}
    left = 0
    best_start = 0
    best_len = 0
    for i, ch in enumerate(s):
        if ch in last and last[ch] >= left:
            left = last[ch] + 1
        last[ch] = i
        length = i - left + 1
        if length > best_len:
            best_start, best_len = left, length
    return best_start, best_len


# --- count_coprime_inclusion_exclusion (intake) ---
def count_coprime(n, m):
    factors = set()
    t = m
    p = 2
    while p * p <= t:
        while t % p == 0:
            factors.add(p)
            t //= p
        p += 1
    if t > 1:
        factors.add(t)
    primes = sorted(factors)
    k = len(primes)
    total = 0
    for mask in range(1 << k):
        prod = 1
        bits = 0
        for i in range(k):
            if mask >> i & 1:
                prod *= primes[i]
                bits += 1
        total += (n // prod) * (1 if bits % 2 == 0 else -1)
    return total

# --- min_eating_speed (intake) ---
def min_eating_speed(piles, h):
    lo, hi = 1, max(piles)
    while lo < hi:
        mid = (lo + hi) // 2
        if sum((p + mid - 1) // mid for p in piles) <= h:
            hi = mid
        else:
            lo = mid + 1
    return lo

# --- tsp_exact_bitmask (intake) ---
def tsp_length(dist):
    n = len(dist)
    FULL = 1 << n
    INF = float('inf')
    dp = [[INF] * n for _ in range(FULL)]
    dp[1][0] = 0
    for mask in range(FULL):
        for u in range(n):
            if not mask >> u & 1 or dp[mask][u] == INF:
                continue
            for v in range(n):
                if mask >> v & 1:
                    continue
                m2 = mask | 1 << v
                c = dp[mask][u] + dist[u][v]
                if c < dp[m2][v]:
                    dp[m2][v] = c
    return min(dp[FULL - 1][v] + dist[v][0] for v in range(n))

# --- predict_the_winner (intake) ---
def predict_winner(nums):
    n = len(nums)
    dp = list(nums)
    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length - 1
            dp[i] = max(nums[i] - dp[i + 1], nums[j] - dp[i])
    return dp[0] >= 0

# --- base_neg2_conversion (intake) ---
def base_neg2(n):
    if n == 0:
        return "0"
    digits = []
    while n != 0:
        r = n & 1
        digits.append(str(r))
        n = (n - r) // -2
    return "".join(reversed(digits))

# --- min_path_effort_grid (intake) ---
def min_effort(heights):
    import heapq
    rows, cols = len(heights), len(heights[0])
    INF = float('inf')
    best = [[INF] * cols for _ in range(rows)]
    best[0][0] = 0
    heap = [(0, 0, 0)]
    while heap:
        d, r, c = heapq.heappop(heap)
        if (r, c) == (rows - 1, cols - 1):
            return d
        if d > best[r][c]:
            continue
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                nd = max(d, abs(heights[nr][nc] - heights[r][c]))
                if nd < best[nr][nc]:
                    best[nr][nc] = nd
                    heapq.heappush(heap, (nd, nr, nc))

# --- matrix_chain_multiplication (intake) ---
def matrix_chain(dims):
    n = len(dims) - 1
    dp = [[0] * n for _ in range(n)]
    for L in range(2, n + 1):
        for i in range(0, n - L + 1):
            j = i + L - 1
            dp[i][j] = min(dp[i][k] + dp[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1] for k in range(i, j))
    return dp[0][n - 1]

# --- subarrays_sum_equals_k (intake) ---
def count_subarrays_k(nums, k):
    from collections import defaultdict
    seen = defaultdict(int)
    seen[0] = 1
    s = 0
    cnt = 0
    for x in nums:
        s += x
        cnt += seen[s - k]
        seen[s] += 1
    return cnt


# --- min_station_refuels (intake) ---
def min_refuels(start_fuel, target, stations):
    import heapq
    heap = []
    fuel = start_fuel
    ans = 0
    i = 0
    pos = 0
    stations = sorted(stations)
    while fuel < target - pos:
        while i < len(stations) and stations[i][0] - pos <= fuel:
            heapq.heappush(heap, -stations[i][1])
            i += 1
        if not heap:
            return -1
        fuel += -heapq.heappop(heap)
        ans += 1
    return ans

# --- render_escaped_template (intake) ---
def render(tpl, env):
    out = []
    i = 0
    while i < len(tpl):
        c = tpl[i]
        if c == "~":
            if i + 1 >= len(tpl):
                raise ValueError("dangling escape")
            out.append(tpl[i + 1])
            i += 2
        elif c == "{" and i + 1 < len(tpl) and tpl[i + 1] == "{":
            j = tpl.find("}}", i + 2)
            if j < 0:
                raise ValueError("unclosed field")
            name = tpl[i + 2:j]
            if name not in env:
                raise KeyError(name)
            out.append(env[name])
            i = j + 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


# --- solve_linear_congruence (intake) ---
def solve_linear_congruence(a, b, m):
    from math import gcd
    g = gcd(a, m)
    if b % g: return -1
    mod = m // g
    a_reduced = (a // g) % mod
    b_reduced = (b // g) % mod
    def extended(x, y):
        if y == 0: return (1, 0)
        u, v = extended(y, x % y)
        return v, u - (x // y) * v
    inverse = extended(a_reduced, mod)[0] % mod
    return (b_reduced * inverse) % mod

# --- max_product_after_cut (intake) ---
def max_product_after_cut(n):
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = max(max(j, dp[j]) * max(i - j, dp[i - j]) for j in range(1, i))
    return dp[n]

# --- count_digit_occurrences (intake) ---
def count_digit_occurrences(n, d):
    total, factor = 0, 1
    while factor <= n:
        lower = n % factor
        current = (n // factor) % 10
        higher = n // (factor * 10)
        if current < d:
            total += higher * factor
        elif current == d:
            total += higher * factor + lower + 1
        else:
            total += (higher + 1) * factor
        factor *= 10
    return total

# --- count_divisible_subsets (intake) ---
def count_divisible_subsets(nums, m, target):
    mod = 1000000007
    dp = [0] * m
    dp[0] = 1
    for x in nums:
        nxt = dp[:]
        for r, count in enumerate(dp):
            destination = (r + x) % m
            nxt[destination] = (nxt[destination] + count) % mod
        dp = nxt
    return dp[target]

# --- min_assignment_cost (intake) ---
def min_assignment_cost(cost):
    n = len(cost)
    dp = [None] * (1 << n)
    dp[0] = 0
    for mask in range(1 << n):
        if dp[mask] is None: continue
        row = bin(mask).count("1")
        if row == n: continue
        for col in range(n):
            bit = 1 << col
            if mask & bit: continue
            new_mask = mask | bit
            value = dp[mask] + cost[row][col]
            if dp[new_mask] is None or value < dp[new_mask]:
                dp[new_mask] = value
    return dp[-1]

# --- minimize_largest_block_sum (intake) ---
def minimize_largest_block_sum(nums, k):
    def feasible(limit):
        groups, current = 1, 0
        for x in nums:
            if current + x > limit:
                groups += 1
                current = x
            else:
                current += x
        return groups <= k
    lo, hi = max(nums), sum(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

# --- count_n_queens (intake) ---
def count_n_queens(n):
    full = (1 << n) - 1
    def search(row, columns, down_left, down_right):
        if row == n: return 1
        free = full & ~(columns | down_left | down_right)
        total = 0
        while free:
            bit = free & -free
            free -= bit
            total += search(row + 1, columns | bit, ((down_left | bit) << 1) & full, (down_right | bit) >> 1)
        return total
    return search(0, 0, 0, 0)

# --- grid_wall_escape (intake) ---
def grid_wall_escape(grid, k):
    from collections import deque
    rows, cols = len(grid), len(grid[0])
    queue = deque([(0, 0, 0, 0)])
    seen = {(0, 0, 0)}
    while queue:
        r, c, used, distance = queue.popleft()
        if (r, c) == (rows - 1, cols - 1): return distance
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols): continue
            new_used = used + (grid[nr][nc] == "#")
            state = (nr, nc, new_used)
            if new_used <= k and state not in seen:
                seen.add(state)
                queue.append((nr, nc, new_used, distance + 1))
    return -1

# --- min_join_cost (intake) ---
def min_join_cost(lengths):
    import heapq
    heap = list(lengths)
    heapq.heapify(heap)
    total = 0
    while len(heap) > 1:
        cost = heapq.heappop(heap) + heapq.heappop(heap)
        total += cost
        heapq.heappush(heap, cost)
    return total

# --- count_domino_tilings (intake) ---
def count_domino_tilings(rows, cols):
    if rows * cols % 2: return 0
    board = [False] * (rows * cols)
    def search():
        try:
            start = board.index(False)
        except ValueError:
            return 1
        r, c = divmod(start, cols)
        board[start] = True
        total = 0
        if c + 1 < cols and not board[start + 1]:
            board[start + 1] = True
            total += search()
            board[start + 1] = False
        if r + 1 < rows and not board[start + cols]:
            board[start + cols] = True
            total += search()
            board[start + cols] = False
        board[start] = False
        return total
    return search()


# --- excel_column_title (intake) ---
def excel_column_title(n):
 if n < 1: raise ValueError("n must be >= 1")
 out = []
 while n: n, r = divmod(n - 1, 26); out.append(chr(65 + r))
 return "".join(reversed(out))

# --- trap_rain_water (intake) ---
def trap_rain_water(heights):
 from itertools import accumulate
 left = list(accumulate(heights, max))
 right = list(accumulate(reversed(heights), max))[::-1]
 return sum(min(a, b) - x for a, b, x in zip(left, right, heights))

# --- daily_temperatures (intake) ---
def daily_temperatures(temps):
    res, stack = [0] * len(temps), []
    for i, x in enumerate(temps):
        while stack and temps[stack[-1]] < x: j = stack.pop(); res[j] = i - j
        stack.append(i)
    return res

# --- word_pattern_bijection (intake) ---
def word_pattern_bijection(pattern, s):
 words = s.split()
 if len(words) != len(pattern): return False
 return len(set(pattern)) == len(set(words)) == len(set(zip(pattern, words)))

# --- count_islands (intake) ---
def count_islands(grid):
    land = {(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == 1}
    count = 0
    while land:
        count += 1; stack = [land.pop()]
        while stack:
            r, c = stack.pop()
            nbs = [p for p in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)) if p in land]
            land.difference_update(nbs); stack.extend(nbs)
    return count

# --- summary_ranges (intake) ---
def summary_ranges(nums):
    from itertools import groupby
    vals = sorted(set(nums))
    runs = [[x for _, x in g] for _, g in groupby(enumerate(vals), key=lambda p: p[1] - p[0])]
    return [str(r[0]) if len(r) == 1 else f"{r[0]}->{r[-1]}" for r in runs]

# --- knapsack_reconstruct (intake) ---
def knapsack_reconstruct(weights, values, capacity):
    n = len(weights)
    if n != len(values) or capacity < 0 or any(x < 0 for x in list(weights) + list(values)): raise ValueError("bad input")
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        w, v = weights[i - 1], values[i - 1]
        dp[i] = [max(dp[i - 1][c], dp[i - 1][c - w] + v) if w <= c else dp[i - 1][c] for c in range(capacity + 1)]
    chosen, c = [], capacity
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]: chosen.append(i - 1); c -= weights[i - 1]
    return dp[n][capacity], sorted(chosen)

# --- eulerian_trail (intake) ---
def eulerian_trail(edges):
    from collections import defaultdict
    if not edges: return []
    adj, diff = defaultdict(list), defaultdict(int)
    for u, v in edges: adj[u].append(v); diff[u] += 1; diff[v] -= 1
    starts = [x for x, d in diff.items() if d == 1]
    if any(abs(d) > 1 for d in diff.values()) or len(starts) > 1: return []
    for u in list(adj): adj[u].reverse()
    stack, path = [starts[0] if starts else edges[0][0]], []
    while stack:
        u = stack[-1]
        if adj[u]: stack.append(adj[u].pop())
        else: path.append(stack.pop())
    path.reverse()
    return path if len(path) == len(edges) + 1 else []


# --- decode_nested_repeat (intake) ---
def decode_nested_repeat(s):
    stack, cur, num, i = [], [], None, 0
    while i < len(s):
        ch = s[i]
        if ch == "\\":
            if num is not None or i + 1 == len(s): raise ValueError("bad escape")
            cur.append(s[i + 1]); i += 2; continue
        if ch in "0123456789": num = (num or 0) * 10 + int(ch)
        elif ch == "[":
            if num is None: raise ValueError("'[' without count")
            stack.append((cur, num)); cur, num = [], None
        elif num is not None: raise ValueError("count not followed by '['")
        elif ch == "]":
            if not stack: raise ValueError("unmatched ']'")
            prev, k = stack.pop(); prev.append("".join(cur) * k); cur = prev
        else: cur.append(ch)
        i += 1
    if stack or num is not None: raise ValueError("unclosed '['")
    return "".join(cur)


# --- shortest_threshold_slice (intake) ---
def shortest_threshold_slice(nums, threshold):
    from collections import deque
    prefixes = [0]
    for value in nums: prefixes.append(prefixes[-1] + value)
    candidates, best = deque(), len(nums) + 1
    for end, total in enumerate(prefixes):
        while candidates and total - prefixes[candidates[0]] >= threshold:
            best = min(best, end - candidates.popleft())
        while candidates and prefixes[candidates[-1]] >= total: candidates.pop()
        candidates.append(end)
    return best if best <= len(nums) else 0

# --- glob_full_match (intake) ---
def glob_full_match(text, pattern):
    previous = [True] + [False] * len(pattern)
    for j, token in enumerate(pattern, 1): previous[j] = previous[j - 1] and token == "*"
    for letter in text:
        current = [False] * (len(pattern) + 1)
        for j, token in enumerate(pattern, 1):
            current[j] = (current[j - 1] or previous[j]) if token == "*" else (previous[j - 1] and (token == "?" or token == letter))
        previous = current
    return previous[-1]

# --- max_product_slice (intake) ---
def max_product_slice(nums):
    if not nums: return None
    high = low = (nums[0], 0)
    best = (nums[0], 0, 1)
    for i in range(1, len(nums)):
        value = nums[i]
        candidates = [(value, i), (high[0] * value, high[1]), (low[0] * value, low[1])]
        high = max(candidates, key=lambda item: (item[0], item[1]))
        low = min(candidates, key=lambda item: (item[0], -item[1]))
        candidate = (high[0], high[1], i + 1)
        if (candidate[0], candidate[1] - candidate[2], -candidate[1]) > (best[0], best[1] - best[2], -best[1]): best = candidate
    return best

# --- largest_ones_square (intake) ---
def largest_ones_square(grid):
    if not grid: return (0, -1, -1)
    cols = len(grid[0])
    if any(len(row) != cols for row in grid): raise ValueError("ragged grid")
    previous, best = [0] * (cols + 1), (0, -1, -1)
    for r, row in enumerate(grid):
        current = [0] * (cols + 1)
        for c, value in enumerate(row):
            if value:
                current[c + 1] = 1 + min(previous[c], previous[c + 1], current[c])
                side = current[c + 1]
                if side > best[0]: best = (side, r - side + 1, c - side + 1)
        previous = current
    return best

# --- next_greater_steps (intake) ---
def next_greater_steps(nums):
    n = len(nums)
    answer, stack = [0] * n, []
    for k in range(2 * n):
        while stack and nums[stack[-1]] < nums[k % n]:
            index = stack.pop()
            answer[index] = k - index
        if k < n: stack.append(k)
    return answer

# --- repeating_decimal (intake) ---
def repeating_decimal(numerator, denominator):
    if denominator == 0: raise ZeroDivisionError("zero denominator")
    if numerator == 0: return "0"
    sign = "-" if (numerator < 0) != (denominator < 0) else ""
    whole, remainder = divmod(abs(numerator), abs(denominator))
    prefix = sign + str(whole)
    if remainder == 0: return prefix
    digits, seen = [], {}
    while remainder and remainder not in seen:
        seen[remainder] = len(digits)
        digit, remainder = divmod(remainder * 10, abs(denominator))
        digits.append(str(digit))
    if remainder:
        digits.insert(seen[remainder], "(")
        digits.append(")")
    return prefix + "." + "".join(digits)

# --- directed_euler_walk (intake) ---
def directed_euler_walk(n, edges):
    if not edges: return [0] if n else []
    adjacency, balance = [[] for _ in range(n)], [0] * n
    for u, v in edges:
        adjacency[u].append(v)
        balance[u] += 1
        balance[v] -= 1
    starts = [v for v in range(n) if balance[v] == 1]
    ends = [v for v in range(n) if balance[v] == -1]
    if any(abs(value) > 1 for value in balance): return None
    if not ((not starts and not ends) or (len(starts) == len(ends) == 1)): return None
    start = starts[0] if starts else next(v for v in range(n) if adjacency[v])
    stack, trail = [start], []
    while stack:
        vertex = stack[-1]
        if adjacency[vertex]: stack.append(adjacency[vertex].pop())
        else: trail.append(stack.pop())
    return trail[::-1] if len(trail) == len(edges) + 1 else None

# --- min_ragged_wrap (intake) ---
def min_ragged_wrap(words, width):
    if width <= 0 or any(len(word) > width for word in words): raise ValueError('invalid width')
    n = len(words)
    costs, following = [0] * (n + 1), [n] * n
    for i in range(n - 1, -1, -1):
        used, best = 0, None
        for j in range(i, n):
            used += len(words[j]) + (j > i)
            if used > width: break
            cost = (0 if j == n - 1 else (width - used) ** 2) + costs[j + 1]
            if best is None or cost < best: best, following[i] = cost, j + 1
        costs[i] = best
    lines, i = [], 0
    while i < n:
        j = following[i]
        lines.append(' '.join(words[i:j]))
        i = j
    return lines

# --- deadline_dispatch (intake) ---
def deadline_dispatch(events):
    import heapq
    heap, pending, out = [], {}, []
    for serial, event in enumerate(events):
        op = event[0] if event else None
        if op == 'set' and len(event) == 3:
            _, task, due = event
            if due < 0: raise ValueError('negative timestamp')
            pending[task] = (due, serial)
            heapq.heappush(heap, (due, serial, task))
        elif op == 'cancel' and len(event) == 2: pending.pop(event[1], None)
        elif op == 'run' and len(event) == 2:
            t = event[1]
            if t < 0: raise ValueError('negative timestamp')
            emitted = []
            while heap and heap[0][0] <= t:
                due, stamp, task = heapq.heappop(heap)
                if pending.get(task) != (due, stamp): continue
                del pending[task]
                emitted.append(task)
            out.append(emitted)
        else: raise ValueError('invalid event')
    return out


# --- expand_counted_pattern (intake) ---
def expand_counted_pattern(s):
    stack, current, i = [], "", 0
    while i < len(s):
        ch = s[i]
        if ch == "\\":
            if i + 1 == len(s): raise ValueError("dangling escape")
            current += s[i + 1]
            i += 2
        elif "0" <= ch <= "9":
            j = i
            while i < len(s) and "0" <= s[i] <= "9": i += 1
            if i == len(s) or s[i] != "[": raise ValueError("missing [")
            stack.append((current, int(s[j:i])))
            current = ""
            i += 1
        elif ch == "[": raise ValueError("missing count")
        elif ch == "]":
            if not stack: raise ValueError("unmatched ]")
            prefix, count = stack.pop()
            current = prefix + count * current
            i += 1
        else:
            current += ch
            i += 1
    if stack: raise ValueError("unclosed [")
    return current
