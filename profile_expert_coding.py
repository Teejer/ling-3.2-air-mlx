"""Profile which MoE layers care most about CODE in Ling-3.2-air.

Loads the built MLX pack through mlx-lm's independent bailing_moe_v3 (same
path as check_pack_python.py), wraps every BailingSparseMoE to record router
top-k indices + scores over a code corpus and a prose baseline, and writes a
per-layer/per-expert activation profile as JSON.

Layer score = conc10 x spec_mass:
  conc10   — share of code routings landing on the layer's top-10 experts
             (how specialist the layer is for code at all)
  spec_mass — share of code routings landing on the top-10 experts by
             code/prose activation ratio (how much of that is code-SPECIFIC,
             not just generally busy experts)

Usage: python profile_expert_coding.py [--pack DIR] [--out expert_profile_coding.json]
"""

import argparse
import json
import re
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.bailing_moe_v3 import (
    BailingSparseMoE,
    Model,
    ModelArgs,
)

ap = argparse.ArgumentParser()
ap.add_argument("--pack", default="Ling-3.2-air-MLX-4bit")
ap.add_argument("--out", default="expert_profile_coding.json")
ap.add_argument("--chunk", type=int, default=1024)
ARGS = ap.parse_args()

PACK = Path(ARGS.pack)

# ------------------------------------------------------------------ corpora
# Code corpus: assistant-style turns across languages and tasks (authored
# here; ~15k tokens of dense code).
CODE = [
    """Write an async job queue in Python with retries and exponential backoff.
```python
import asyncio
import random
from dataclasses import dataclass, field
from typing import Awaitable, Callable

@dataclass
class Job:
    fn: Callable[..., Awaitable[object]]
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    attempts: int = 0
    max_attempts: int = 5

    async def run(self) -> object:
        self.attempts += 1
        try:
            return await self.fn(*self.args, **self.kwargs)
        except Exception:
            if self.attempts >= self.max_attempts:
                raise
            delay = min(30.0, (2 ** self.attempts) + random.random())
            await asyncio.sleep(delay)
            return await self.run()

class Queue:
    def __init__(self, workers: int = 4):
        self._q: asyncio.Queue[Job] = asyncio.Queue()
        self._workers = [asyncio.create_task(self._worker(i)) for i in range(workers)]

    async def _worker(self, idx: int) -> None:
        while True:
            job = await self._q.get()
            try:
                await job.run()
            finally:
                self._q.task_done()

    async def submit(self, fn, *a, **kw) -> None:
        await self._q.put(Job(fn, a, kw))

    async def join(self) -> None:
        await self._q.join()
        for w in self._workers:
            w.cancel()
```""",
    """Implement a Rust slab allocator arena with a bump pointer and free list.
```rust
use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

const BLOCK: usize = 4096;
const ARENA: usize = 1 << 20;

pub struct Arena {
    base: *mut u8,
    bump: AtomicUsize,
    layout: Layout,
}

unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Arena {
    pub fn new() -> Self {
        let layout = Layout::from_size_align(ARENA, BLOCK).unwrap();
        unsafe {
            let base = alloc(layout);
            if base.is_null() {
                handle_alloc_error(layout);
            }
            Self { base, bump: AtomicUsize::new(0), layout }
        }
    }

    pub fn alloc(&self, size: usize, align: usize) -> *mut u8 {
        let align = align.max(16);
        loop {
            let cur = self.bump.load(Ordering::Acquire);
            let addr = self.base as usize;
            let start = (addr + cur + align - 1) & !(align - 1);
            let new_off = (start - addr) + size;
            if new_off > ARENA {
                return ptr::null_mut();
            }
            match self.bump.compare_exchange_weak(
                cur, new_off, Ordering::AcqRel, Ordering::Acquire,
            ) {
                Ok(_) => return start as *mut u8,
                Err(_) => continue,
            }
        }
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        unsafe { dealloc(self.base, self.layout) }
    }
}
```""",
    """Review this TypeScript generic cache with TTL and LRU eviction for bugs.
```typescript
type Node<V> = { key: string; value: V; expires: number };

export class TtlLru<V> {
  private map = new Map<string, Node<V>>();

  constructor(private capacity = 128, private ttlMs = 60_000) {}

  get(key: string): V | undefined {
    const node = this.map.get(key);
    if (!node) return undefined;
    if (node.expires < Date.now()) {
      this.map.delete(key);
      return undefined;
    }
    // refresh recency: delete + reinsert moves to MRU end
    this.map.delete(key);
    this.map.set(key, node);
    return node.value;
  }

  set(key: string, value: V, ttlMs = this.ttlMs): void {
    if (this.map.has(key)) this.map.delete(key);
    this.map.set(key, { key, value, expires: Date.now() + ttlMs });
    if (this.map.size > this.capacity) {
      const oldest = this.map.keys().next().value!;
      this.map.delete(oldest);
    }
  }

  prune(): number {
    const now = Date.now();
    let removed = 0;
    for (const [k, n] of this.map) {
      if (n.expires < now) {
        this.map.delete(k);
        removed += 1;
      }
    }
    return removed;
  }
}
// BUG: prune() mutates the Map while iterating it; and get() on an
// expired node returns undefined but leaves the stale insert order.
```""",
    """A Go worker pool with context cancellation and errgroup-style collection:
```go
package pool

import (
	"context"
	"errors"
	"sync"
)

type Task func(ctx context.Context) error

type Pool struct {
	tasks chan Task
	wg    sync.WaitGroup
	mu    sync.Mutex
	errs  []error
}

func New(ctx context.Context, workers int, queue int) *Pool {
	p := &Pool{tasks: make(chan Task, queue)}
	for i := 0; i < workers; i++ {
		p.wg.Add(1)
		go func() {
			defer p.wg.Done()
			for t := range p.tasks {
				if err := t(ctx); err != nil && !errors.Is(err, context.Canceled) {
					p.mu.Lock()
					p.errs = append(p.errs, err)
					p.mu.Unlock()
				}
			}
		}()
	}
	go func() {
		<-ctx.Done()
		close(p.tasks)
	}()
	return p
}

func (p *Pool) Submit(t Task) bool {
	select {
	case p.tasks <- t:
		return true
	default:
		return false
	}
}

func (p *Pool) Wait() error {
	p.wg.Wait()
	return errors.Join(p.errs...)
}
```""",
    """Zig: an arena-backed string interner with FNV-1a hashing.
```zig
const std = @import("std");

pub const Interner = struct: {
    const Entry = struct { hash: u64, offset: u32, len: u32 };

    table: std.AutoHashMap(u64, u32),
    arena: std.heap.ArenaAllocator,
    buf: std.ArrayList(u8),

    pub fn init(gpa: std.mem.Allocator) Interner {
        return .{
            .table = std.AutoHashMap(u64, u32).init(gpa),
            .arena = .init(gpa),
            .buf = std.ArrayList(u8).init(gpa),
        };
    }

    pub fn intern(self: *Interner, s: []const u8) ![]const u8 {
        const hash = std.hash.Fnv1a_64.hash(s);
        if (self.table.get(hash)) |off| {
            const stored = self.buf.items[off..][0..@intCast(self.entryLen(off) orelse return error.Corrupt)];
            if (std.mem.eql(u8, stored, s)) return stored;
        }
        const off: u32 = @intCast(self.buf.items.len);
        try self.buf.appendSlice(s);
        try self.table.put(hash, off);
        return self.buf.items[off..];
    }

    fn entryLen(self: *Interner, off: u32) ?u32 {
        _ = self;
        return @intCast(off); // placeholder: real impl stores len in Entry
    }
};
```""",
    """Debug this C ring buffer; the wrap-around read is wrong:
```c
typedef struct {
    uint8_t *data;
    size_t cap, head, tail;
} ring_t;

int ring_write(ring_t *r, const uint8_t *src, size_t n) {
    if (n > ring_free(r)) return -1;
    size_t first = r->cap - r->head;
    size_t a = n < first ? n : first;
    memcpy(r->data + r->head, src, a);
    memcpy(r->data, src + a, n - a);
    r->head = (r->head + n) % r->cap;
    return 0;
}

size_t ring_read(ring_t *r, uint8_t *dst, size_t n) {
    size_t avail = (r->head + r->cap - r->tail) % r->cap;
    if (n > avail) n = avail;
    /* BUG: forgot the wrap case; reads past head linearly */
    memcpy(dst, r->data + r->tail, n);
    r->tail = (r->tail + n) % r->cap;
    return n;
}
```
Fix ring_read to split the copy at the wrap boundary like ring_write does.""",
    """SQL: window-function query for sessions and per-session event gaps.
```sql
WITH ordered AS (
  SELECT user_id, ts, event_type,
         LAG(ts) OVER (PARTITION BY user_id ORDER BY ts) AS prev_ts
  FROM events
  WHERE ts >= NOW() - INTERVAL '30 days'
),
sessionized AS (
  SELECT *,
         SUM(CASE WHEN prev_ts IS NULL OR ts - prev_ts > INTERVAL '30 min'
             THEN 1 ELSE 0 END)
           OVER (PARTITION BY user_id ORDER BY ts
                 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS session_no
  FROM ordered
)
SELECT user_id, session_no,
       MIN(ts) AS started_at,
       MAX(ts) AS ended_at,
       COUNT(*) FILTER (WHERE event_type = 'click') AS clicks,
       EXTRACT(EPOCH FROM MAX(ts) - MIN(ts)) AS duration_s
FROM sessionized
GROUP BY user_id, session_no
HAVING COUNT(*) > 1
ORDER BY started_at DESC
LIMIT 100;
```""",
    """Bash deploy script with rollback on failed health check:
```bash
#!/usr/bin/env bash
set -euo pipefail

APP=api SERVICE_DIR=/srv/api RELEASES=/srv/api/releases
NEW=${1:?usage: deploy.sh <sha>}

ln -s "$RELEASES/$NEW" "$SERVICE_DIR/new"
mv -Tf "$SERVICE_DIR/new" "$SERVICE_DIR/current"
systemctl reload "$APP"

for i in {1..30}; do
  if curl -fsS localhost:8080/healthz >/dev/null; then
    echo "healthy after ${i}s"; exit 0
  fi
  sleep 1
done

echo "health check failed; rolling back"
mv -Tf "$SERVICE_DIR/previous" "$SERVICE_DIR/current"
systemctl reload "$APP"
exit 1
```""",
    """Dijkstra with a binary heap in Python for the routing overlay:
```python
import heapq
from collections import defaultdict

def dijkstra(graph, src, dst):
    dist = {src: 0}
    prev = {}
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        if u == dst:
            break
        for v, w in graph[u].items():
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if dst not in dist:
        return None, float("inf")
    path, cur = [], dst
    while cur != src:
        path.append(cur)
        cur = prev[cur]
    path.append(src)
    return path[::-1], dist[dst]

graph = defaultdict(dict)
for a, b, w in [("a", "b", 4), ("a", "c", 1), ("c", "b", 2), ("b", "d", 5)]:
    graph[a][b] = w
    graph[b][a] = w
print(dijkstra(graph, "a", "d"))
```""",
    """Refactor this Python class so the parser is testable without a network:
```python
class WeatherClient:
    def __init__(self, session, base="https://api.example.com"):
        self.session = session
        self.base = base

    def forecast(self, city: str, days: int = 3) -> dict:
        r = self.session.get(f"{self.base}/v2/fc", params={"q": city, "d": days}, timeout=5)
        r.raise_for_status()
        data = r.json()
        out = {}
        for row in data["list"]:
            out[row["dt"]] = {
                "temp": row["main"]["temp"],
                "pop": row["pop"],
                "desc": row["weather"][0]["description"],
            }
        return out
```
Extract the HTTP call behind a `Fetcher` protocol, inject a fake in tests, and keep `forecast` pure.""",
    """JS: a JSON-lines stream parser handling partial chunks and CRLF.
```javascript
const { Transform } = require("node:stream");

class JsonlParser extends Transform {
  #buf = "";

  _transform(chunk, _enc, cb) {
    this.#buf += chunk.toString("utf8");
    let idx;
    while ((idx = this.#buf.indexOf("\n")) !== -1) {
      let line = this.#buf.slice(0, idx);
      this.#buf = this.#buf.slice(idx + 1);
      if (line.endsWith("\r")) line = line.slice(0, -1);
      if (!line) continue;
      try {
        this.push(JSON.parse(line));
      } catch (err) {
        this.emit("error", new SyntaxError(`bad line: ${err.message}`));
        return cb();
      }
    }
    cb();
  }

  _flush(cb) {
    if (this.#buf.trim()) {
      try { this.push(JSON.parse(this.#buf)); } catch { /* drop tail */ }
    }
    this.#buf = "";
    cb();
  }
}

module.exports = { JsonlParser };
```""",
    """Rust: parse a binary TLV stream with nom-style combinators by hand.
```rust
#[derive(Debug, PartialEq)]
pub enum TlvError { Short, BadTag(u8), NestedTooDeep }

pub struct Tlv<'a> { pub tag: u8, pub body: &'a [u8] }

pub fn parse_tlv<'a>(depth: u32, input: &'a [u8]) -> Result<(Tlv<'a>, &'a [u8]), TlvError> {
    if depth > 8 { return Err(TlvError::NestedTooDeep); }
    if input.len() < 3 { return Err(TlvError::Short); }
    let tag = input[0];
    let len = u16::from_be_bytes([input[1], input[2]]) as usize;
    if input.len() < 3 + len { return Err(TlvError::Short); }
    let body = &input[3..3 + len];
    match tag {
        0x30 | 0x31 => {
            let mut rest = body;
            while !rest.is_empty() {
                let (item, r) = parse_tlv(depth + 1, rest)?;
                println!("{item:?}");
                rest = r;
            }
        }
        t if t & 0x80 != 0 => return Err(TlvError::BadTag(t)),
        _ => {}
    }
    Ok((Tlv { tag, body }, &input[3 + len..]))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_is_short() { assert_eq!(parse_tlv(0, b"\x02"), Err(TlvError::Short)); }
}
```""",
    """Explain and fix the race in this Python double-buffered config swap.
```python
import threading

class Config:
    _lock = threading.Lock()
    _current: dict = {}

    @classmethod
    def reload(cls, new: dict) -> None:
        # writers build a whole new dict, then swap the reference
        staged = dict(new)
        staged["version"] = cls._current.get("version", 0) + 1
        with cls._lock:
            cls._current = staged   # readers do: cfg = Config._current

    @classmethod
    def get(cls, key: str):
        # RACE: _current is read without the lock and __setitem__
        # elsewhere mutates the dict in place:
        cls._current[key] = None
        return cls._current.get(key)
```
The in-place mutation in get() breaks copy-on-read; remove it and show the immutable-read version.""",
    """C++: RAII file lock with flock and a move-only handle.
```cpp
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <string>

class FileLock {
    int fd_ = -1;
    std::string path_;
public:
    explicit FileLock(const std::string& path) : path_(path) {
        fd_ = ::open(path.c_str(), O_RDWR | O_CREAT, 0644);
        if (fd_ < 0) throw std::runtime_error("open " + path);
        if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("locked: " + path);
        }
    }
    FileLock(FileLock&& o) noexcept : fd_(o.fd_) { o.fd_ = -1; std::swap(path_, o.path_); }
    FileLock& operator=(FileLock&& o) noexcept {
        if (this != &o) { release(); fd_ = o.fd_; o.fd_ = -1; std::swap(path_, o.path_); }
        return *this;
    }
    FileLock(const FileLock&) = delete;
    FileLock& operator=(const FileLock&) = delete;
    ~FileLock() { release(); }
    void release() {
        if (fd_ >= 0) { ::flock(fd_, LOCK_UN); ::close(fd_); fd_ = -1; }
    }
};
```""",
    """Kotlin flow: debounce a search field and cancel stale requests.
```kotlin
class SearchRepo(private val api: Api) {
    fun search(query: Flow<String>): Flow<List<Item>> = query
        .debounce(250)
        .distinctUntilChanged()
        .map { it.trim() }
        .filter { it.length >= 2 }
        .flatMapLatest { q ->
            flow {
                emitAll(api.search(q))
            }.catch { e ->
                if (e !is CancellationException) emit(listOf())
            }
        }
        .flowOn(Dispatchers.IO)
}
```""",
    """Python: property-based tests for an interval merge function.
```python
from hypothesis import given, strategies as st

def merge(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out

iv = st.lists(
    st.integers(0, 50).flatmap(lambda s:
        st.tuples(st.just(s), st.integers(s, s + 20))),
    max_size=12,
)

@given(iv)
def test_cover_same_points(xs):
    assert {p for a, b in xs for p in range(a, b + 1)} == {
        p for a, b in merge(xs) for p in range(a, b + 1)}

@given(iv)
def test_sorted_disjoint(xs):
    m = merge(xs)
    assert all(a <= b for a, b in m)
    assert all(prev[1] < cur[0] for prev, cur in zip(m, m[1:]))
```""",
    """Go: an LRU cache with generics and a TTL janitor goroutine.
```go
type entry[V any] struct {
	val      V
	expire   time.Time
}

type Cache[V any] struct {
	mu    sync.Mutex
	items map[string]entry[V]
	order []string
	cap   int
	ttl   time.Duration
}

func (c *Cache[V]) Get(k string) (V, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.items[k]
	if !ok || time.Now().After(e.expire) {
		var zero V
		delete(c.items, k)
		return zero, false
	}
	c.touch(k)
	return e.val, true
}

func (c *Cache[V]) janitor(ctx context.Context) {
	t := time.NewTicker(c.ttl / 2)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			c.mu.Lock()
			now := time.Now()
			for k, e := range c.items {
				if now.After(e.expire) {
					delete(c.items, k)
					c.evictKey(k)
				}
			}
			c.mu.Unlock()
		}
	}
}
```""",
    """Write a regex-free tokenizer for a small expression language in Python.
```python
import string

KEYWORDS = {"let", "in", "fn"}

def tokenize(src):
    toks, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if c in string.whitespace:
            i += 1
        elif c in "(){}[],:=":
            toks.append((c, c)); i += 1
        elif c.isdigit():
            j = i
            while j < n and (src[j].isdigit() or src[j] == "."):
                j += 1
            toks.append(("num", float(src[i:j]))); i = j
        elif c.isalpha() or c == "_":
            j = i
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            word = src[i:j]
            toks.append(("kw" if word in KEYWORDS else "id", word)); i = j
        else:
            raise ValueError(f"char {c!r} at {i}")
    return toks
```""",
    """Fix this Python GIL-bound CPU profiler wrapper to use perf_counter_ns and thread time.
```python
import threading
import time
from contextlib import contextmanager

@contextmanager
def profile(label):
    t0 = time.perf_counter_ns()
    cpu0 = time.thread_time_ns()
    yield
    wall = (time.perf_counter_ns() - t0) / 1e6
    cpu = (time.thread_time_ns() - cpu0) / 1e6
    print(f"{label}: wall={wall:.2f}ms cpu={cpu:.2f}ms "
          f"parallelism={cpu / max(wall, 1e-9):.2f}")

with profile("hash loop"):
    threads = [threading.Thread(target=lambda: hash(list(range(100_000)))) for _ in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
```
Expect parallelism near 1.0 — that is the GIL. Move to ProcessPoolExecutor and compare.""",
    """Zig engine patch style: add a hook to a struct without touching callers.
```zig
pub const Layer = struct {
    id: u32,
    inner: Inner,
    hooks: ?*const Hooks = null,

    pub const Hooks = struct {
        on_load: ?*const fn (*Layer) void = null,
        on_step: ?*const fn (*Layer, f32) void = null,
    };

    pub fn step(self: *Layer, dt: f32) void {
        self.inner.advance(dt);
        if (self.hooks) |h| if (h.on_step) |f| f(self, dt);
    }
};
```""",
]

# Prose baseline: original encyclopedic-style paragraphs (non-code).
PROSE = [
    """The formation of sedimentary rock occurs over vast spans of time. Layers of
sand, silt and organic debris settle in basins, and the weight of newer beds
squeezes water from the older ones. Minerals precipitate in the remaining pores
and cement the grains together. Fossils trapped between the beds record the
environments of their deposition, and geologists read the sequence like a book,
each formation a chapter about ancient rivers, seas and climates.""",
    """Coffee spread from the highlands of Ethiopia across the Red Sea to Yemen in
the fifteenth century, where Sufi orders cultivated it and drank it through
nighttime devotions. Ottoman coffeehouses became centers of conversation and
news, prompting rulers to watch them closely. The bean traveled to Europe
through Venetian merchants, and plantations later carried it across the
Atlantic, reshaping the labor history of whole continents.""",
    """A beehive is a superorganism in miniature. Foragers scout for blossoms and
advertise their location through a waggle dance whose angle encodes direction
relative to the sun. Nurses feed larvae differently, and chemistry decides
destiny: a diet of royal jelly raises queens. Guard bees ventilate the comb by
fanning, and when the colony swarms, tens of thousands of individuals leave in
a matter of minutes to find a new home.""",
    """The development of mechanical clocks in medieval Europe changed how people
organized their day. Monastic hours had divided time into prayer, but civic
bells and workshop whistles demanded precision. Escapement mechanisms tamed
the fall of weights, and later pendulums made timepieces accurate enough for
navigation. Standard time zones arrived with the railways, synchronizing what
had once been the local noon of every town.""",
    """Coral reefs occupy a fraction of the ocean floor yet shelter a quarter of
marine species. The animals themselves build the structure, secreting calcium
carbonate beneath symbiotic algae that repay the favor with sugars from
sunlight. When the water runs too warm the partnership breaks, the algae
depart, and the colony bleaches. Recovery is possible, but only if the sea
cools before starvation sets in.""",
    """The silk roads were less a road than a lattice of caravan tracks. Cities
flourished where water and passes allowed the routes to touch: Samarkand,
Kashgar, Palmyra. Goods moved in relays rather than end to end, and with bolts
of silk and ingots of glass traveled religions, numerals, and plagues. The
caravanserai, spaced a day's march apart, was the infrastructure that made the
network possible.""",
    """Chess openings encode centuries of collective analysis. The Ruy Lopez
presses Black's e5 pawn before committing pieces elsewhere; the Sicilian
concedes space on the kingside for a long game of counterattack on the
flanks. Engines later rehabilitated old defenses and buried new ones, and the
theory books, rewritten every season, now read as annotations to machine
judgment.""",
    """Tides arise from the differential pull of moon and sun across the face of
the earth. Oceans respond as a single sloshing basin, and coastlines shape the
wave: a funneling estuary can double a tide, while a shallow shelf slows it.
Twice a day the water withdraws to reveal the intertidal, a brutal
neighborhood where organisms must survive both drowning and drying.""",
    """The history of cartography is a history of compromise. Every flat map tears
the sphere somehow: Mercator preserves angles for navigators while inflating
the poles; equal-area projections keep the sizes honest but mangle the shapes.
Modern web maps hide the argument by switching projections mid-zoom, and
satellite observation finally let mapmakers trace roads the surveyors never
walked.""",
    """Vaccines train the immune system with a rehearsal. A weakened pathogen, a
purified fragment, or a messenger instruction presents the body with markers
it has not met, and the adaptive system answers with clones of cells that
remember. Booster shots refine that memory. The principle, born from smallpox
material, now reaches toward malaria and a dozen cancers.""",
]


def main():
    # ---- load pack (same path as check_pack_python.py)
    cfg = json.load(open(PACK / "config.json"))
    q = cfg["quantization"]
    args = ModelArgs.from_dict(cfg)
    model = Model(args)

    import mlx.nn as nn

    vp = model.quant_predicate

    weights = {}
    for f in sorted(PACK.glob("model-*.safetensors")):
        weights.update(mx.load(str(f)))

    def infer_bits(prefix):
        w = weights.get(f"{prefix}.weight")
        s = weights.get(f"{prefix}.scales")
        if w is None or s is None:
            return q["bits"]
        in_dim = s.shape[-1] * q["group_size"]
        # this MLX build packs into uint32 words (U32 in safetensors)
        per_elem = {mx.uint8: 8, mx.uint16: 16, mx.uint32: 32}.get(w.dtype, 8)
        return w.shape[-1] * per_elem // in_dim

    def class_predicate(path, mod):
        if not hasattr(mod, "to_quantized"):
            return False
        p = vp(path, mod)
        return p if p is not True else {
            "group_size": q["group_size"], "bits": infer_bits(path)
        }

    nn.quantize(model, group_size=q["group_size"], bits=q["bits"], mode="affine",
                class_predicate=class_predicate)

    n = args.num_hidden_layers
    w2 = {k: v for k, v in weights.items()
          if not (k.startswith("model.layers.") and k.split(".")[2].isdigit()
                  and int(k.split(".")[2]) >= n)}
    for li in range(n):
        pre = f"model.layers.{li}.attention"
        if f"{pre}.kv_b_proj.weight" not in w2 or f"{pre}.embed_q.weight" in w2:
            continue
        kw = w2.pop(f"{pre}.kv_b_proj.weight")
        ks, kb = w2.pop(f"{pre}.kv_b_proj.scales", None), w2.pop(f"{pre}.kv_b_proj.biases", None)
        kvb = kw if ks is None else mx.dequantize(
            kw, ks, kb, group_size=q["group_size"], bits=infer_bits(f"{pre}.kv_b_proj"), mode="affine")
        heads, nope, vd = args.num_attention_heads, args.qk_nope_head_dim, args.v_head_dim
        v = kvb.reshape(heads, nope + vd, -1)
        w2[f"{pre}.embed_q.weight"] = mx.contiguous(v[:, :nope, :].swapaxes(-1, -2))
        w2[f"{pre}.unembed_out.weight"] = mx.contiguous(v[:, nope:, :])

    missing = model.load_weights(list(w2.items()), strict=False)
    print("missing/unexpected:", list(missing)[:8], "total", len(missing))
    # pre-load lazily per layer: one giant eval trips the Metal command
    # buffer watchdog, especially while the NVMe is busy
    for layer in model.model.layers:
        mx.eval(layer.parameters())
    mx.eval(model.model.word_embeddings.parameters(), model.lm_head.parameters())

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(PACK))

    # ---- routing recorder: wrap BailingSparseMoE.__call__ once, tag instances
    n_exp = args.num_experts
    CODE_C = [np.zeros(n_exp, np.int64) for _ in range(n)]
    CODE_S = [np.zeros(n_exp, np.float64) for _ in range(n)]
    GEN_C = [np.zeros(n_exp, np.int64) for _ in range(n)]
    GEN_S = [np.zeros(n_exp, np.float64) for _ in range(n)]
    SINK = {"counts": None, "scores": None}

    orig_call = BailingSparseMoE.__call__

    def prof_call(self, x):
        indices, scores = self.gate(x)
        if SINK["counts"] is not None:
            mx.eval(indices, scores)
            idx = np.asarray(indices).reshape(-1)
            sc = np.asarray(scores).reshape(-1)
            np.add.at(SINK["counts"][self._pid], idx, 1)
            np.add.at(SINK["scores"][self._pid], idx, sc)
        routed = self.switch_mlp(x, indices)
        routed = (routed * scores[..., None]).sum(axis=-2).astype(x.dtype)
        return routed + self.shared_experts(x)

    BailingSparseMoE.__call__ = prof_call
    tagged = 0
    for L, layer in enumerate(model.model.layers):
        if isinstance(layer.mlp, BailingSparseMoE):
            layer.mlp._pid = L
            tagged += 1
    print(f"tagged {tagged} MoE layers")

    cache = model.make_cache()

    def run_text(text, sink_c, sink_s):
        SINK["counts"], SINK["scores"] = sink_c, sink_s
        ids = tok.encode(text)
        for i in range(0, len(ids), ARGS.chunk):
            chunk = mx.array([ids[i : i + ARGS.chunk]])
            model(chunk, cache)
            mx.eval(model.parameters())  # drain lazy graph; routing already recorded
        SINK["counts"] = SINK["scores"] = None

    t0 = time.time()
    for i, c in enumerate(CODE):
        run_text(c, CODE_C, CODE_S)
        print(f"code {i + 1}/{len(CODE)}  ({time.time() - t0:.0f}s)", flush=True)
    for i, p in enumerate(PROSE):
        run_text(p, GEN_C, GEN_S)
        print(f"prose {i + 1}/{len(PROSE)}  ({time.time() - t0:.0f}s)", flush=True)

    # ---- metrics
    alpha = 1.0
    layers = []
    for L in range(n):
        c, g = CODE_C[L], GEN_C[L].astype(np.float64)
        tot_c, tot_g = c.sum(), g.sum()
        if tot_c == 0:
            continue
        cn = c / c.sum()
        gn = g / max(g.sum(), 1)
        conc10 = float(np.sort(c)[-10:].sum() / tot_c)
        spec = (cn + alpha / n_exp) / (gn + alpha / n_exp)
        top_spec = np.argsort(spec)[-10:]
        spec_mass = float(c[top_spec].sum() / tot_c)
        layers.append(
            {
                "layer": L,
                "code_tokens_routed": int(tot_c),
                "conc10": round(conc10, 4),
                "spec_mass": round(spec_mass, 4),
                "score": round(conc10 * spec_mass, 4),
                "top_coding_experts": [
                    int(e) for e in np.argsort(c)[-20:][::-1]
                ],
                "expert_code_counts": c.tolist(),
            }
        )
    layers_by_score = sorted(layers, key=lambda d: -d["score"])
    out = {
        "pack": str(PACK),
        "corpus": {"code_items": len(CODE), "prose_items": len(PROSE)},
        "layers": layers_by_score,
        "recommended_8bit_layers": sorted(d["layer"] for d in layers_by_score[:12]),
    }
    Path(ARGS.out).write_text(json.dumps(out))
    print("\nranked layers (layer conc10 spec_mass score):")
    for d in layers_by_score:
        print(f"  {d['layer']:>2}  {d['conc10']:.3f}  {d['spec_mass']:.3f}  {d['score']:.3f}")
    print("\nRECOMMENDED 8-bit layers:", out["recommended_8bit_layers"])


if __name__ == "__main__":
    main()
