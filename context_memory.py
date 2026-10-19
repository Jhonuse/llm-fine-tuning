"""
VersionedContextMemory - efficient, transparent context memory for long-running agents.

- Model-agnostic: tokenizer and summarizer are plain callables you can swap out.
- Every message is placed in a tier: VERBATIM (kept word-for-word), SUMMARY, or DISCARD.
- Every decision is logged with a reason and stored in a chain of versioned commits,
  each labelled with a topic. Originals are never deleted, so every decision is reversible.
- Supports restore, branch, merge, manual trim/expand/pin, and explain().

Run:  python context_memory.py      (runs the demo + benchmark)
"""
import re, random, itertools
from dataclasses import dataclass, field

VERBATIM, SUMMARY, DISCARD = "verbatim", "summary", "discard"
RANK = {DISCARD: 0, SUMMARY: 1, VERBATIM: 2}

# ---------- pluggable pieces (swap for any model/tokenizer) ----------
def default_tokens(text): return max(1, len(text) // 4)

def default_summarize(text, max_words=14):
    words = text.split()
    return " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")

KEEP_RE = re.compile(r"\b(decid\w*|must|never|always|deadline|budget|approved|constraint|"
                     r"requirement|chosen|rule)\b|\d{3,}|```", re.I)
CHATTER_RE = re.compile(r"^\s*(ok|okay|thanks|thank you|sure|got it|hi|hello|great|yes|no|cool)\b", re.I)

@dataclass
class Message:
    id: int
    session: int
    role: str
    text: str
    topic: str

@dataclass
class Commit:
    id: str
    parents: list
    topic: str
    branch: str
    state: dict          # msg_id -> (tier, reason)
    note: str = ""

class VersionedContextMemory:
    def __init__(self, tokens=default_tokens, summarize=default_summarize):
        self.tokens, self.summarize = tokens, summarize
        self.archive = {}                     # msg_id -> Message (never deleted)
        self.summaries = {}                   # msg_id -> short text
        self.pinned = set()
        self.commits = {}
        self.branches = {"main": None}
        self.current = "main"
        self.state = {}
        self._seen = set()
        self._n = itertools.count(1)

    # ---------- ingest: classify every new message, then commit ----------
    def classify(self, m):
        norm = m.text.strip().lower()
        if m.id in self.pinned:
            return VERBATIM, "pinned by user"
        if norm in self._seen:
            return DISCARD, "exact duplicate of earlier message"
        if KEEP_RE.search(m.text):
            return VERBATIM, "contains decision / constraint / number / code"
        if len(m.text.split()) < 8 and CHATTER_RE.match(m.text):
            return DISCARD, "short acknowledgement with no information"
        return SUMMARY, "general discussion; reduced to short summary"

    def ingest(self, messages, topic):
        for m in messages:
            self.archive[m.id] = m
            tier, why = self.classify(m)
            self._seen.add(m.text.strip().lower())
            if tier == SUMMARY:
                self.summaries[m.id] = self.summarize(m.text)
            self.state[m.id] = (tier, why)
        return self._commit(topic, f"ingested {len(messages)} messages")

    def _commit(self, topic, note, parents=None):
        cid = f"c{next(self._n)}"
        head = self.branches[self.current]
        c = Commit(cid, parents if parents is not None else ([head] if head else []),
                   topic, self.current, dict(self.state), note)
        self.commits[cid] = c
        self.branches[self.current] = cid
        return cid

    # ---------- version control ----------
    def restore(self, commit_id):
        """Return the working memory to an earlier point (recorded as a new commit)."""
        self.state = dict(self.commits[commit_id].state)
        return self._commit(self.commits[commit_id].topic, f"restored to {commit_id}")

    def branch(self, name, from_commit=None):
        src = from_commit or self.branches[self.current]
        self.branches[name] = src
        self.current = name
        if src: self.state = dict(self.commits[src].state)

    def switch(self, name):
        self.current = name
        h = self.branches[name]
        self.state = dict(self.commits[h].state) if h else {}

    def merge(self, other):
        """Merge another branch into the current one; higher tier wins, conflicts are logged."""
        a, b = self.branches[self.current], self.branches[other]
        merged, conflicts = dict(self.state), []
        for mid, (tier, why) in self.commits[b].state.items():
            if mid not in merged:
                merged[mid] = (tier, why)
            elif merged[mid][0] != tier:
                keep = tier if RANK[tier] > RANK[merged[mid][0]] else merged[mid][0]
                conflicts.append((mid, merged[mid][0], tier, keep))
                merged[mid] = (keep, f"merge conflict resolved: kept higher tier ({keep})")
        self.state = merged
        cid = self._commit("merge", f"merged '{other}' into '{self.current}'; {len(conflicts)} conflicts", [a, b])
        return cid, conflicts

    # ---------- manual control ----------
    def pin(self, mid):    self.pinned.add(mid); self.set_tier(mid, VERBATIM, "pinned by user")
    def trim(self, mid):   self.set_tier(mid, DISCARD, "trimmed by user")
    def expand(self, mid): self.set_tier(mid, VERBATIM, "expanded by user")
    def set_tier(self, mid, tier, why):
        if tier == SUMMARY and mid not in self.summaries:
            self.summaries[mid] = self.summarize(self.archive[mid].text)
        self.state[mid] = (tier, why)
        return self._commit(self.archive[mid].topic, f"manual: msg {mid} -> {tier}")

    # ---------- build context for the model ----------
    def context(self, query=None, budget=None):
        """Verbatim items + summaries. If a query is given, matching summaries are auto-expanded.
        If over budget, the oldest summaries are dropped first (verbatim is kept)."""
        qwords = set(re.findall(r"\w{4,}", (query or "").lower()))
        items = []
        for mid in sorted(self.state):
            tier, _ = self.state[mid]
            m = self.archive[mid]
            if tier == VERBATIM:
                items.append((mid, tier, m.text))
            elif tier == SUMMARY:
                hit = qwords & set(re.findall(r"\w{4,}", m.text.lower()))
                items.append((mid, VERBATIM if len(hit) >= 2 else SUMMARY,
                              m.text if len(hit) >= 2 else self.summaries[mid]))
        if budget:
            while sum(self.tokens(t) for _, _, t in items) > budget:
                idx = next((i for i, x in enumerate(items) if x[1] == SUMMARY), None)
                if idx is None: break
                items.pop(idx)
        return "\n".join(t for _, _, t in items)

    # ---------- transparency ----------
    def explain(self, mid=None):
        if mid is not None:
            t, why = self.state.get(mid, (None, "not in current memory"))
            return f"msg {mid}: {t} - {why}"
        counts = {t: 0 for t in RANK}
        reasons = {}
        for t, why in self.state.values():
            counts[t] += 1; reasons[(t, why)] = reasons.get((t, why), 0) + 1
        lines = [f"Memory on '{self.current}': {counts}"]
        lines += [f"  {n:4d} x {t:8s} because: {w}" for (t, w), n in sorted(reasons.items())]
        return "\n".join(lines)

    def history(self):
        out, cid = [], self.branches[self.current]
        while cid:
            c = self.commits[cid]; out.append(f"{c.id} [{c.topic}] {c.note}")
            cid = c.parents[0] if c.parents else None
        return out[::-1]

# ---------- benchmark ----------
FILLER = ["We went over the general background and looked at several options without settling on anything.",
          "I walked through how the layout and the overall style could be organised in a flexible way.",
          "There are many small observations about the process that might help us think about next steps.",
          "Let me describe the surrounding situation in more detail so everyone shares the same picture."]
WORDS = ["alpha","river","stone","maple","orbit","delta","harbor","violet","summit","ember","cobalt","meadow"]
CHAT = ["ok", "thanks", "sure", "got it", "great", "ok thanks"]

def make_conversation(sessions=5, per_session=120, seed=1):
    random.seed(seed)
    topics = ["planning", "vendor", "design", "testing", "rollout"]
    facts, msgs, mid = [], [], 0
    for s in range(sessions):
        for i in range(per_session):
            mid += 1
            if i % 40 == 5:
                code = f"{random.randint(1000, 9999)}"
                text = f"Decision: approved budget code {code} for the {topics[s]} stage."
                facts.append(code)
            elif i % 3 == 0:
                text = random.choice(CHAT)
            else:
                text = random.choice(FILLER) + " " + random.choice(FILLER) + f" Ref {random.choice(WORDS)} {random.choice(WORDS)}."
            msgs.append(Message(mid, s, "user" if i % 2 == 0 else "agent", text, topics[s]))
    return msgs, facts

def run_benchmark():
    msgs, facts = make_conversation()
    mem = VersionedContextMemory()
    for s in range(5):
        batch = [m for m in msgs if m.session == s]
        mem.ingest(batch, batch[0].topic)
    price, base_ms, ms_tok = 0.003, 200, 0.1      # simulated cost / latency model
    full = "\n".join(m.text for m in msgs)
    trunc = "\n".join(m.text for m in msgs[-60:])
    cx = mem.context()
    rows = []
    for name, ctx in [("Full history", full), ("Truncate (last 60)", trunc), ("Versioned memory", cx)]:
        tk = mem.tokens(ctx)
        recall = sum(f in ctx for f in facts) / len(facts)
        rows.append((name, tk, tk / 1000 * price, base_ms + ms_tok * tk, recall))
    print(f"\nBENCHMARK: {len(msgs)} messages, 5 sessions, {len(facts)} planted facts (simulated cost/latency)")
    print(f"{'Method':20s}{'Tokens':>8s}{'Cost/call $':>13s}{'Latency ms':>12s}{'Fact recall':>13s}")
    for n, tk, c, l, r in rows:
        print(f"{n:20s}{tk:8d}{c:13.4f}{l:12.0f}{r:12.0%}")
    saved = 1 - rows[2][1] / rows[0][1]
    print(f"Token reduction vs full history: {saved:.0%}")
    return mem

def demo(mem):
    print("\nEXPLAIN\n" + mem.explain())
    print("\nHISTORY (main)\n" + "\n".join(mem.history()))
    head = mem.branches["main"]
    mem.branch("alt-design", head)
    victim = next(i for i, (t, _) in mem.state.items() if t == SUMMARY)
    mem.expand(victim)
    mem.trim(next(i for i, (t, _) in mem.state.items() if t == VERBATIM and i != victim))
    print("\nBRANCH 'alt-design':", mem.explain(victim))
    mem.switch("main")
    cid, conflicts = mem.merge("alt-design")
    print(f"MERGE alt-design -> main: commit {cid}, {len(conflicts)} conflict(s) resolved")
    mem.restore(head)
    print("RESTORE to", head, "->", mem.history()[-1])

if __name__ == "__main__":
    demo(run_benchmark())
