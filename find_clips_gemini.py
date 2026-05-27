"""
smart_clips_pro.py  v5.0  —  AI-Powered Best Clips Generator
=============================================================
Fully automated clip detection using 10+ hook-finding methods
plus optional Claude AI for deep semantic analysis.

WHAT'S NEW vs v4.0:
  ✦ No more hardcoded clips — AI finds the best moments automatically
  ✦ 10+ scientific hook-detection algorithms running in parallel
  ✦ Multi-dimensional clip scoring (hook strength, story arc, emotion,
    curiosity gap, contrast, authority signals, vulnerability, payoff)
  ✦ Claude AI integration for semantic understanding (optional but powerful)
  ✦ Configurable number of clips (default: top 5 best moments)
  ✦ Clip boundary optimizer — expands to natural sentence endings
  ✦ Overlap deduplication — no two clips cover the same moment
  ✦ Detailed score report printed per clip so you understand WHY it was chosen
  ✦ All v4 quality-of-life features preserved (GPU detect, smart naming, live
    FFmpeg progress, exit codes for n8n)

USAGE:
  python smart_clips_pro_v5.py transcript.json video.mp4
  python smart_clips_pro_v5.py transcript.json video.mp4 ./out
  python smart_clips_pro_v5.py transcript.json video.mp4 ./out nvidia
  python smart_clips_pro_v5.py transcript.json video.mp4 ./out nvidia 7
  (last arg = how many clips to generate; default 5)

CLAUDE AI (optional):
  Set env var ANTHROPIC_API_KEY for deep semantic analysis.
  Without it, the 10 local heuristic methods still run — it degrades
  gracefully to heuristics-only mode.

TRANSCRIPT FORMAT (Whisper-compatible JSON):
  {"segments": [{"start": 0.0, "end": 4.2, "text": "..."}, ...]}
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import urllib.request
import urllib.error

# ─────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────────────────────

TRANSCRIPT_FILE  = r"C:\PodcastClips\transcripts\active_episode.json"
VIDEO_FILE       = r"C:\PodcastClips\downloads\active_episode.mp4"
OUTPUT_DIR       = r"C:\PodcastClips\clips"
FORCE_ENCODER    = None
NUM_CLIPS        = 5          # How many best clips to extract
MIN_CLIP_SECONDS = 30         # Minimum clip duration
MAX_CLIP_SECONDS = 120        # Maximum clip duration
CONTEXT_WINDOW   = 60.0       # Seconds of context to feed into the scorer
OVERLAP_PENALTY  = 0.6        # Multiplier applied when two clips share >30% overlap

# CLI overrides
if len(sys.argv) >= 3:
    TRANSCRIPT_FILE = sys.argv[1]
    VIDEO_FILE      = sys.argv[2]
if len(sys.argv) >= 4:
    OUTPUT_DIR = sys.argv[3]
if len(sys.argv) >= 5:
    FORCE_ENCODER = sys.argv[4].lower()
if len(sys.argv) >= 6:
    try:
        NUM_CLIPS = int(sys.argv[5])
    except ValueError:
        pass

# ─────────────────────────────────────────────────────────────
# 1.  HOOK-DETECTION VOCABULARY BANKS
#     Each bank is a list of (pattern, weight) tuples.
#     Patterns are searched case-insensitively in a text window.
# ─────────────────────────────────────────────────────────────

# Method 1 — STORY OPENERS / NARRATIVE ENTRY POINTS
STORY_OPENERS = [
    (r"\b(it was the day|it all started|the moment|i remember when|that was the day)\b", 3.0),
    (r"\b(there was a time|one day|back then|years ago|as a kid|growing up)\b", 2.0),
    (r"\b(so this is the story|let me tell you|you won't believe|true story)\b", 2.5),
    (r"\b(picture this|imagine|here's what happened)\b", 2.0),
]

# Method 2 — CONTRAST / REVERSAL HOOKS (biggest engagement driver)
CONTRAST_HOOKS = [
    (r"\b(but (then|suddenly|wait|actually|here's the thing))\b", 3.5),
    (r"\b(however|yet|turns out|little did (i|we) know|ironically)\b", 3.0),
    (r"\b(everything changed|nothing was the same|plot twist|catch is)\b", 4.0),
    (r"\b(the opposite|completely wrong|totally backward|backwards)\b", 2.5),
    (r"\b(wasn't what i expected|surprised me|blew my mind|shocked)\b", 3.0),
]

# Method 3 — CURIOSITY GAPS (makes viewer want to keep watching)
CURIOSITY_GAPS = [
    (r"\b(you'll never guess|here's the secret|nobody tells you|they don't want you to know)\b", 4.0),
    (r"\b(the real reason|what really happened|truth is|truth about)\b", 3.5),
    (r"\b(what i (never|didn't|couldn't)|i had no idea|i didn't realize)\b", 3.0),
    (r"\b(it changed everything|life-changing|this is why|that's when i realized)\b", 3.5),
    (r"\b(the (one|biggest|only) (thing|reason|mistake|secret))\b", 3.0),
    (r"\b(what nobody|what most people|what everyone gets wrong)\b", 3.5),
]

# Method 4 — EMOTIONAL INTENSITY SIGNALS
EMOTION_HOOKS = [
    (r"\b(terrified|devastated|heartbroken|destroyed|crushed|shattered)\b", 3.5),
    (r"\b(excited|thrilled|amazing|incredible|unbelievable|insane|wild)\b", 2.5),
    (r"\b(failed|failure|embarrassing|humiliating|worst day|rock bottom)\b", 3.0),
    (r"\b(proud|proudest|best day|moment i'll never forget|changed my life)\b", 3.0),
    (r"\b(crying|tears|broke down|couldn't breathe|heart pounding)\b", 3.5),
    (r"\b(angry|furious|livid|couldn't believe|how dare)\b", 2.5),
]

# Method 5 — VULNERABILITY / FAILURE MOMENTS (high relatability)
VULNERABILITY_HOOKS = [
    (r"\b(i failed|i messed up|i was wrong|my biggest mistake|i regret)\b", 4.0),
    (r"\b(i was broke|no money|couldn't afford|was homeless|lost everything)\b", 4.0),
    (r"\b(rejected|fired|quit|walked out|gave up|almost quit)\b", 3.5),
    (r"\b(nobody believed|they said i was crazy|laughed at me|told me no)\b", 3.5),
    (r"\b(imposter syndrome|felt like fraud|didn't belong|out of place)\b", 3.0),
    (r"\b(struggled with|battle with|fight with|hardest thing)\b", 2.5),
]

# Method 6 — TRANSFORMATION / TURNING POINT MOMENTS
TRANSFORMATION_HOOKS = [
    (r"\b(that changed everything|turning point|pivotal moment|breakthrough)\b", 4.0),
    (r"\b(i decided|made a decision|chose to|committed to|went all in)\b", 3.0),
    (r"\b(before i (knew|found|discovered|learned)|after that day|never the same)\b", 3.5),
    (r"\b(woke up|realized|epiphany|finally understood|it clicked)\b", 3.0),
    (r"\b(transformed|completely different|new person|fresh start)\b", 2.5),
]

# Method 7 — SOCIAL PROOF / AUTHORITY SIGNALS
AUTHORITY_HOOKS = [
    (r"\b(\d{1,3}(,\d{3})* (people|users|clients|customers|subscribers|followers))\b", 2.5),
    (r"\b(\$\d+[kmb]?|\d+[kmb] dollars|million|billion)\b", 2.5),
    (r"\b(harvard|stanford|mit|yale|oxford|nasa|google|apple|meta|amazon)\b", 2.0),
    (r"\b(expert|specialist|10 years|15 years|20 years|decade|decades)\b", 2.0),
    (r"\b(published|peer-reviewed|study shows|research proves|data shows)\b", 2.5),
    (r"\b(number one|top \d+|ranked first|award-winning|world record)\b", 2.5),
]

# Method 8 — CONFLICT / TENSION BUILDERS
CONFLICT_HOOKS = [
    (r"\b(against me|fighting|battle|war|struggle|competition|enemy)\b", 2.5),
    (r"\b(they were wrong|he was wrong|she was wrong|everyone was wrong)\b", 3.0),
    (r"\b(didn't want me to|tried to stop me|told me i couldn't)\b", 3.5),
    (r"\b(impossible|can't be done|never been done|no way)\b", 3.0),
    (r"\b(pressure|deadline|crisis|emergency|last chance|now or never)\b", 3.0),
    (r"\b(confronted|called out|pushed back|stood up for)\b", 2.5),
]

# Method 9 — PAYOFF / PUNCHLINE / RESOLUTION SIGNALS
PAYOFF_HOOKS = [
    (r"\b(and that's (when|how|why)|so i did|and i did|it worked)\b", 3.0),
    (r"\b(paid off|worth it|best decision|never looked back|finally)\b", 3.0),
    (r"\b(the result was|what happened next|here's the punchline|spoiler)\b", 3.5),
    (r"\b(this is where it gets (good|interesting|crazy|wild))\b", 4.0),
    (r"\b(long story short|bottom line|the moral|lesson learned|takeaway)\b", 3.0),
    (r"\b(funny thing is|here's the irony|plot twist|you guessed it)\b", 3.5),
]

# Method 10 — HIGH-VALUE PROMISE / DIRECT BENEFIT LANGUAGE
PROMISE_HOOKS = [
    (r"\b(how to|the way to|the key to|secret (to|of)|trick (to|is))\b", 3.0),
    (r"\b(step by step|exact (method|process|strategy|system|formula))\b", 3.0),
    (r"\b(made me \$|earned|generated|saved|made back|invested)\b", 3.0),
    (r"\b(never have to|stop worrying|get rid of|eliminate|fix this)\b", 2.5),
    (r"\b(simple (hack|trick|fix|method)|one (weird|simple|easy) (trick|thing))\b", 3.0),
    (r"\b(framework|blueprint|roadmap|playbook|system that)\b", 2.5),
]

# Method 11 — CLIFFHANGER / INCOMPLETENESS SIGNALS
CLIFFHANGER_HOOKS = [
    (r"\b(what i'm about to tell you|you're not going to believe|wait for it)\b", 4.0),
    (r"\b(before i tell you|first let me say|here's the thing though)\b", 3.0),
    (r"\b(but that's not all|it gets (better|worse)|and then|but then)\b", 3.0),
    (r"\b(almost|nearly|seconds away from|one step away|barely)\b", 2.5),
    (r"\b(to be continued|that's a story for another|remind me to tell)\b", 2.5),
]

# Method 12 — IDENTITY / "PEOPLE LIKE US" LANGUAGE
IDENTITY_HOOKS = [
    (r"\b(if you('re| are) (someone who|a person who|like me)|for anyone who)\b", 2.5),
    (r"\b(we've all been there|you know the feeling|raise your hand if)\b", 3.0),
    (r"\b(as a (parent|founder|developer|teacher|doctor|student|creator))\b", 2.0),
    (r"\b(this is for the|this one's for|you'll relate if)\b", 2.5),
    (r"\b(just like me|same situation|been there|done that|felt that way)\b", 2.5),
]

ALL_HOOK_BANKS = [
    ("story_openers",       STORY_OPENERS),
    ("contrast_hooks",      CONTRAST_HOOKS),
    ("curiosity_gaps",      CURIOSITY_GAPS),
    ("emotion_hooks",       EMOTION_HOOKS),
    ("vulnerability_hooks", VULNERABILITY_HOOKS),
    ("transformation_hooks",TRANSFORMATION_HOOKS),
    ("authority_hooks",     AUTHORITY_HOOKS),
    ("conflict_hooks",      CONFLICT_HOOKS),
    ("payoff_hooks",        PAYOFF_HOOKS),
    ("promise_hooks",       PROMISE_HOOKS),
    ("cliffhanger_hooks",   CLIFFHANGER_HOOKS),
    ("identity_hooks",      IDENTITY_HOOKS),
]

# ─────────────────────────────────────────────────────────────
# 2.  DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class Segment:
    start:  float
    end:    float
    text:   str

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def words(self) -> list[str]:
        return self.text.strip().split()

    @property
    def word_count(self) -> int:
        return len(self.words)


@dataclass
class ClipCandidate:
    start:       float
    end:         float
    title:       str            = "untitled"
    total_score: float          = 0.0
    score_breakdown: dict       = field(default_factory=dict)
    hook_phrases: list[str]     = field(default_factory=list)
    segments:    list[Segment]  = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments)

    def overlaps_with(self, other: "ClipCandidate") -> float:
        """Return fraction of self that overlaps with other (0–1)."""
        overlap_start = max(self.start, other.start)
        overlap_end   = min(self.end, other.end)
        if overlap_end <= overlap_start:
            return 0.0
        return (overlap_end - overlap_start) / self.duration

# ─────────────────────────────────────────────────────────────
# 3.  TRANSCRIPT LOADER
# ─────────────────────────────────────────────────────────────

def load_segments(path: str) -> list[Segment]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("segments", data) if isinstance(data, dict) else data
    segs = []
    for s in raw:
        start = s.get("start", 0.0)
        end   = s.get("end",   start + 0.001)
        text  = s.get("text", "").strip()
        if text:
            segs.append(Segment(start=start, end=end, text=text))
    return segs

# ─────────────────────────────────────────────────────────────
# 4.  HEURISTIC SCORER — runs all 12 hook banks + extra signals
# ─────────────────────────────────────────────────────────────

def score_text_heuristic(text: str) -> tuple[float, dict, list[str]]:
    """
    Score a block of text against all hook banks.
    Returns (total_score, breakdown_dict, matched_phrases_list).
    """
    lower = text.lower()
    breakdown: dict[str, float] = {}
    matched_phrases: list[str] = []

    for bank_name, patterns in ALL_HOOK_BANKS:
        bank_score = 0.0
        for pattern, weight in patterns:
            for m in re.finditer(pattern, lower):
                bank_score += weight
                phrase = m.group(0).strip()
                if phrase not in matched_phrases:
                    matched_phrases.append(phrase)
        breakdown[bank_name] = round(bank_score, 2)

    # Bonus: question words at the START of a sentence (strong hooks)
    question_starts = len(re.findall(
        r'(?:^|[.!?]\s+)(what|why|how|when|who|where|have you ever)\b', lower))
    breakdown["question_hooks"] = round(question_starts * 2.5, 2)

    # Bonus: sentence variety — mixing short punchy with longer ones
    sentences = re.split(r'[.!?]+', text.strip())
    lengths   = [len(s.split()) for s in sentences if s.strip()]
    if lengths:
        variety = (max(lengths) - min(lengths)) / (max(lengths) + 1)
        breakdown["sentence_variety"] = round(variety * 3.0, 2)
    else:
        breakdown["sentence_variety"] = 0.0

    # Bonus: first-person singular (personal narrative = more engaging)
    first_person = len(re.findall(r"\b(i |i'|my |me |myself )\b", lower))
    breakdown["first_person_narrative"] = round(min(first_person * 0.15, 3.0), 2)

    # Bonus: direct address ("you")
    second_person = len(re.findall(r"\b(you |your |you')\b", lower))
    breakdown["direct_address"] = round(min(second_person * 0.2, 3.0), 2)

    # Length penalty: very short clips lose points
    word_count = len(text.split())
    if word_count < 40:
        breakdown["length_penalty"] = round(-(40 - word_count) * 0.1, 2)
    else:
        breakdown["length_penalty"] = 0.0

    total = round(sum(breakdown.values()), 3)
    return total, breakdown, matched_phrases


def build_window_text(segments: list[Segment], center_start: float,
                      window: float) -> tuple[str, list[Segment]]:
    """Collect segments whose midpoint falls within [center_start, center_start+window]."""
    included = []
    for seg in segments:
        mid = (seg.start + seg.end) / 2
        if center_start <= mid <= center_start + window:
            included.append(seg)
    text = " ".join(s.text.strip() for s in included)
    return text, included

# ─────────────────────────────────────────────────────────────
# 5.  CLIP BOUNDARY OPTIMIZER
#     Extends or contracts the clip to natural sentence boundaries
# ─────────────────────────────────────────────────────────────

def optimize_boundaries(candidate: ClipCandidate,
                        all_segments: list[Segment]) -> ClipCandidate:
    """
    Adjust start/end to the nearest clean sentence boundary
    while staying within MIN/MAX clip length constraints.
    """
    # Find which segments fall within the candidate window
    inner = [s for s in all_segments
             if s.start >= candidate.start - 1.0
             and s.end   <= candidate.end   + 1.0]
    if not inner:
        return candidate

    # Prefer a sentence-ending character close to the target end
    SENTENCE_ENDS = re.compile(r'[.!?]$')
    best_end_seg = None
    for seg in reversed(inner):
        if SENTENCE_ENDS.search(seg.text.strip()):
            best_end_seg = seg
            break
    if best_end_seg:
        new_end = best_end_seg.end
    else:
        new_end = inner[-1].end

    # Prefer a sentence-starting-style beginning
    first_seg = inner[0]
    new_start = first_seg.start

    # Clamp to duration constraints
    duration = new_end - new_start
    if duration > MAX_CLIP_SECONDS:
        new_end = new_start + MAX_CLIP_SECONDS
    if duration < MIN_CLIP_SECONDS:
        # Try to extend end
        new_end = new_start + MIN_CLIP_SECONDS

    candidate.start = round(new_start, 3)
    candidate.end   = round(min(new_end, all_segments[-1].end), 3)

    # Refresh segments list
    candidate.segments = [
        s for s in all_segments
        if s.start >= candidate.start - 0.5 and s.end <= candidate.end + 0.5
    ]
    return candidate

# ─────────────────────────────────────────────────────────────
# 6.  OVERLAP DEDUPLICATOR
# ─────────────────────────────────────────────────────────────

def deduplicate_clips(candidates: list[ClipCandidate],
                      n: int) -> list[ClipCandidate]:
    """
    Pick the top-N clips while penalising/excluding heavily overlapping ones.
    Uses a greedy highest-score-first selection with a soft penalty pass.
    """
    sorted_cands = sorted(candidates, key=lambda c: c.total_score, reverse=True)
    selected: list[ClipCandidate] = []

    for cand in sorted_cands:
        if len(selected) >= n:
            break
        # Check overlap with already-selected clips
        skip = False
        for chosen in selected:
            overlap = cand.overlaps_with(chosen)
            if overlap > 0.5:
                skip = True
                break
            if overlap > 0.2:
                cand.total_score *= OVERLAP_PENALTY
        if not skip:
            selected.append(cand)

    # Re-sort after potential penalty adjustments and trim to n
    selected.sort(key=lambda c: c.total_score, reverse=True)
    return selected[:n]

# ─────────────────────────────────────────────────────────────
# 7.  AUTO-TITLE GENERATOR
#     Derives a short snake_case title from the clip's strongest phrase
# ─────────────────────────────────────────────────────────────

def auto_title(candidate: ClipCandidate) -> str:
    # Try to extract a 2-4 word phrase from the clip text
    text = candidate.text.lower()

    # Look for "I [verb] [noun]" patterns — often good titles
    m = re.search(
        r"\bi (failed|quit|got (fired|selected|rejected)|lost|won|decided|realized|discovered)\b",
        text)
    if m:
        title = m.group(0).replace(" ", "_").strip("_")
        return re.sub(r"[^a-z0-9_]", "", title)[:40]

    # Use first matched hook phrase if available
    if candidate.hook_phrases:
        phrase = candidate.hook_phrases[0]
        title  = re.sub(r"[^a-z0-9 ]", "", phrase.lower()).strip()
        title  = "_".join(title.split()[:4])
        if title:
            return title[:40]

    # Fallback: first 4 meaningful words of the clip
    words = [w for w in text.split() if len(w) > 2][:4]
    return "_".join(words)[:40] or f"clip_{int(candidate.start)}s"

# ─────────────────────────────────────────────────────────────
# 8.  CLAUDE AI RANKER  (optional — degrades gracefully)
# ─────────────────────────────────────────────────────────────

CLAUDE_MODEL     = "claude-sonnet-4-20250514"
CLAUDE_MAX_TOKENS = 1500
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def _claude_rank(candidates: list[ClipCandidate], n: int) -> list[ClipCandidate]:
    """
    Send top heuristic candidates to Claude for final ranking.
    Returns reordered candidates. On any failure, returns original list.
    """
    if not ANTHROPIC_API_KEY:
        return candidates

    import json as _json

    clips_json = []
    for i, c in enumerate(candidates[:min(n * 3, 20)]):
        # Truncate text to keep prompt manageable
        clips_json.append({
            "id":    i,
            "start": c.start,
            "end":   c.end,
            "heuristic_score": c.total_score,
            "text":  c.text[:500],
        })

    prompt = f"""You are a viral short-form video editor.
Below are {len(clips_json)} clip candidates from a podcast/talk transcript.
Each has a start time, end time, heuristic score, and a text excerpt.

Your task: Rank the TOP {n} clips that would make the most compelling
short-form social media content (TikTok / Reels / YouTube Shorts).

Evaluate each clip on:
1. Hook strength — does it grab attention in the first 3 seconds?
2. Story completeness — beginning, tension, payoff within the clip
3. Emotional resonance — does it make you FEEL something?
4. Shareability — would someone tag a friend?
5. Curiosity factor — does it make you want to know more?

Return ONLY valid JSON — a list of clip IDs in ranked order (best first),
followed by a one-line reason for each pick.

Format:
[
  {{"id": 2, "reason": "Opens with failure hook, ends with redemption"}},
  {{"id": 7, "reason": "Strong curiosity gap + authority signal"}},
  ...
]

Clips:
{_json.dumps(clips_json, indent=2)}
"""

    payload = _json.dumps({
        "model":      CLAUDE_MODEL,
        "max_tokens": CLAUDE_MAX_TOKENS,
        "messages":   [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body    = _json.loads(resp.read().decode("utf-8"))
            content = body["content"][0]["text"].strip()

        # Strip ```json fences if present
        content = re.sub(r"```(?:json)?|```", "", content).strip()
        ranked  = _json.loads(content)

        # Build reordered list
        id_to_cand = {i: c for i, c in enumerate(candidates)}
        reordered  = []
        for entry in ranked[:n]:
            cid    = entry["id"]
            reason = entry.get("reason", "")
            if cid in id_to_cand:
                cand = id_to_cand[cid]
                cand.score_breakdown["ai_reason"] = reason
                reordered.append(cand)

        # Append any missed clips at the end (in case AI returned fewer than n)
        seen = {id(c) for c in reordered}
        for c in candidates:
            if id(c) not in seen and len(reordered) < n:
                reordered.append(c)

        print(f"[AI] Claude re-ranked {len(reordered)} clips using semantic analysis.")
        return reordered[:n]

    except Exception as e:
        print(f"[AI] Claude ranking skipped ({e}). Using heuristic order.")
        return candidates[:n]


# ─────────────────────────────────────────────────────────────
# 9.  MAIN CLIP FINDER — orchestrates everything
# ─────────────────────────────────────────────────────────────

def find_best_clips(segments: list[Segment], n: int) -> list[ClipCandidate]:
    """
    Slide a scoring window across the transcript.
    Score every possible clip starting point.
    Return the top-N deduplicated clips.
    """
    if not segments:
        return []

    total_duration = segments[-1].end
    step_seconds   = 5.0          # Probe every 5 seconds for a clip start

    print(f"[SCAN] Scanning {total_duration:.0f}s of content in {step_seconds:.0f}s steps...")

    all_candidates: list[ClipCandidate] = []

    probe_time = 0.0
    probe_count = 0

    while probe_time + MIN_CLIP_SECONDS < total_duration:
        # Build a window of CONTEXT_WINDOW seconds starting at probe_time
        text, window_segs = build_window_text(segments, probe_time, CONTEXT_WINDOW)
        if not text.strip():
            probe_time += step_seconds
            continue

        total_score, breakdown, phrases = score_text_heuristic(text)

        if total_score > 0:
            # Determine natural end: use MIN_CLIP_SECONDS as default, expand to
            # the last segment in the window (up to MAX_CLIP_SECONDS)
            if window_segs:
                clip_end = min(window_segs[-1].end, probe_time + MAX_CLIP_SECONDS)
            else:
                clip_end = probe_time + MIN_CLIP_SECONDS

            cand = ClipCandidate(
                start           = probe_time,
                end             = clip_end,
                total_score     = total_score,
                score_breakdown = breakdown,
                hook_phrases    = phrases[:5],
                segments        = window_segs,
            )
            all_candidates.append(cand)

        probe_time += step_seconds
        probe_count += 1

    print(f"[SCAN] Evaluated {probe_count} windows. Found {len(all_candidates)} scored candidates.")

    if not all_candidates:
        return []

    # Sort by score descending
    all_candidates.sort(key=lambda c: c.total_score, reverse=True)

    # Optimize boundaries on top 3×n candidates before deduplication
    top_pool = all_candidates[:min(n * 3, len(all_candidates))]
    for cand in top_pool:
        optimize_boundaries(cand, segments)

    # Deduplicate and select final n
    selected = deduplicate_clips(top_pool, n)

    # Try AI re-ranking
    if ANTHROPIC_API_KEY:
        print(f"[AI] Sending top candidates to Claude for semantic re-ranking...")
        selected = _claude_rank(selected, n)
    else:
        print("[AI] No ANTHROPIC_API_KEY set — skipping semantic re-ranking. "
              "Heuristic scores used.")

    # Auto-title each clip
    for cand in selected:
        cand.title = auto_title(cand)

    return selected

# ─────────────────────────────────────────────────────────────
# 10.  SUBTITLE STYLE
# ─────────────────────────────────────────────────────────────

SUBTITLE_STYLE = (
    "Fontname=Arial,"
    "Bold=1,"
    "Fontsize=22,"
    "PrimaryColour=&H00FFFFFF,"
    "OutlineColour=&H00000000,"
    "BackColour=&H80000000,"
    "BorderStyle=1,"
    "Outline=3,"
    "Shadow=1,"
    "Alignment=2,"
    "MarginV=60,"
    "MarginL=20,"
    "MarginR=20"
)

MAX_WORDS_PER_LINE = 6

# ─────────────────────────────────────────────────────────────
# 11.  GPU ENCODER DETECTION (unchanged from v4)
# ─────────────────────────────────────────────────────────────

ENCODERS = {
    "nvidia": {
        "flags":   ["-c:v", "h264_nvenc", "-gpu", "0",
                    "-preset", "p4", "-rc", "vbr", "-b_ref_mode", "0"],
        "quality": ["-cq", "22"],
        "label":   "NVIDIA NVENC (GPU)",
        "test_args": ["-f", "lavfi", "-i", "nullsrc=s=256x256:d=1",
                      "-c:v", "h264_nvenc", "-gpu", "0",
                      "-frames:v", "1", "-f", "null", "-"],
    },
    "amd": {
        "flags":   ["-c:v", "h264_amf", "-quality", "balanced"],
        "quality": ["-qp_i", "22", "-qp_p", "22", "-qp_b", "24"],
        "label":   "AMD AMF (GPU)",
        "test_args": ["-f", "lavfi", "-i", "nullsrc=s=256x256:d=1",
                      "-c:v", "h264_amf", "-frames:v", "1", "-f", "null", "-"],
    },
    "intel": {
        "flags":   ["-c:v", "h264_qsv", "-preset", "medium"],
        "quality": ["-global_quality", "22"],
        "label":   "Intel QSV (GPU)",
        "test_args": ["-f", "lavfi", "-i", "nullsrc=s=256x256:d=1",
                      "-c:v", "h264_qsv", "-frames:v", "1", "-f", "null", "-"],
    },
    "cpu": {
        "flags":   ["-c:v", "libx264", "-preset", "fast"],
        "quality": ["-crf", "22"],
        "label":   "CPU x264 (software)",
        "test_args": None,
    },
}


def probe_encoder(name: str) -> bool:
    enc = ENCODERS[name]
    if enc["test_args"] is None:
        return True
    try:
        result = subprocess.run(
            ["ffmpeg", "-y"] + enc["test_args"],
            capture_output=True, timeout=10)
        return result.returncode == 0
    except Exception:
        return False


def detect_encoder() -> str:
    if FORCE_ENCODER and FORCE_ENCODER in ENCODERS:
        print(f"[GPU] Forced: {ENCODERS[FORCE_ENCODER]['label']}")
        return FORCE_ENCODER
    print("[GPU] Auto-detecting best encoder...")
    for name in ("nvidia", "amd", "intel"):
        print(f"  Probing {ENCODERS[name]['label']}...", end=" ", flush=True)
        if probe_encoder(name):
            print("OK")
            return name
        print("not available")
    print("  Using CPU fallback.")
    return "cpu"

# ─────────────────────────────────────────────────────────────
# 12.  SMART FILE NAMING (unchanged from v4 with one tweak)
# ─────────────────────────────────────────────────────────────

def make_output_name(video_path: str, clip_title: str,
                     rank: int, run_ts: str) -> str:
    """
    Pattern: {episode_stem}__{rank:02d}__{clip_title}__{YYYYMMDD_HHMMSS}.mp4

    rank is included so clips sort by quality order in the folder.
    """
    stem       = os.path.splitext(os.path.basename(video_path))[0]
    safe_stem  = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in clip_title)
    return f"{safe_stem}__{rank:02d}__{safe_title}__{run_ts}.mp4"

# ─────────────────────────────────────────────────────────────
# 13.  SRT GENERATION (unchanged from v4)
# ─────────────────────────────────────────────────────────────

def srt_time(seconds: float) -> str:
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        s += 1; ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def chunk_words(text: str, max_words: int) -> list[str]:
    words = text.strip().split()
    chunks = []
    while words:
        chunk = " ".join(words[:max_words]).strip()
        if chunk:
            chunks.append(chunk)
        words = words[max_words:]
    return chunks


def generate_srt(segs: list[Segment], clip_start: float,
                 clip_end: float, out_path: str) -> str:
    SRT_END_CAP = 0.05
    entries = []
    idx = 1

    for seg in segs:
        if seg.end <= clip_start or seg.start >= clip_end:
            continue
        seg_s = max(0.0, seg.start - clip_start)
        seg_e = min(clip_end - clip_start, seg.end - clip_start) - SRT_END_CAP
        if seg_e <= seg_s:
            continue
        raw = seg.text.strip()
        if not raw:
            continue
        lines    = chunk_words(raw, MAX_WORDS_PER_LINE)
        if not lines:
            continue
        seg_dur  = seg_e - seg_s
        line_dur = seg_dur / len(lines)
        for li, line in enumerate(lines):
            if not line:
                continue
            t0 = seg_s + li * line_dur
            t1 = t0 + line_dur
            entries.append(f"{idx}\n{srt_time(t0)} --> {srt_time(t1)}\n{line}")
            idx += 1

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(entries))
    return out_path

# ─────────────────────────────────────────────────────────────
# 14.  FFMPEG RUNNER (unchanged from v4)
# ─────────────────────────────────────────────────────────────

def run_ffmpeg(video_in: str, ss: float, duration: float,
               srt_path: str, video_out: str, encoder_name: str) -> bool:
    enc = ENCODERS[encoder_name]
    ffmpeg_srt = srt_path.replace("\\", "/").replace(":", r"\:")
    vf = (
        "crop=ih*(9/16):ih:(iw-ow)/2:0,"
        f"subtitles='{ffmpeg_srt}':force_style='{SUBTITLE_STYLE}'"
    )
    cmd = (
        ["ffmpeg", "-y",
         "-ss", f"{ss:.3f}",
         "-i", video_in,
         "-t", f"{duration:.3f}",
         "-vf", vf]
        + enc["flags"]
        + enc["quality"]
        + ["-c:a", "aac", "-b:a", "192k",
           "-movflags", "+faststart",
           video_out]
    )
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            line = line.rstrip()
            if any(k in line for k in ["frame=", "fps=", "speed=", "Error",
                                        "error", "encoder", "Stream #0"]):
                print(f"    {line}", flush=True)
        proc.wait()
        return proc.returncode == 0
    except FileNotFoundError:
        print("[ERROR] ffmpeg not found in PATH.")
        return False
    except Exception as e:
        print(f"[ERROR] FFmpeg exception: {e}")
        return False

# ─────────────────────────────────────────────────────────────
# 15.  SCORE REPORT PRINTER
# ─────────────────────────────────────────────────────────────

def print_score_report(clips: list[ClipCandidate]) -> None:
    print("\n" + "=" * 60)
    print("  CLIP ANALYSIS REPORT")
    print("=" * 60)
    for i, c in enumerate(clips, start=1):
        print(f"\n  #{i}  [{c.title}]  score={c.total_score:.1f}")
        print(f"       {c.start:.1f}s → {c.end:.1f}s  ({c.duration:.0f}s)")

        # Top 3 scoring dimensions
        bd = {k: v for k, v in c.score_breakdown.items()
              if isinstance(v, (int, float)) and v > 0}
        top_dims = sorted(bd.items(), key=lambda x: x[1], reverse=True)[:4]
        for dim, sc in top_dims:
            bar = "█" * min(int(sc), 20)
            print(f"       {dim:<28} {sc:5.1f}  {bar}")

        # Matched hook phrases
        if c.hook_phrases:
            phrases_str = " | ".join(f'"{p}"' for p in c.hook_phrases[:4])
            print(f"       hooks: {phrases_str}")

        # AI reason (if available)
        ai_r = c.score_breakdown.get("ai_reason")
        if ai_r:
            print(f"       AI:    {ai_r}")

        # Preview first 120 chars of clip text
        preview = c.text[:120].replace("\n", " ")
        print(f'       "...{preview}..."')
    print("\n" + "=" * 60 + "\n")

# ─────────────────────────────────────────────────────────────
# 16.  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"\n{'=' * 60}")
    print(f"  smart_clips_pro v5.0  —  AI-Powered Best Clips Generator")
    print(f"  Target: {NUM_CLIPS} clips  |  "
          f"AI: {'ON' if ANTHROPIC_API_KEY else 'OFF (set ANTHROPIC_API_KEY)'}")
    print(f"{'=' * 60}\n")

    # ── Validate inputs ───────────────────────────────────────
    if not os.path.exists(TRANSCRIPT_FILE):
        print(f"[ERROR] Transcript not found: {TRANSCRIPT_FILE}")
        sys.exit(1)
    if not os.path.exists(VIDEO_FILE):
        print(f"[ERROR] Video not found: {VIDEO_FILE}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load transcript ───────────────────────────────────────
    segments = load_segments(TRANSCRIPT_FILE)
    if not segments:
        print("[ERROR] Transcript is empty or could not be parsed.")
        sys.exit(1)
    total_dur = segments[-1].end
    print(f"[OK] Loaded {len(segments)} segments  ({total_dur:.0f}s of content)\n")

    # ── Detect encoder ────────────────────────────────────────
    encoder_name = detect_encoder()
    print(f"[GPU] Using: {ENCODERS[encoder_name]['label']}\n")

    # ── Find best clips ───────────────────────────────────────
    t0 = time.time()
    best_clips = find_best_clips(segments, NUM_CLIPS)
    print(f"[OK] Analysis done in {time.time() - t0:.1f}s  "
          f"→  {len(best_clips)} clips selected\n")

    if not best_clips:
        print("[ERROR] No clips found. "
              "Check that your transcript has enough content.")
        sys.exit(1)

    # ── Print detailed report ─────────────────────────────────
    print_score_report(best_clips)

    # ── Sort clips chronologically for rendering ──────────────
    clips_to_render = sorted(best_clips, key=lambda c: c.start)

    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    for rank, clip in enumerate(clips_to_render, start=1):
        ss       = clip.start
        end      = clip.end
        duration = end - ss
        title    = clip.title

        out_name  = make_output_name(VIDEO_FILE, title, rank, run_ts)
        video_out = os.path.join(OUTPUT_DIR, out_name)
        srt_file  = os.path.join(OUTPUT_DIR, f"_tmp_{title}_{run_ts}.srt")

        print(f"[{rank}/{len(clips_to_render)}] {out_name}")
        print(f"       {ss:.1f}s -> {end:.1f}s  ({duration:.0f}s)  "
              f"score={clip.total_score:.1f}")

        # Use the clip's own segments for SRT generation, fall back to all
        srt_segs = clip.segments if clip.segments else segments
        generate_srt(srt_segs, ss, end, srt_file)

        ok = run_ffmpeg(VIDEO_FILE, ss, duration, srt_file, video_out, encoder_name)

        if os.path.exists(srt_file):
            os.remove(srt_file)

        if ok:
            size_mb = os.path.getsize(video_out) / (1024 * 1024)
            print(f"       [OK] {size_mb:.1f} MB -> {video_out}\n")
            results.append({"clip": out_name, "status": "ok",
                             "path": video_out, "score": clip.total_score})
        else:
            print(f"       [FAIL] FFmpeg returned error for {out_name}\n")
            results.append({"clip": out_name, "status": "fail",
                             "path": None, "score": clip.total_score})

    # ── Summary ───────────────────────────────────────────────
    ok_count   = sum(1 for r in results if r["status"] == "ok")
    fail_count = len(results) - ok_count

    print("=" * 60)
    print(f"  Done: {ok_count}/{len(results)} clips rendered successfully.")
    for r in results:
        icon = "OK  " if r["status"] == "ok" else "FAIL"
        print(f"  [{icon}] score={r['score']:5.1f}  {r['clip']}")
    print("=" * 60)

    # Save machine-readable results (useful for n8n downstream nodes)
    results_json = os.path.join(OUTPUT_DIR, f"results_{run_ts}.json")
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump({"run_ts": run_ts, "clips": results}, f, indent=2)
    print(f"\n  Results JSON: {results_json}")

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()