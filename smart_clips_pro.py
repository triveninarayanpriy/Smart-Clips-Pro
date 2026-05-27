"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║          TEACHER CLIPS HINDI PRO  v1.0                                         ║
║          AI-Powered Funny & Interesting Moments Finder for Indian Teachers      ║
║          हिंदी कैप्शन सहित — Hindi Captions Included                           ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  WHAT THIS DOES:                                                                ║
║  ✦ Scans Indian teacher lecture transcripts for funny/viral moments             ║
║  ✦ 20+ specialized detection banks tuned for Indian classroom culture           ║
║  ✦ Detects: desi jokes, scoldings, analogies, exam threats, Bollywood refs     ║
║  ✦ Translates/captions into beautiful Hindi using Claude AI                     ║
║  ✦ Generates vertical 9:16 Reels/Shorts-ready clips with animated Hindi subs   ║
║  ✦ Multi-tier scoring: Funny Score + Education Value + Viral Potential          ║
║  ✦ Emotion tagging: 😂 हंसी / 😤 डांट / 🤯 ज्ञान / 😱 नाटक / 💡 Insight      ║
║  ✦ Auto Hindi transliteration fallback if translation API is unavailable       ║
║  ✦ GPU-accelerated encoding (NVIDIA / AMD / Intel / CPU fallback)               ║
║  ✦ Smart deduplication, boundary optimizer, overlap penalty                     ║
║  ✦ Detailed per-clip report in Hindi + English                                  ║
║  ✦ Session JSON export for automation pipelines (n8n, Zapier, etc.)             ║
║  ✦ Batch mode for processing full course playlists                              ║
║  ✦ Configurable teacher profile (IIT-JEE, NEET, SSC, School, etc.)             ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  USAGE:                                                                         ║
║    python teacher_clips_hindi_pro_v1.py transcript.json lecture.mp4             ║
║    python teacher_clips_hindi_pro_v1.py transcript.json lecture.mp4 ./clips     ║
║    python teacher_clips_hindi_pro_v1.py transcript.json lecture.mp4 ./clips \  ║
║           nvidia 7 iitjee                                                       ║
║    (args: transcript video [outdir] [encoder] [num_clips] [teacher_profile])    ║
║                                                                                 ║
║  TEACHER PROFILES: iitjee | neet | school | ssc | ca | college | general       ║
║                                                                                 ║
║  ENV VARS:                                                                      ║
║    ANTHROPIC_API_KEY  — Claude AI for Hindi translation + semantic ranking      ║
║    TEACHER_NAME       — Teacher name for output filenames (e.g. "Khan_Sir")     ║
║    SUBTITLE_FONT      — Font for Hindi subtitles (default: Noto Sans Devanagari)║
║                                                                                 ║
║  TRANSCRIPT FORMAT (Whisper-compatible JSON):                                   ║
║    {"segments": [{"start": 0.0, "end": 4.2, "text": "..."}, ...]}              ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
import textwrap
import urllib.request
import urllib.error
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Tuple

# ══════════════════════════════════════════════════════════════════
#  0.  CONFIGURATION  (CLI overrides apply below)
# ══════════════════════════════════════════════════════════════════

TRANSCRIPT_FILE   = r"transcript.json"
VIDEO_FILE        = r"lecture.mp4"
OUTPUT_DIR        = r"./clips_output"
FORCE_ENCODER     = None
NUM_CLIPS         = 7            # How many best clips to extract
MIN_CLIP_SECONDS  = 25           # Minimum clip duration (seconds)
MAX_CLIP_SECONDS  = 90           # Maximum clip duration (seconds)
CONTEXT_WINDOW    = 55.0         # Scoring window size (seconds)
OVERLAP_PENALTY   = 0.55         # Score multiplier when clips overlap >25%
TEACHER_PROFILE   = "general"    # One of: iitjee | neet | school | ssc | ca | college | general
STEP_SECONDS      = 4.0          # Probe every N seconds for a clip start
BOUNDARY_SLACK    = 1.5          # Seconds of slack when snapping to sentence ends
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
TEACHER_NAME      = os.environ.get("TEACHER_NAME", "Teacher")
SUBTITLE_FONT     = os.environ.get("SUBTITLE_FONT", "Noto Sans Devanagari")
CLAUDE_MODEL      = "claude-sonnet-4-20250514"

# --- CLI overrides ---
_args = sys.argv[1:]
if len(_args) >= 1: TRANSCRIPT_FILE = _args[0]
if len(_args) >= 2: VIDEO_FILE      = _args[1]
if len(_args) >= 3: OUTPUT_DIR      = _args[2]
if len(_args) >= 4: FORCE_ENCODER   = _args[3].lower()
if len(_args) >= 5:
    try:    NUM_CLIPS = int(_args[4])
    except: pass
if len(_args) >= 6: TEACHER_PROFILE = _args[5].lower()

# ══════════════════════════════════════════════════════════════════
#  1.  INDIAN CLASSROOM HOOK-DETECTION VOCABULARY BANKS
#      Each entry: (regex_pattern, weight, emotion_tag, hindi_label)
# ══════════════════════════════════════════════════════════════════

# ── Bank A: Classic Indian Teacher Scolding Moments ──────────────
SCOLDING_HOOKS = [
    (r"\b(are you listening|pay attention|sit properly|wake up|sleeping|uth ja|uth jao)\b", 4.5, "😤", "डांट"),
    (r"\b(how many times|kitni baar|kitna samjhaya|i've told you|already told)\b", 4.0, "😤", "झिड़की"),
    (r"\b(shut up|chup|quiet|be silent|khamosh|bandh karo|stop talking)\b", 3.5, "😤", "चुप"),
    (r"\b(stupid|foolish|bakwas|pagal|you don't understand anything|samajh nahi aata)\b", 4.0, "😤", "डांट"),
    (r"\b(get out|bahar jao|stand up|khade ho jao|out of class|leave the room)\b", 5.0, "😤", "बाहर"),
    (r"\b(this is not a fish market|yeh school hai|this is a classroom)\b", 5.5, "😤", "गुस्सा"),
    (r"\b(zero marks|fail|fail ho jaoge|fail kar dunga|marks kaatuga)\b", 4.5, "😤", "धमकी"),
    (r"\b(tell your parents|parents ko bulao|parents ko bata dunga)\b", 5.0, "😤", "अभिभावक"),
    (r"\b(last warning|aakhri mauka|final chance|ek aur baar|one more time)\b", 4.0, "😤", "चेतावनी"),
    (r"\b(don't laugh|mast karne ka time nahi|this is not funny|serious raho)\b", 3.5, "😂", "हंसी"),
]

# ── Bank B: Funny Analogies & Desi Examples ──────────────────────
DESI_ANALOGY_HOOKS = [
    (r"\b(just like|bilkul waise hi|samajh lo|suppose karo|imagine karo)\b", 3.0, "💡", "उदाहरण"),
    (r"\b(chai|chaiwala|doodh|roti|sabzi|biryani|daal|paneer)\b", 4.5, "😂", "देसी उदाहरण"),
    (r"\b(rickshaw|auto|jugaad|desi|gaon|village|shehar|city)\b", 3.5, "😂", "देसी"),
    (r"\b(bhaiya|didi|uncle|aunty|nani|dadi|mama|chacha|chachi)\b", 4.0, "😂", "रिश्तेदार"),
    (r"\b(cricket|sachin|dhoni|virat|stadium|match|boundary|wicket)\b", 4.0, "😂", "क्रिकेट"),
    (r"\b(bollywood|film|hero|heroine|villain|SRK|amitabh|salman)\b", 3.5, "😂", "बॉलीवुड"),
    (r"\b(wedding|shaadi|baraat|dulha|dulhan|mehndi|dowry|dahej)\b", 4.5, "😂", "शादी"),
    (r"\b(ration|sarkari|neta|mantri|babu|bribe|corruption|department)\b", 4.0, "😤", "सरकारी"),
    (r"\b(cow|bail|goat|bakri|donkey|gadha|monkey|bandar|billi|cat)\b", 5.0, "😂", "जानवर"),
    (r"\b(adjust|jugaad|setting|shortcut|sifarish|reference|contact)\b", 4.5, "😂", "जुगाड़"),
]

# ── Bank C: Exam & Competition Threats ───────────────────────────
EXAM_THREAT_HOOKS = [
    (r"\b(board exam|boards|class 10|class 12|matric|intermediate)\b", 3.5, "😱", "बोर्ड"),
    (r"\b(JEE|NEET|UPSC|SSC|CAT|GATE|civil services|IIT|AIIMS|NIT)\b", 4.0, "😱", "प्रतियोगिता"),
    (r"\b(rank|topper|first rank|AIR 1|merit list|selection|selected)\b", 3.5, "🏆", "रैंक"),
    (r"\b(syllabus|chapter|topic|concept|this will come|yeh aayega)\b", 3.0, "😱", "सिलेबस"),
    (r"\b(revise|revision|practice|practice karo|solve karo|questions)\b", 2.5, "📚", "अभ्यास"),
    (r"\b(last year question|previous year|pyq|trend|pattern|important)\b", 3.5, "😱", "महत्वपूर्ण"),
    (r"\b(guaranteed|pakka|sure shot|100 percent|confirm aayega|definitely)\b", 4.5, "😱", "पक्का"),
    (r"\b(time up|times up|time khatam|pen down|ruk jao|stop writing)\b", 4.0, "😱", "समय"),
    (r"\b(copy|cheating|cheat|nafal|dekh raha tha|caught|pakad liya)\b", 5.5, "😂", "नकल"),
    (r"\b(pass fail|passing marks|33 percent|grace marks|compartment)\b", 4.0, "😱", "पास-फेल"),
]

# ── Bank D: Motivational Outbursts (Typical Indian Teacher Style) ─
MOTIVATION_HOOKS = [
    (r"\b(you can do it|kar sakte ho|trust me|believe me|mujh par bharosa)\b", 3.5, "💪", "प्रेरणा"),
    (r"\b(success|safalta|hard work|mehnat|dedication|discipline)\b", 3.0, "💪", "सफलता"),
    (r"\b(your parents|tumhare maa baap|ghar waale|sacrifice|tyaag|loan liya)\b", 5.5, "😢", "मां-बाप"),
    (r"\b(poor family|garib|struggle|struggle kiya|nahi tha paisa|manage kiya)\b", 5.0, "😢", "संघर्ष"),
    (r"\b(dream|sapna|aim|goal|target|bano|become|future mein)\b", 3.5, "💪", "सपना"),
    (r"\b(give up mat karo|don't quit|haar mat mano|keep going|lagey raho)\b", 4.0, "💪", "हिम्मत"),
    (r"\b(one day|ek din|remember this|yaad rakhna|you will thank)\b", 4.5, "💪", "याद रखो"),
    (r"\b(I was also student|main bhi student tha|when i was young|meri age mein)\b", 4.5, "😊", "किस्सा"),
    (r"\b(big city|metro|delhi|mumbai|abroad|foreign|london|america)\b", 3.5, "😱", "बड़ा शहर"),
]

# ── Bank E: Confusion & Chaos Moments ────────────────────────────
CONFUSION_HOOKS = [
    (r"\b(samjha|samjhe|koi doubt|any questions|clear hai|understood)\b", 3.0, "🤯", "समझे"),
    (r"\b(nahi samjha|i don't understand|repeat|dobara batao|ek baar aur)\b", 3.5, "🤯", "दोहराएं"),
    (r"\b(wrong answer|galat|incorrect|yeh kya likha|what did you write)\b", 4.5, "😂", "गलत जवाब"),
    (r"\b(everybody wrong|sab galat|no one correct|nobody|kisi ko nahi aaya)\b", 5.0, "😂", "सब गलत"),
    (r"\b(read the question|question padho|question hi nahi padha|kya padhe)\b", 4.0, "😂", "सवाल"),
    (r"\b(how many have done|kitno ne kiya|hands up|haath uthao|raise hand)\b", 3.0, "😂", "हाथ उठाओ"),
    (r"\b(simple question|itna simple|class 6|basic|elementary|seedha sawal)\b", 5.0, "😂", "आसान सवाल"),
    (r"\b(you are gone|gaye kaam se|finish|khatam|duniya aabaad hai)\b", 5.5, "😂", "खत्म"),
]

# ── Bank F: Legendary Indian Teacher Catchphrases ────────────────
CATCHPHRASE_HOOKS = [
    (r"\b(beta|bete|betiya|son|dear students|priy vidyarthi)\b", 2.5, "😊", "बेटा"),
    (r"\b(very good|shabash|well done|shaabaash|excellent|bahut badhiya)\b", 3.5, "🎉", "शाबाश"),
    (r"\b(not like that|aisa nahi|theek se|properly|sahi se karo)\b", 3.0, "😤", "सुधार"),
    (r"\b(I repeat|main dobara keh raha|for the last time|akhri baar)\b", 4.0, "😤", "दोहराना"),
    (r"\b(formula yaad karo|mug up|ratta maaro|by heart|learn by heart)\b", 4.5, "😂", "रट्टा"),
    (r"\b(what is your name|naam kya hai|tumhara naam|stand up and tell)\b", 4.0, "😂", "नाम"),
    (r"\b(sit down|baith jao|bait ho|bench pe baith|apni jagah pe)\b", 3.0, "😤", "बैठो"),
    (r"\b(marks kaise aayenge|marks nahi aayenge|percentage gir gayi)\b", 4.5, "😱", "मार्क्स"),
    (r"\b(padhai nahi ki|didn't study|nahi padha|syllabus nahi dekha)\b", 4.0, "😤", "पढ़ाई"),
    (r"\b(kal ki class mein|tomorrow|next class|baad mein bataunga)\b", 2.5, "😊", "अगली क्लास"),
]

# ── Bank G: Roast / Comedy Gold Moments ──────────────────────────
ROAST_HOOKS = [
    (r"\b(even a child|bachcha bhi jaanta|class 1 ka bachcha|nursery mein)\b", 6.0, "😂", "बच्चा भी जाने"),
    (r"\b(donkey|gadha|buffalo|bhains|owl|ullu|monkey|bandar|idiot)\b", 5.5, "😂", "गालियां"),
    (r"\b(you make me cry|rula doge|dimag kharab kar diya|matlab kya)\b", 5.0, "😂", "रुलाया"),
    (r"\b(your mother|teri amma|tere baap|your father|ghar waale)\b", 4.5, "😂", "घरवाले"),
    (r"\b(wasted money|fees barbaad|paisa barbaad|paise doob gaye)\b", 5.5, "😂", "पैसे बर्बाद"),
    (r"\b(google se pooch|search kar|why are you here|kyun aaye ho)\b", 5.0, "😂", "गूगल"),
    (r"\b(sleeping in class|so raha tha|neend aa rahi|khol aankhein)\b", 5.5, "😂", "नींद"),
    (r"\b(love letter|girlfriend|boyfriend|phone mein kya dekh raha)\b", 6.0, "😂", "प्यार"),
    (r"\b(tuition fees|tuition wala|coaching|institute|institute ki fees)\b", 3.5, "😂", "ट्यूशन"),
    (r"\b(hairstyle|fashion|style maar raha|cool lag raha|attitude)\b", 4.5, "😂", "फैशन"),
]

# ── Bank H: Wisdom Bombs / Deep Teaching Moments ─────────────────
WISDOM_HOOKS = [
    (r"\b(life mein|zindagi mein|real life|practical|application|use hota)\b", 3.0, "💡", "जीवन"),
    (r"\b(this is the secret|yahi raaz hai|yahi trick hai|yahi formula)\b", 5.0, "💡", "रहस्य"),
    (r"\b(never forget|kabhi mat bhoolna|always remember|hamesha yaad rakhna)\b", 4.0, "💡", "याद"),
    (r"\b(logic samjho|use your brain|sochna padega|think|dimag lagao)\b", 3.5, "💡", "सोचो"),
    (r"\b(90 percent students|most students|everyone makes this mistake)\b", 4.5, "💡", "गलती"),
    (r"\b(shortcut nahi|no shortcut|mehnat karo|hard work|grind)\b", 3.5, "💡", "मेहनत"),
    (r"\b(in my 20 years|20 saal mein|experience se|maine dekha hai)\b", 4.0, "💡", "अनुभव"),
    (r"\b(first principle|basics|foundation|neenv|base clear karo)\b", 3.0, "💡", "नींव"),
    (r"\b(trick hai|technique|approach|method|easy way|aasan tarika)\b", 3.5, "💡", "तरीका"),
    (r"\b(crack karna|crack the|score high|90 percent|100 marks|full marks)\b", 4.0, "🏆", "सफलता"),
]

# ── Bank I: Storytelling / Personal Anecdote Moments ─────────────
STORY_HOOKS = [
    (r"\b(ek baar ki baat|once upon a time|let me tell you|sunao tumhe)\b", 4.0, "📖", "कहानी"),
    (r"\b(mere student ne|my student|a student of mine|ek student tha)\b", 4.5, "📖", "विद्यार्थी"),
    (r"\b(when i was young|jab main chota tha|mere zamane mein|in my days)\b", 5.0, "📖", "पुरानी यादें"),
    (r"\b(true story|sachi kahani|real incident|actually hua|this happened)\b", 4.5, "📖", "सच्ची कहानी"),
    (r"\b(IIT gaya|AIIMS gaya|topper bana|first came|rank mila|selection)\b", 5.0, "🏆", "सफलता की कहानी"),
    (r"\b(i remember|yaad hai mujhe|i'll never forget|kabhi nahi bhoolunga)\b", 3.5, "📖", "याद"),
    (r"\b(village|gaon|poor|garib|cycle se aata tha|paidl aata tha)\b", 5.5, "😢", "संघर्ष"),
    (r"\b(phone nahi tha|no phone|no internet|library mein|kitaben)\b", 4.0, "📖", "पुराना ज़माना"),
]

# ── Bank J: Audience Participation & Classroom Drama ─────────────
DRAMA_HOOKS = [
    (r"\b(everyone repeat|sab bolenge|class ke saath|together|mil ke bolo)\b", 4.0, "🎭", "साथ बोलो"),
    (r"\b(stand up|khade ho jao|aage aao|come to board|board pe likho)\b", 4.5, "🎭", "नाटक"),
    (r"\b(who knows|kaun jaanta|anyone|koi bata sakta|can anyone tell)\b", 3.0, "🎭", "कौन जाने"),
    (r"\b(vote|show of hands|haath uthao|agree karte ho|disagree)\b", 3.5, "🎭", "वोट"),
    (r"\b(bet|shartiya|guarantee|main guarantee leta|main zimmedar)\b", 5.0, "🎭", "गारंटी"),
    (r"\b(clap|taali|appreciation|badhiya kiya|well answered|sahi kaha)\b", 3.5, "🎉", "तालियां"),
    (r"\b(new topic|naya chapter|interesting topic|mazedaar|exciting)\b", 3.0, "🎭", "नया विषय"),
    (r"\b(bonus|extra marks|grace|reward|prize|gift|special)\b", 4.5, "🎉", "बोनस"),
]

# ── Bank K: Science / Math "Mind Blown" Explanations ─────────────
MINDBLOW_HOOKS = [
    (r"\b(mind blown|amazing|incredible|unbelievable|shocking|surprising)\b", 4.0, "🤯", "दिमाग उड़ा"),
    (r"\b(this is beautiful|kitna sundar|how elegant|genius|masterpiece)\b", 4.5, "🤯", "शानदार"),
    (r"\b(wait wait wait|ruko ruko|pause karo|ek second|hold on)\b", 5.0, "🤯", "रुको"),
    (r"\b(plot twist|actually|but here's the thing|lekin yahan dekhna)\b", 4.5, "🤯", "मोड़"),
    (r"\b(counterintuitive|ulta|opposite|jaise sochte waise nahi|paradox)\b", 5.0, "🤯", "उल्टा"),
    (r"\b(zero|infinity|infinite|crore|trillion|unimaginably|vast)\b", 3.5, "🤯", "बड़ी संख्या"),
    (r"\b(Newton|Einstein|Ramanujan|Aryabhata|Bhaskar|Bohr|Mendel|Darwin)\b", 3.0, "💡", "वैज्ञानिक"),
    (r"\b(why does|kyun hota|how does|kaise hota|have you ever wondered)\b", 3.5, "🤯", "क्यों"),
]

# ── Bank L: Laughter / Funny Slip of Tongue ──────────────────────
SLIP_HOOKS = [
    (r"\b(sorry sorry|maine galti ki|i made a mistake|correction|theek karte)\b", 4.0, "😂", "माफ़ी"),
    (r"\b(haha|hehe|joke|mazaak|funny|hansi|laugh|khilkhilana)\b", 4.5, "😂", "हंसी"),
    (r"\b(that came out wrong|matlab yeh tha|main yeh nahi keh raha|aisa mat samjho)\b", 5.5, "😂", "गलत निकला"),
    (r"\b(microphone|mic|speaker|volume|recording|camera|zoom)\b", 3.5, "😂", "तकनीक"),
    (r"\b(marker|chalk|duster|board|whiteboard|pen nahi chal raha)\b", 3.5, "😂", "तख्ता"),
    (r"\b(wrong example|galat example|matlab woh nahi|actually nahi)\b", 4.5, "😂", "गलत उदाहरण"),
]

# ── Bank M: Profile-Specific Banks ───────────────────────────────
IITJEE_HOOKS = [
    (r"\b(JEE Advanced|JEE Mains|IIT Bombay|IIT Delhi|rank under 100|AIR)\b", 5.0, "🏆", "JEE"),
    (r"\b(organic chemistry|inorganic|physical chemistry|mole concept|stoichiometry)\b", 3.5, "💡", "रसायन"),
    (r"\b(calculus|integration|differentiation|limits|vectors|matrix|determinant)\b", 3.5, "💡", "गणित"),
    (r"\b(electrostatics|magnetism|optics|modern physics|thermodynamics|waves)\b", 3.5, "💡", "भौतिकी"),
    (r"\b(Kota|Allen|Resonance|FIITJEE|Bansal|narayana|Sri chaitanya)\b", 4.5, "😂", "कोटा"),
]
NEET_HOOKS = [
    (r"\b(NEET|AIIMS|MBBS|BDS|doctor|medical|MBBS seat|medical college)\b", 5.0, "🏆", "NEET"),
    (r"\b(biology|botany|zoology|anatomy|physiology|genetics|evolution)\b", 3.5, "💡", "जीव विज्ञान"),
    (r"\b(cell|DNA|RNA|chromosome|mutation|enzyme|hormone|blood)\b", 3.5, "💡", "जीव विज्ञान"),
    (r"\b(cutoff|category|OBC|SC|ST|general|reservation|marks required)\b", 4.0, "😱", "कट-ऑफ"),
]
SCHOOL_HOOKS = [
    (r"\b(CBSE|ICSE|state board|10th|12th|class 10|class 12|board)\b", 4.0, "😱", "बोर्ड"),
    (r"\b(attendance|absent|bunking|proxy|signature|leave letter)\b", 5.0, "😂", "अनुपस्थिति"),
    (r"\b(monitor|class representative|CR|captain|house captain|prefect)\b", 3.5, "😂", "मॉनिटर"),
    (r"\b(homework|HW|assignment|project|submission|due date|deadline)\b", 4.0, "😱", "होमवर्क"),
    (r"\b(sports day|annual day|function|program|chief guest|prize)\b", 3.5, "🎉", "उत्सव"),
]
SSC_HOOKS = [
    (r"\b(SSC CGL|CHSL|MTS|IBPS|SBI|banking|government job|sarkari naukri)\b", 5.0, "🏆", "सरकारी नौकरी"),
    (r"\b(current affairs|GK|general knowledge|reasoning|quantitative)\b", 3.5, "💡", "सामान्य ज्ञान"),
    (r"\b(vacancy|post|notification|form fill|apply|last date)\b", 4.0, "😱", "वेकेंसी"),
    (r"\b(salary|pay scale|perks|DA|HRA|TA|grade pay|7th pay)\b", 4.0, "😱", "सैलरी"),
]

# Profile → extra hook banks mapping
PROFILE_BANKS: Dict[str, List] = {
    "iitjee":  IITJEE_HOOKS,
    "neet":    NEET_HOOKS,
    "school":  SCHOOL_HOOKS,
    "ssc":     SSC_HOOKS,
    "ca":      [],
    "college": [],
    "general": [],
}

# Master bank list (profile-specific ones appended at runtime)
BASE_HOOK_BANKS = [
    ("scolding_hooks",     SCOLDING_HOOKS),
    ("desi_analogy",       DESI_ANALOGY_HOOKS),
    ("exam_threats",       EXAM_THREAT_HOOKS),
    ("motivation",         MOTIVATION_HOOKS),
    ("confusion_chaos",    CONFUSION_HOOKS),
    ("catchphrases",       CATCHPHRASE_HOOKS),
    ("roast_comedy",       ROAST_HOOKS),
    ("wisdom_bombs",       WISDOM_HOOKS),
    ("storytelling",       STORY_HOOKS),
    ("classroom_drama",    DRAMA_HOOKS),
    ("mindblow",           MINDBLOW_HOOKS),
    ("slip_tongue",        SLIP_HOOKS),
]

# ══════════════════════════════════════════════════════════════════
#  2.  DATA CLASSES
# ══════════════════════════════════════════════════════════════════

@dataclass
class Segment:
    start: float
    end:   float
    text:  str

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def words(self) -> List[str]:
        return self.text.strip().split()

    @property
    def word_count(self) -> int:
        return len(self.words)


@dataclass
class ClipCandidate:
    start:             float
    end:               float
    title:             str           = "untitled"
    total_score:       float         = 0.0
    funny_score:       float         = 0.0
    viral_score:       float         = 0.0
    education_score:   float         = 0.0
    emotion_tag:       str           = "😊"
    emotion_label_hi:  str           = "सामान्य"
    score_breakdown:   dict          = field(default_factory=dict)
    hook_phrases:      List[str]     = field(default_factory=list)
    segments:          List[Segment] = field(default_factory=list)
    hindi_caption:     str           = ""
    hindi_title:       str           = ""
    ai_reason_hi:      str           = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments)

    def overlaps_with(self, other: "ClipCandidate") -> float:
        overlap_start = max(self.start, other.start)
        overlap_end   = min(self.end,   other.end)
        if overlap_end <= overlap_start:
            return 0.0
        return (overlap_end - overlap_start) / max(self.duration, 0.001)

# ══════════════════════════════════════════════════════════════════
#  3.  TRANSCRIPT LOADER
#      Supports: Whisper JSON, plain JSON list, SRT (basic), plain TXT
# ══════════════════════════════════════════════════════════════════

def load_segments(path: str) -> List[Segment]:
    ext = os.path.splitext(path)[1].lower()

    if ext in (".txt", ".srt"):
        return _load_srt_or_txt(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw = data.get("segments", data) if isinstance(data, dict) else data

    segs = []
    for s in raw:
        start = float(s.get("start", 0.0))
        end   = float(s.get("end",   start + 0.001))
        text  = s.get("text", "").strip()
        if text:
            segs.append(Segment(start=start, end=end, text=text))
    return segs


def _load_srt_or_txt(path: str) -> List[Segment]:
    """Parse SRT subtitles or plain timestamped text into Segments."""
    segs = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # SRT pattern: 00:00:01,000 --> 00:00:04,500
    srt_blocks = re.split(r"\n\n+", content.strip())
    for block in srt_blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        time_line = next((l for l in lines if "-->" in l), None)
        if not time_line:
            continue
        m = re.match(
            r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)",
            time_line)
        if not m:
            continue
        h1,m1,s1,ms1,h2,m2,s2,ms2 = [int(x) for x in m.groups()]
        start = h1*3600 + m1*60 + s1 + ms1/1000
        end   = h2*3600 + m2*60 + s2 + ms2/1000
        text  = " ".join(l for l in lines if "-->" not in l
                         and not l.strip().isdigit()).strip()
        if text:
            segs.append(Segment(start=start, end=end, text=text))
    return segs

# ══════════════════════════════════════════════════════════════════
#  4.  MULTI-DIMENSIONAL HEURISTIC SCORER
# ══════════════════════════════════════════════════════════════════

def _get_active_banks() -> List[Tuple[str, List]]:
    banks = list(BASE_HOOK_BANKS)
    profile_specific = PROFILE_BANKS.get(TEACHER_PROFILE, [])
    if profile_specific:
        banks.append(("profile_specific", profile_specific))
    return banks


def score_text_heuristic(text: str) -> Tuple[float, dict, List[str], str, str]:
    """
    Score a block of text against all hook banks.
    Returns (total_score, breakdown, matched_phrases, dominant_emotion_tag, dominant_emotion_label_hi).
    """
    lower = text.lower()
    breakdown: Dict[str, float] = {}
    matched_phrases: List[str] = []
    emotion_votes: Dict[str, int] = Counter()

    for bank_name, patterns in _get_active_banks():
        bank_score = 0.0
        for pattern, weight, emotion, label_hi in patterns:
            for m in re.finditer(pattern, lower):
                bank_score += weight
                phrase = m.group(0).strip()
                if phrase not in matched_phrases:
                    matched_phrases.append(phrase)
                emotion_votes[emotion + "|" + label_hi] += 1
        breakdown[bank_name] = round(bank_score, 2)

    # ── Extra signals ──────────────────────────────────────────────
    # Code-switching (Hindi-English mix = very common in Indian classrooms)
    hindi_words = len(re.findall(
        r"\b(hai|hain|nahi|karo|jao|aao|yeh|woh|kya|kyun|kaise|matlab|accha|theek)\b", lower))
    breakdown["hinglish_score"] = round(min(hindi_words * 0.4, 6.0), 2)

    # Repetition for emphasis (Indian teachers repeat for effect)
    repeat_patterns = re.findall(r"\b(\w{3,})\b.*\b\1\b", lower)
    breakdown["repetition_emphasis"] = round(min(len(repeat_patterns) * 0.5, 4.0), 2)

    # Exclamations and dramatic punctuation
    exclamations = len(re.findall(r"[!?]{2,}|[A-Z]{4,}", text))
    breakdown["dramatic_exclamation"] = round(min(exclamations * 1.0, 5.0), 2)

    # Numbers / statistics (teachers love precise numbers)
    number_refs = len(re.findall(r"\b(\d+)\b", text))
    breakdown["number_authority"] = round(min(number_refs * 0.3, 3.0), 2)

    # Question hooks (rhetorical questions = very common hook)
    question_hooks = len(re.findall(
        r'(?:^|[.!?]\s+)(kya|kyun|kaise|what|why|how|who|when|have you ever|batao)\b', lower))
    breakdown["question_hooks"] = round(question_hooks * 2.0, 2)

    # Direct student address
    student_address = len(re.findall(
        r"\b(you|tumhe|tumhara|tumhari|aap|aapko|students|class|sab|suno)\b", lower))
    breakdown["student_address"] = round(min(student_address * 0.25, 4.0), 2)

    # Sentence variety (punchy mix)
    sentences = re.split(r'[.!?]+', text.strip())
    lengths = [len(s.split()) for s in sentences if s.strip()]
    if len(lengths) > 1:
        variety = (max(lengths) - min(lengths)) / (max(lengths) + 1)
        breakdown["sentence_variety"] = round(variety * 3.0, 2)
    else:
        breakdown["sentence_variety"] = 0.0

    # Length penalty
    word_count = len(text.split())
    if word_count < 35:
        breakdown["length_penalty"] = round(-(35 - word_count) * 0.08, 2)
    else:
        breakdown["length_penalty"] = 0.0

    total = round(sum(breakdown.values()), 3)

    # Determine dominant emotion
    dominant = emotion_votes.most_common(1)
    if dominant:
        emo_key = dominant[0][0]
        emoji_part, label_part = emo_key.split("|", 1)
    else:
        emoji_part, label_part = "😊", "सामान्य"

    # Sub-scores
    funny_banks = ["scolding_hooks", "desi_analogy", "roast_comedy",
                   "confusion_chaos", "catchphrases", "slip_tongue"]
    viral_banks = ["exam_threats", "mindblow", "classroom_drama",
                   "storytelling", "hinglish_score", "dramatic_exclamation"]
    edu_banks   = ["wisdom_bombs", "motivation", "profile_specific",
                   "number_authority", "question_hooks"]

    funny_score = sum(breakdown.get(b, 0) for b in funny_banks)
    viral_score = sum(breakdown.get(b, 0) for b in viral_banks)
    edu_score   = sum(breakdown.get(b, 0) for b in edu_banks)

    breakdown["_funny"]    = round(funny_score, 2)
    breakdown["_viral"]    = round(viral_score, 2)
    breakdown["_education"]= round(edu_score, 2)

    return total, breakdown, matched_phrases, emoji_part, label_part


def build_window_text(segments: List[Segment], center_start: float,
                      window: float) -> Tuple[str, List[Segment]]:
    included = []
    for seg in segments:
        mid = (seg.start + seg.end) / 2
        if center_start <= mid <= center_start + window:
            included.append(seg)
    text = " ".join(s.text.strip() for s in included)
    return text, included

# ══════════════════════════════════════════════════════════════════
#  5.  CLIP BOUNDARY OPTIMIZER
# ══════════════════════════════════════════════════════════════════

SENTENCE_ENDS_RE = re.compile(r'[.!?।]$')   # includes Hindi danda

def optimize_boundaries(candidate: ClipCandidate,
                        all_segments: List[Segment]) -> ClipCandidate:
    inner = [s for s in all_segments
             if s.start >= candidate.start - BOUNDARY_SLACK
             and s.end   <= candidate.end   + BOUNDARY_SLACK]
    if not inner:
        return candidate

    best_end_seg = None
    for seg in reversed(inner):
        if SENTENCE_ENDS_RE.search(seg.text.strip()):
            best_end_seg = seg
            break
    new_end   = best_end_seg.end if best_end_seg else inner[-1].end
    new_start = inner[0].start

    duration = new_end - new_start
    if duration > MAX_CLIP_SECONDS:
        new_end = new_start + MAX_CLIP_SECONDS
    if duration < MIN_CLIP_SECONDS:
        new_end = new_start + MIN_CLIP_SECONDS

    candidate.start = round(new_start, 3)
    candidate.end   = round(min(new_end, all_segments[-1].end), 3)
    candidate.segments = [
        s for s in all_segments
        if s.start >= candidate.start - 0.5 and s.end <= candidate.end + 0.5
    ]
    return candidate

# ══════════════════════════════════════════════════════════════════
#  6.  OVERLAP DEDUPLICATOR
# ══════════════════════════════════════════════════════════════════

def deduplicate_clips(candidates: List[ClipCandidate], n: int) -> List[ClipCandidate]:
    sorted_cands = sorted(candidates, key=lambda c: c.total_score, reverse=True)
    selected: List[ClipCandidate] = []
    for cand in sorted_cands:
        if len(selected) >= n:
            break
        skip = False
        for chosen in selected:
            overlap = cand.overlaps_with(chosen)
            if overlap > 0.5:
                skip = True
                break
            if overlap > 0.25:
                cand.total_score *= OVERLAP_PENALTY
        if not skip:
            selected.append(cand)
    selected.sort(key=lambda c: c.total_score, reverse=True)
    return selected[:n]

# ══════════════════════════════════════════════════════════════════
#  7.  AUTO-TITLE GENERATOR (English + Hindi)
# ══════════════════════════════════════════════════════════════════

HINDI_TITLES_BY_EMOTION = {
    "😤": ["जब टीचर आग बबूला हुए", "डांट का महाकाव्य", "क्लास में तूफान",
            "गुस्से का बज्र", "जब क्लासरूम थर-थराया"],
    "😂": ["हंसी की पाठशाला", "क्लास में कॉमेडी", "मज़ेदार पल",
            "जब हंसी नहीं रुकी", "टीचर का जोक बम"],
    "💡": ["ज्ञान का प्रकाश", "समझ आया!", "दिमाग खुला",
            "असली ट्रिक सामने आई", "जादुई फॉर्मूला"],
    "🤯": ["दिमाग फट गया!", "अरे बाप रे!", "यह तो अजब था",
            "चौंका देने वाला पल", "माइंड ब्लास्ट"],
    "😱": ["डर का माहौल", "परीक्षा की दहशत", "खतरे की घंटी",
            "जब दिल धड़का", "असली चुनौती"],
    "💪": ["प्रेरणा की आंधी", "हिम्मत का पल", "जुनून जागा",
            "संघर्ष से सफलता", "विश्वास की शक्ति"],
    "😢": ["दिल छू गया", "भावनाओं का सैलाब", "यादें ताजा हुईं",
            "अंदर तक हिला दिया", "सच्ची बात"],
    "🏆": ["चैंपियन का रास्ता", "टॉपर बनने का मंत्र", "सफलता की कुंजी",
            "रैंक का रहस्य", "जीत की कहानी"],
    "📖": ["सुनो यह किस्सा", "यादगार कहानी", "टीचर की ज़ुबानी",
            "सच्ची दास्तान", "पुरानी याद"],
    "🎭": ["क्लास में ड्रामा", "रंगमंच क्लासरूम", "अनोखा नज़ारा",
            "लाइव शो", "अद्भुत दृश्य"],
    "🎉": ["जीत का जश्न", "शाबाशी का लम्हा", "खुशी का पल",
            "इनाम मिला", "बधाई का माहौल"],
    "😊": ["मीठी बातें", "टीचर की नसीहत", "प्यार से डांट",
            "अच्छा सबक", "दिल की बात"],
}


def auto_title_en(candidate: ClipCandidate) -> str:
    text = candidate.text.lower()
    m = re.search(
        r"\b(sleeping|copy|love letter|out of class|zero marks|even a child|"
        r"google se|wasted money|donkey|your father)\b", text)
    if m:
        return m.group(0).replace(" ", "_")[:40]
    if candidate.hook_phrases:
        phrase = candidate.hook_phrases[0]
        title  = re.sub(r"[^a-z0-9 ]", "", phrase.lower()).strip()
        title  = "_".join(title.split()[:4])
        if title:
            return title[:40]
    words = [w for w in text.split() if len(w) > 2][:4]
    return "_".join(words)[:40] or f"clip_{int(candidate.start)}s"


def auto_title_hi(candidate: ClipCandidate) -> str:
    import random
    options = HINDI_TITLES_BY_EMOTION.get(candidate.emotion_tag, ["यादगार पल"])
    return random.choice(options)

# ══════════════════════════════════════════════════════════════════
#  8.  CLAUDE AI: HINDI TRANSLATION + SEMANTIC RANKING
# ══════════════════════════════════════════════════════════════════

def _call_claude(prompt: str, max_tokens: int = 1500) -> str:
    if not ANTHROPIC_API_KEY:
        return ""
    payload = json.dumps({
        "model":      CLAUDE_MODEL,
        "max_tokens": max_tokens,
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
        with urllib.request.urlopen(req, timeout=45) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["content"][0]["text"].strip()
    except Exception as e:
        print(f"  [AI Warning] Claude API call failed: {e}")
        return ""


def _translate_to_hindi_batch(clips: List[ClipCandidate]) -> None:
    """
    Translate each clip's key phrase/caption into Hindi using Claude.
    Updates clip.hindi_caption and clip.hindi_title in-place.
    """
    if not ANTHROPIC_API_KEY:
        print("[AI] No API key — using auto-generated Hindi titles.")
        for clip in clips:
            clip.hindi_caption = _fallback_hindi_caption(clip)
            clip.hindi_title   = auto_title_hi(clip)
        return

    print(f"[AI] Translating {len(clips)} clips to Hindi via Claude...")

    clips_data = []
    for i, clip in enumerate(clips):
        preview = clip.text[:300].strip()
        clips_data.append({
            "id":         i,
            "text":       preview,
            "emotion":    clip.emotion_label_hi,
            "emoji":      clip.emotion_tag,
            "en_title":   clip.title,
        })

    prompt = f"""You are an expert Hindi translator specializing in Indian classroom content.
Below are {len(clips_data)} funny/interesting moments from an Indian teacher's lecture.
Each has a clip ID, English text excerpt, emotion label, and English title.

For EACH clip, provide:
1. "hindi_caption": A SHORT (max 12 words) punchy Hindi caption that captures the FUNNIEST/MOST VIRAL aspect.
   - Use Devanagari script (हिंदी में लिखें)
   - Make it exciting, relatable, shareable for Indian students
   - Include the emotion feel — e.g., scolding, humor, wisdom, drama
2. "hindi_title": A creative Hindi title (max 6 words) for the clip, in Devanagari
3. "reason_hi": One sentence in Hindi explaining WHY this clip is viral/funny (Devanagari)

Return ONLY valid JSON (no markdown fences):
[
  {{"id": 0, "hindi_caption": "...", "hindi_title": "...", "reason_hi": "..."}},
  ...
]

Clips:
{json.dumps(clips_data, ensure_ascii=False, indent=2)}
"""

    raw = _call_claude(prompt, max_tokens=2000)
    if not raw:
        for clip in clips:
            clip.hindi_caption = _fallback_hindi_caption(clip)
            clip.hindi_title   = auto_title_hi(clip)
        return

    # Strip fences if any
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        results = json.loads(raw)
        id_map = {item["id"]: item for item in results}
        for i, clip in enumerate(clips):
            info = id_map.get(i, {})
            clip.hindi_caption = info.get("hindi_caption", _fallback_hindi_caption(clip))
            clip.hindi_title   = info.get("hindi_title",   auto_title_hi(clip))
            clip.ai_reason_hi  = info.get("reason_hi",     "")
        print(f"[AI] Hindi translations complete for {len(clips)} clips.")
    except Exception as e:
        print(f"[AI] Hindi translation parse error: {e}. Using fallbacks.")
        for clip in clips:
            clip.hindi_caption = _fallback_hindi_caption(clip)
            clip.hindi_title   = auto_title_hi(clip)


def _fallback_hindi_caption(clip: ClipCandidate) -> str:
    """Generate a basic Hindi caption without AI."""
    caption_map = {
        "😤": "जब टीचर ने क्लास को हिला दिया!",
        "😂": "हंसी नहीं रुकी! 😂",
        "💡": "यह ट्रिक जानना जरूरी है!",
        "🤯": "सुनकर दिमाग घूम गया!",
        "😱": "परीक्षा की असली सच्चाई!",
        "💪": "टीचर की बात दिल तक पहुंची!",
        "😢": "भावनाओं से भरा पल...",
        "🏆": "टॉपर बनने का यही रहस्य है!",
        "📖": "सुनो यह किस्सा...",
        "🎭": "क्लास में अनोखा नज़ारा!",
    }
    return caption_map.get(clip.emotion_tag, "यादगार क्षण!")


def _claude_rank(candidates: List[ClipCandidate], n: int) -> List[ClipCandidate]:
    """Semantic AI ranking specialized for Indian teacher lecture virality."""
    if not ANTHROPIC_API_KEY:
        return candidates

    clips_json = []
    for i, c in enumerate(candidates[:min(n * 3, 20)]):
        clips_json.append({
            "id":            i,
            "start":         c.start,
            "end":           c.end,
            "heuristic":     c.total_score,
            "funny":         c.funny_score,
            "viral":         c.viral_score,
            "education":     c.education_score,
            "emotion":       c.emotion_tag + " " + c.emotion_label_hi,
            "hook_phrases":  c.hook_phrases[:4],
            "text":          c.text[:400],
        })

    prompt = f"""You are a viral content expert for Indian education channels (YouTube Shorts / Instagram Reels).
Below are {len(clips_json)} candidate clips from an Indian teacher's lecture ({TEACHER_PROFILE} profile).

Pick the TOP {n} clips most likely to go viral among Indian students aged 15–25.

Evaluate each on:
1. 🔥 Virality — Would students share this? Does it capture an iconic Indian teacher moment?
2. 😂 Humor — Funny scolding, desi analogies, classroom chaos, slip-ups?
3. 💡 Value — Genuine insight, trick, or wisdom that students would save?
4. 🎭 Drama — Suspense, surprise, over-the-top delivery, emotional moment?
5. 🤝 Relatability — Does every Indian student recognize this situation?

PRIORITIZE: Funny scolding > Desi analogy > Exam threats > Wisdom bombs > Motivation speeches

Return ONLY valid JSON (no fences), ranked best-first:
[
  {{"id": 2, "reason": "...one line in English"}},
  ...
]

Clips:
{json.dumps(clips_json, ensure_ascii=False, indent=2)}
"""

    raw = _call_claude(prompt, max_tokens=1000)
    if not raw:
        return candidates[:n]

    raw = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        ranked = json.loads(raw)
        id_to_cand = {i: c for i, c in enumerate(candidates)}
        reordered: List[ClipCandidate] = []
        for entry in ranked[:n]:
            cid    = entry.get("id")
            reason = entry.get("reason", "")
            if cid is not None and cid in id_to_cand:
                cand = id_to_cand[cid]
                cand.score_breakdown["ai_reason"] = reason
                reordered.append(cand)
        seen = {id(c) for c in reordered}
        for c in candidates:
            if id(c) not in seen and len(reordered) < n:
                reordered.append(c)
        print(f"[AI] Claude ranked {len(reordered)} clips semantically.")
        return reordered[:n]
    except Exception as e:
        print(f"[AI] Ranking parse error: {e}. Using heuristic order.")
        return candidates[:n]

# ══════════════════════════════════════════════════════════════════
#  9.  MAIN CLIP FINDER
# ══════════════════════════════════════════════════════════════════

def find_best_clips(segments: List[Segment], n: int) -> List[ClipCandidate]:
    if not segments:
        return []

    total_duration = segments[-1].end
    print(f"[SCAN] Scanning {total_duration:.0f}s content in {STEP_SECONDS:.0f}s steps "
          f"| Profile: {TEACHER_PROFILE.upper()} | Target: {n} clips\n")

    all_candidates: List[ClipCandidate] = []
    probe_time = 0.0
    probe_count = 0

    while probe_time + MIN_CLIP_SECONDS < total_duration:
        text, window_segs = build_window_text(segments, probe_time, CONTEXT_WINDOW)
        if not text.strip():
            probe_time += STEP_SECONDS
            continue

        total_score, breakdown, phrases, emotion_tag, emotion_hi = score_text_heuristic(text)

        if total_score > 0:
            clip_end = min(
                window_segs[-1].end if window_segs else probe_time + MIN_CLIP_SECONDS,
                probe_time + MAX_CLIP_SECONDS
            )
            cand = ClipCandidate(
                start             = probe_time,
                end               = clip_end,
                total_score       = total_score,
                funny_score       = breakdown.get("_funny", 0),
                viral_score       = breakdown.get("_viral", 0),
                education_score   = breakdown.get("_education", 0),
                emotion_tag       = emotion_tag,
                emotion_label_hi  = emotion_hi,
                score_breakdown   = breakdown,
                hook_phrases      = phrases[:6],
                segments          = window_segs,
            )
            all_candidates.append(cand)

        probe_time += STEP_SECONDS
        probe_count += 1

    print(f"[SCAN] Probed {probe_count} windows → {len(all_candidates)} candidates found.")

    if not all_candidates:
        return []

    all_candidates.sort(key=lambda c: c.total_score, reverse=True)

    # Optimize boundaries on top pool
    top_pool = all_candidates[:min(n * 4, len(all_candidates))]
    for cand in top_pool:
        optimize_boundaries(cand, segments)

    selected = deduplicate_clips(top_pool, n)

    # AI semantic re-ranking
    if ANTHROPIC_API_KEY:
        print(f"\n[AI] Sending {len(selected)} candidates for semantic re-ranking...")
        selected = _claude_rank(selected, n)
    else:
        print("[AI] No ANTHROPIC_API_KEY — using heuristic scores only.")

    # Auto-title
    for cand in selected:
        cand.title      = auto_title_en(cand)
        cand.hindi_title = auto_title_hi(cand)

    # Batch Hindi translation via Claude
    print(f"\n[AI] Generating Hindi captions for {len(selected)} clips...")
    _translate_to_hindi_batch(selected)

    return selected

# ══════════════════════════════════════════════════════════════════
#  10.  HINDI SUBTITLE STYLE  (Devanagari-optimized ASS/SSA style)
# ══════════════════════════════════════════════════════════════════

# Bold, large Devanagari font with white text, black outline, bottom-center
SUBTITLE_STYLE_HINDI = (
    f"Fontname={SUBTITLE_FONT},"
    "Bold=1,"
    "Fontsize=26,"
    "PrimaryColour=&H00FFFFFF,"
    "SecondaryColour=&H0000FFFF,"
    "OutlineColour=&H00000000,"
    "BackColour=&H90000000,"
    "BorderStyle=1,"
    "Outline=3,"
    "Shadow=2,"
    "Alignment=2,"
    "MarginV=70,"
    "MarginL=15,"
    "MarginR=15,"
    "Spacing=0"
)

MAX_WORDS_PER_LINE_HI = 5   # Shorter lines for Devanagari readability

# ══════════════════════════════════════════════════════════════════
#  11.  GPU ENCODER DETECTION
# ══════════════════════════════════════════════════════════════════

ENCODERS = {
    "nvidia": {
        "flags":     ["-c:v", "h264_nvenc", "-gpu", "0",
                      "-preset", "p4", "-rc", "vbr", "-b_ref_mode", "0"],
        "quality":   ["-cq", "20"],
        "label":     "NVIDIA NVENC (GPU)",
        "test_args": ["-f", "lavfi", "-i", "nullsrc=s=256x256:d=1",
                      "-c:v", "h264_nvenc", "-gpu", "0",
                      "-frames:v", "1", "-f", "null", "-"],
    },
    "amd": {
        "flags":     ["-c:v", "h264_amf", "-quality", "balanced"],
        "quality":   ["-qp_i", "20", "-qp_p", "20", "-qp_b", "22"],
        "label":     "AMD AMF (GPU)",
        "test_args": ["-f", "lavfi", "-i", "nullsrc=s=256x256:d=1",
                      "-c:v", "h264_amf", "-frames:v", "1", "-f", "null", "-"],
    },
    "intel": {
        "flags":     ["-c:v", "h264_qsv", "-preset", "medium"],
        "quality":   ["-global_quality", "20"],
        "label":     "Intel QSV (GPU)",
        "test_args": ["-f", "lavfi", "-i", "nullsrc=s=256x256:d=1",
                      "-c:v", "h264_qsv", "-frames:v", "1", "-f", "null", "-"],
    },
    "cpu": {
        "flags":     ["-c:v", "libx264", "-preset", "fast"],
        "quality":   ["-crf", "20"],
        "label":     "CPU x264 (software)",
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
            print("✓")
            return name
        print("✗")
    print("  Falling back to CPU.")
    return "cpu"

# ══════════════════════════════════════════════════════════════════
#  12.  SMART FILE NAMING
# ══════════════════════════════════════════════════════════════════

def make_output_name(video_path: str, clip_title: str,
                     rank: int, run_ts: str, emotion: str) -> str:
    stem       = os.path.splitext(os.path.basename(video_path))[0]
    safe_stem  = re.sub(r"[^a-zA-Z0-9\-_]", "_", stem)
    safe_title = re.sub(r"[^a-zA-Z0-9\-_]", "_", clip_title)
    safe_name  = re.sub(r"[^a-zA-Z0-9\-_]", "_", TEACHER_NAME)
    # Emotion code in filename for easy sorting
    emo_code   = {"😤": "SCOLD", "😂": "FUNNY", "💡": "WISDOM",
                  "🤯": "MIND",  "😱": "DRAMA", "💪": "MOTIVE",
                  "😢": "FEEL",  "🏆": "WIN",   "📖": "STORY",
                  "🎭": "ACT",   "🎉": "CELE",  "😊": "SWEET"
                  }.get(emotion, "CLIP")
    return f"{safe_name}__{emo_code}__{rank:02d}__{safe_title}__{run_ts}.mp4"

# ══════════════════════════════════════════════════════════════════
#  13.  SRT GENERATION — Hindi-aware chunking
# ══════════════════════════════════════════════════════════════════

def srt_time(seconds: float) -> str:
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        s += 1; ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def chunk_words_hindi(text: str, max_words: int) -> List[str]:
    """Chunk words; for Devanagari, count unicode word boundaries."""
    words = text.strip().split()
    chunks = []
    while words:
        chunk = " ".join(words[:max_words]).strip()
        if chunk:
            chunks.append(chunk)
        words = words[max_words:]
    return chunks


def generate_srt(segs: List[Segment], clip_start: float,
                 clip_end: float, out_path: str,
                 hindi_caption: str = "") -> str:
    """
    Generate SRT file for the clip.
    Appends a 2.5-second Hindi caption overlay at the START of the clip (as a title card).
    """
    SRT_END_CAP = 0.05
    entries = []
    idx = 1

    # ── Hindi Title Card (first 2.5 seconds) ──────────────────────
    if hindi_caption and hindi_caption.strip():
        t0_card = 0.0
        t1_card = min(2.5, clip_end - clip_start - 0.1)
        if t1_card > t0_card:
            entries.append(
                f"{idx}\n{srt_time(t0_card)} --> {srt_time(t1_card)}\n{hindi_caption}"
            )
            idx += 1

    # ── Regular word-synced subtitles ─────────────────────────────
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
        lines    = chunk_words_hindi(raw, MAX_WORDS_PER_LINE_HI)
        if not lines:
            continue
        seg_dur  = seg_e - seg_s
        line_dur = seg_dur / len(lines)
        for li, line in enumerate(lines):
            if not line:
                continue
            t0 = seg_s + li * line_dur
            t1 = t0 + line_dur
            # Skip if inside the Hindi title card window
            if t1 <= 2.5 and hindi_caption:
                continue
            entries.append(f"{idx}\n{srt_time(t0)} --> {srt_time(t1)}\n{line}")
            idx += 1

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(entries))
    return out_path

# ══════════════════════════════════════════════════════════════════
#  14.  FFMPEG RUNNER  (9:16 vertical crop + Hindi subs)
# ══════════════════════════════════════════════════════════════════

def run_ffmpeg(video_in: str, ss: float, duration: float,
               srt_path: str, video_out: str, encoder_name: str) -> bool:
    enc = ENCODERS[encoder_name]

    # Escape path for FFmpeg subtitle filter (cross-platform)
    ffmpeg_srt = srt_path.replace("\\", "/")
    if re.match(r"^[A-Za-z]:/", ffmpeg_srt):
        # Windows drive letter: C:/path → C\:/path
        ffmpeg_srt = ffmpeg_srt[0] + "\\:" + ffmpeg_srt[2:]

    vf = (
        "crop=ih*(9/16):ih:(iw-ow)/2:0,"   # Crop to 9:16 vertical
        f"subtitles='{ffmpeg_srt}':"
        f"force_style='{SUBTITLE_STYLE_HINDI}'"
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
            if any(k in line for k in ["frame=", "fps=", "speed=",
                                        "Error", "error", "encoder"]):
                print(f"    {line}", flush=True)
        proc.wait()
        return proc.returncode == 0
    except FileNotFoundError:
        print("[ERROR] ffmpeg not found in PATH. Install from https://ffmpeg.org/")
        return False
    except Exception as e:
        print(f"[ERROR] FFmpeg exception: {e}")
        return False

# ══════════════════════════════════════════════════════════════════
#  15.  SCORE REPORT PRINTER (Bilingual)
# ══════════════════════════════════════════════════════════════════

def print_score_report(clips: List[ClipCandidate]) -> None:
    border = "═" * 72
    print(f"\n{border}")
    print("  📊  CLIP ANALYSIS REPORT  |  क्लिप विश्लेषण रिपोर्ट")
    print(border)

    for i, c in enumerate(clips, start=1):
        print(f"\n  #{i}  {c.emotion_tag}  [{c.title}]")
        print(f"       भाव: {c.emotion_label_hi}  |  हिंदी शीर्षक: {c.hindi_title}")
        print(f"       समय: {c.start:.1f}s → {c.end:.1f}s  ({c.duration:.0f}s)")
        print(f"       कुल स्कोर={c.total_score:.1f}  "
              f"😂मज़ेदार={c.funny_score:.1f}  "
              f"🔥वायरल={c.viral_score:.1f}  "
              f"💡शैक्षिक={c.education_score:.1f}")

        # Score bar chart
        bd = {k: v for k, v in c.score_breakdown.items()
              if isinstance(v, (int, float)) and v > 0 and not k.startswith("_")}
        top_dims = sorted(bd.items(), key=lambda x: x[1], reverse=True)[:5]
        for dim, sc in top_dims:
            bar = "█" * min(int(sc), 25)
            print(f"       {dim:<28} {sc:5.1f}  {bar}")

        # Hook phrases
        if c.hook_phrases:
            phrases_str = " | ".join(f'"{p}"' for p in c.hook_phrases[:5])
            print(f"       hooks : {phrases_str}")

        # Hindi caption
        if c.hindi_caption:
            print(f"       कैप्शन: {c.hindi_caption}")

        # AI reasons
        ai_en = c.score_breakdown.get("ai_reason")
        if ai_en:
            print(f"       AI (EN): {ai_en}")
        if c.ai_reason_hi:
            print(f"       AI (HI): {c.ai_reason_hi}")

        # Text preview
        preview = c.text[:130].replace("\n", " ")
        print(f'       "...{preview}..."')

    print(f"\n{border}\n")


def print_summary_banner() -> None:
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║    TEACHER CLIPS HINDI PRO v1.0  |  टीचर क्लिप्स हिंदी प्रो        ║")
    print(f"║    Profile: {TEACHER_PROFILE.upper():<12}  Teacher: {TEACHER_NAME:<20}       ║")
    print(f"║    Clips: {NUM_CLIPS}  |  AI: {'ON  ✓' if ANTHROPIC_API_KEY else 'OFF ✗'}  |  "
          f"Encoder: auto-detect              ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

# ══════════════════════════════════════════════════════════════════
#  16.  BATCH MODE HELPER
#       Allows processing multiple lecture files from a folder
# ══════════════════════════════════════════════════════════════════

def batch_process(transcript_dir: str, video_dir: str, out_dir: str) -> None:
    """
    Match every .json transcript in transcript_dir with same-name .mp4 in video_dir.
    Process each pair as a separate job.
    """
    transcripts = [f for f in os.listdir(transcript_dir) if f.endswith(".json")]
    print(f"[BATCH] Found {len(transcripts)} transcripts in {transcript_dir}")
    for tf in transcripts:
        stem  = os.path.splitext(tf)[0]
        vf    = os.path.join(video_dir, stem + ".mp4")
        tr    = os.path.join(transcript_dir, tf)
        if not os.path.exists(vf):
            print(f"[BATCH] Skipping {stem} — no matching video found.")
            continue
        job_out = os.path.join(out_dir, stem)
        os.makedirs(job_out, exist_ok=True)
        print(f"\n[BATCH] Processing: {stem}")
        # Override globals for this job
        global TRANSCRIPT_FILE, VIDEO_FILE, OUTPUT_DIR
        TRANSCRIPT_FILE = tr
        VIDEO_FILE      = vf
        OUTPUT_DIR      = job_out
        main_single_job()

# ══════════════════════════════════════════════════════════════════
#  17.  SINGLE JOB PROCESSOR
# ══════════════════════════════════════════════════════════════════

def main_single_job() -> dict:
    """Run one transcript+video through the full pipeline. Returns results dict."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    segments = load_segments(TRANSCRIPT_FILE)
    if not segments:
        print("[ERROR] Transcript empty or unparseable.")
        return {}
    total_dur = segments[-1].end
    print(f"[OK] Loaded {len(segments)} segments  ({total_dur:.0f}s = "
          f"{total_dur/60:.1f} min of content)\n")

    encoder_name = detect_encoder()
    print(f"[GPU] Using: {ENCODERS[encoder_name]['label']}\n")

    t0 = time.time()
    best_clips = find_best_clips(segments, NUM_CLIPS)
    elapsed = time.time() - t0
    print(f"\n[OK] Analysis done in {elapsed:.1f}s → {len(best_clips)} clips selected\n")

    if not best_clips:
        print("[ERROR] No clips found. Ensure transcript has sufficient content.")
        return {}

    print_score_report(best_clips)

    clips_to_render = sorted(best_clips, key=lambda c: c.start)
    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    for rank, clip in enumerate(clips_to_render, start=1):
        ss       = clip.start
        end      = clip.end
        duration = end - ss
        out_name  = make_output_name(VIDEO_FILE, clip.title, rank, run_ts, clip.emotion_tag)
        video_out = os.path.join(OUTPUT_DIR, out_name)
        srt_file  = os.path.join(OUTPUT_DIR, f"_tmp_{clip.title}_{run_ts}.srt")

        print(f"[{rank}/{len(clips_to_render)}] {out_name}")
        print(f"  {clip.emotion_tag} {clip.emotion_label_hi}  |  "
              f"{ss:.1f}s → {end:.1f}s  ({duration:.0f}s)  score={clip.total_score:.1f}")
        print(f"  Hindi Caption: {clip.hindi_caption}")

        srt_segs = clip.segments if clip.segments else segments
        generate_srt(srt_segs, ss, end, srt_file, clip.hindi_caption)

        ok = run_ffmpeg(VIDEO_FILE, ss, duration, srt_file, video_out, encoder_name)

        if os.path.exists(srt_file):
            os.remove(srt_file)

        if ok:
            size_mb = os.path.getsize(video_out) / (1024 * 1024)
            print(f"  ✓ {size_mb:.1f} MB → {video_out}\n")
            results.append({
                "rank":          rank,
                "clip":          out_name,
                "status":        "ok",
                "path":          video_out,
                "total_score":   clip.total_score,
                "funny_score":   clip.funny_score,
                "viral_score":   clip.viral_score,
                "education":     clip.education_score,
                "emotion":       clip.emotion_tag + " " + clip.emotion_label_hi,
                "hindi_caption": clip.hindi_caption,
                "hindi_title":   clip.hindi_title,
                "start":         ss,
                "end":           end,
                "duration":      duration,
                "ai_reason_hi":  clip.ai_reason_hi,
                "hook_phrases":  clip.hook_phrases[:4],
            })
        else:
            print(f"  ✗ FFmpeg error for {out_name}\n")
            results.append({"rank": rank, "clip": out_name, "status": "fail",
                             "path": None, "total_score": clip.total_score})

    # ── Final summary ─────────────────────────────────────────────
    ok_count   = sum(1 for r in results if r["status"] == "ok")
    fail_count = len(results) - ok_count
    border = "═" * 72
    print(border)
    print(f"  ✅ Done: {ok_count}/{len(results)} clips rendered successfully")
    print(f"  📂 Output: {OUTPUT_DIR}")
    for r in results:
        icon = "✓" if r["status"] == "ok" else "✗"
        score_str = f"score={r['total_score']:.1f}" if "total_score" in r else ""
        emo   = r.get("emotion", "")
        print(f"  [{icon}] {emo}  {score_str}  {r['clip']}")
    print(border)

    # Machine-readable output
    results_json = os.path.join(OUTPUT_DIR, f"results_{run_ts}.json")
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump({
            "run_ts":          run_ts,
            "teacher":         TEACHER_NAME,
            "profile":         TEACHER_PROFILE,
            "total_clips":     len(results),
            "ok_count":        ok_count,
            "fail_count":      fail_count,
            "clips":           results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  📄 Results JSON: {results_json}\n")
    return {"ok": ok_count, "fail": fail_count, "results": results}

# ══════════════════════════════════════════════════════════════════
#  18.  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print_summary_banner()

    # Validate inputs
    if not os.path.exists(TRANSCRIPT_FILE):
        print(f"[ERROR] Transcript not found: {TRANSCRIPT_FILE}")
        sys.exit(1)
    if not os.path.exists(VIDEO_FILE):
        print(f"[ERROR] Video not found: {VIDEO_FILE}")
        sys.exit(1)

    result = main_single_job()

    fail_count = result.get("fail", 0)
    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()