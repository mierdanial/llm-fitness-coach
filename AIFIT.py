#  LLM-Enabled Intelligent Fitness Coaching System
#  FYP Project — Amier Danial Bin Mohd Fadzil (21001964)

import os, json, pickle, hashlib, re, time, random
from pathlib import Path
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import faiss
import requests
from sentence_transformers import SentenceTransformer

# ── CONFIG ──────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
OPENROUTER_MODEL   = "openai/gpt-oss-20b:free"
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
CHUNK_SIZE         = 400
CHUNK_OVERLAP      = 80
TOP_K              = 5
DATA_DIR           = Path("fitcoach_data")
INDEX_CACHE        = DATA_DIR / "rag_index.pkl"
PROFILE_FILE       = DATA_DIR / "profile.json"
PLAN_FILE          = DATA_DIR / "plan.txt"

STREAK_MESSAGES = {
    0:  ("Just getting started", "🌱", "#6b7a99"),
    1:  ("First step taken!", "✨", "#00d4ff"),
    3:  ("Building momentum!", "🔥", "#f59e0b"),
    7:  ("One week strong!", "💪", "#10b981"),
    14: ("Two weeks — forming a habit!", "⚡", "#7c3aed"),
    30: ("One month warrior!", "🏆", "#f59e0b"),
}

FITNESS_TIPS = [
    ("Rest days aren't lazy days", "Muscle growth happens during recovery, not during the workout itself."),
    ("Protein timing matters", "Aim to consume protein within 30–60 min post-workout for optimal muscle repair."),
    ("Progressive overload is key", "Add just 2.5–5% more weight or 1–2 reps each week to keep adapting."),
    ("Sleep is your secret weapon", "7–9 hours of sleep can improve strength gains by up to 20%."),
    ("Dehydration kills performance", "Even a 2% drop in hydration can reduce strength output significantly."),
    ("Warm up properly", "5–10 min of dynamic movement reduces injury risk and improves performance."),
    ("Mind-muscle connection", "Consciously contracting the target muscle during a rep increases activation by ~20%."),
    ("Form over weight", "Poor form leads to injury; perfect form leads to growth."),
]

EXERCISE_QUICK_LOG = {
    "Strength": ["Bench Press", "Squat", "Deadlift", "Overhead Press", "Pull-ups", "Rows", "Lunges", "Dips"],
    "Cardio":   ["Running", "Cycling", "Jump Rope", "Rowing", "Swimming", "Stair Climb", "Elliptical"],
    "Flexibility": ["Yoga", "Stretching Session", "Foam Rolling", "Mobility Work"],
    "Recovery": ["Active Recovery Walk", "Ice Bath", "Massage", "Rest Day Tracking"],
}

# ── RAG PIPELINE ─────────────────────────────────────────────────────────────
_embedder: Optional[SentenceTransformer] = None

def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder

def chunk_text(text: str, source: str = "knowledge_base") -> List[dict]:
    chunks, text, start = [], text.strip(), 0
    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append({"text": chunk, "source": source,
                           "id": hashlib.md5(chunk.encode()).hexdigest()[:6]})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

@st.cache_resource(show_spinner=False)
def build_rag_index() -> Tuple[faiss.Index, List[dict]]:
    DATA_DIR.mkdir(exist_ok=True)
    if INDEX_CACHE.exists():
        with open(INDEX_CACHE, "rb") as f:
            return pickle.load(f)
    chunks = []
    for f in DATA_DIR.glob("*.txt"):
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                chunks.extend(chunk_text(text, source=f.name))
        except Exception as e:
            print(f"Failed to load {f.name}: {e}")
    if not chunks:
        raise ValueError("No knowledge base documents found inside fitcoach_data folder.")
    embedder = get_embedder()
    embs = np.array(embedder.encode([c["text"] for c in chunks],
                                    show_progress_bar=False, batch_size=64), dtype="float32")
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    with open(INDEX_CACHE, "wb") as f:
        pickle.dump((index, chunks), f)
    return index, chunks

def retrieve(query: str, index: faiss.Index, chunks: List[dict], k: int = TOP_K) -> str:
    embedder = get_embedder()
    q = np.array(embedder.encode([query], show_progress_bar=False), dtype="float32")
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, k)
    parts = [f"[{chunks[i]['source']}]\n{chunks[i]['text']}"
             for score, i in zip(scores[0], idxs[0]) if i >= 0]
    return "\n\n".join(parts) if parts else "No specific knowledge retrieved."

# ── LLM ──────────────────────────────────────────────────────────────────────
def call_llm(messages: List[dict], max_tokens: int = 1800) -> str:
    if not OPENROUTER_API_KEY:
        return "❌ No API key set"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost",
        "X-Title": "FitCoach AI",
        "Content-Type": "application/json"
    }
    payload = {"model": OPENROUTER_MODEL, "messages": messages,
               "max_tokens": max_tokens, "temperature": 0.4}
    try:
        r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        if r.status_code == 401: return "❌ Invalid API Key — check your .env file."
        if r.status_code == 402: return "❌ No credits left on OpenRouter."
        if r.status_code == 429: return "❌ Rate limit hit — please wait a moment and try again."
        r.raise_for_status()
        data = r.json()
        if "choices" not in data:
            return f"❌ Unexpected response: {data}"
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Connection error: {str(e)}"

def generate_plan(profile: dict, rag_context: str) -> str:
    days = profile.get("days", 4)
    prompt = f"""You are a certified personal trainer and sport scientist.
Generate a personalised fitness plan using ONLY the knowledge below.

=== SPORT SCIENCE KNOWLEDGE (RAG) ===
{rag_context}

=== CLIENT PROFILE ===
Age: {profile.get('age','?')} | Gender: {profile.get('gender','?')}
Weight: {profile.get('weight','?')}kg | Height: {profile.get('height','?')}cm | BMI: {profile.get('bmi','?')}
Fitness Level: {profile.get('level','Beginner')}
Goals: {', '.join(profile.get('goals', ['General fitness']))}
Target: {profile.get('target','Not specified')}
Workout days/week: {days}
Equipment: {', '.join(profile.get('equipment', ['No equipment']))}
Dietary Preferences: {', '.join(profile.get('diet', ['No restrictions']))}
Injuries/Limitations: {profile.get('injuries','None')}

Write with EXACTLY these headers:

## WEEKLY WORKOUT PLAN
{days}-day structured plan. Each day: muscle focus, exercises with sets x reps, rest periods, one coaching cue each.

## PROGRESSIVE OVERLOAD STRATEGY
Week-by-week progression targets based on this client's level. Be specific.

## NUTRITION GUIDE
Daily calorie target, macro breakdown (protein/carbs/fats in grams), meal timing, sample meal day.

## RECOVERY PROTOCOL
Sleep targets, stretching routine, active recovery tips, injury notes.

## 12-WEEK MILESTONES
Week 2, Week 4, Week 8, Week 12 concrete targets.

## COACH'S NOTES
3-5 direct personalised tips for this client.
"""
    return call_llm([
        {"role": "system", "content": "You are a certified personal trainer. Be specific, evidence-based, and practical."},
        {"role": "user", "content": prompt}
    ], max_tokens=1800)

def generate_adaptive_update(profile: dict, log: List[dict], rag_context: str) -> str:
    history = "\n".join([
        f"- {w['date']}: {w['exercise']} {w['sets']}x{w['reps']} @ {w['weight']}kg | Vol: {w['volume']}kg"
        for w in log[-15:]
    ])
    return call_llm([{"role": "user", "content": f"""You are a personal trainer reviewing workout history.

=== SPORT SCIENCE KNOWLEDGE ===
{rag_context}

=== CLIENT ===
Level: {profile.get('level','?')} | Goals: {', '.join(profile.get('goals',[]))}
Injuries: {profile.get('injuries','None')}

=== RECENT SESSIONS ===
{history}

Respond using EXACTLY these three markdown headers, each on its own line, in this exact order, with no extra headers, no bold text instead of headers, and no merging sections together:

## PROGRESS ANALYSIS
Volume trends, consistency, what's working.

## ADAPTIVE RECOMMENDATIONS
Specific load changes, exercise swaps, set/rep adjustments based on progressive overload principles.

## NEXT WEEK TARGETS
3–5 concrete targets for next training week.

Do not skip any header even if a section is short. Do not add commentary before the first header.
"""}], max_tokens=1500)

def chat_response(history: List[dict], profile: dict, plan: str, log: List[dict], rag_context: str) -> str:
    system = f"""You are FitCoach AI — a friendly, encouraging, evidence-based personal trainer.

You MUST answer using ONLY the SPORT SCIENCE KNOWLEDGE below.
Be warm, motivating, and practical — like a real coach who knows the client.

SPORT SCIENCE KNOWLEDGE:
{rag_context}

CLIENT:
Level: {profile.get('level','Beginner')}
Goals: {', '.join(profile.get('goals',[]))}
Injuries: {profile.get('injuries','None')}

RULES:
- If user asks about progression → use Progressive Overload rules
- If user asks about exercises → use Exercise Library
- If injury is mentioned → apply Injury Modifications
- Keep answer under 150 words
- Use bullet points when helpful
- Be encouraging and specific
"""
    messages = [{"role": "system", "content": system}] + history[-10:]
    return call_llm(messages, max_tokens=1000)

# ── SESSION / DATA HELPERS ───────────────────────────────────────────────────
def init_state():
    defaults = {
        "profile": {}, "plan": "", "plan_ready": False,
        "chat_history": [], "workout_log": [], "adaptive_report": "",
        "app_ready": False, "confirm_delete_id": None,
        "daily_tip": random.choice(FITNESS_TIPS),
        "last_logged": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if PROFILE_FILE.exists() and not st.session_state.profile:
        try:
            st.session_state.profile = json.loads(PROFILE_FILE.read_text())
        except Exception:
            pass

    log_path = DATA_DIR / "workout_log.json"
    if log_path.exists() and not st.session_state.workout_log:
        try:
            st.session_state.workout_log = json.loads(log_path.read_text())
        except Exception:
            pass

    if PLAN_FILE.exists() and not st.session_state.plan:
        try:
            st.session_state.plan = PLAN_FILE.read_text(encoding="utf-8")
            st.session_state.plan_ready = True
        except Exception:
            pass

def save_log():
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "workout_log.json").write_text(
        json.dumps(st.session_state.workout_log, indent=2))

def save_profile():
    DATA_DIR.mkdir(exist_ok=True)
    PROFILE_FILE.write_text(json.dumps(st.session_state.profile, indent=2))

def save_plan():
    DATA_DIR.mkdir(exist_ok=True)
    PLAN_FILE.write_text(st.session_state.plan, encoding="utf-8")

def add_entry(exercise, sets, reps, weight, category, notes, duration=0, distance=0.0):
    entry = {
        "id": max((e.get("id", 0) for e in st.session_state.workout_log), default=0) + 1,
        "date": date.today().strftime("%Y-%m-%d"),
        "exercise": exercise, "sets": sets, "reps": reps, "weight": weight,
        "volume": round(sets * reps * weight, 1) if category != "Cardio" else 0.0,
        "category": category, "duration": duration, "distance": distance, "notes": notes
    }
    st.session_state.workout_log.append(entry)
    st.session_state.last_logged = entry
    save_log()
    return entry

def get_stats() -> dict:
    logs = st.session_state.workout_log
    if not logs:
        return {"sessions": 0, "volume": 0, "avg": 0, "streak": 0,
                "top_ex": "—", "cats": {}, "weekly": 0, "prs": {}}
    df = pd.DataFrame(logs)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df["date"]   = pd.to_datetime(df["date"])
    cats  = df["category"].value_counts().to_dict()
    top   = df.groupby("exercise")["volume"].sum().idxmax()
    udates = sorted(df["date"].dt.date.unique(), reverse=True)
    streak, today_ = 0, date.today()
    for i, d in enumerate(udates):
        if d == today_ - pd.Timedelta(days=i): streak += 1
        else: break
    week_start = pd.Timestamp(today_ - pd.Timedelta(days=today_.weekday()))
    weekly = len(df[df["date"] >= week_start])
    prs = {}
    for ex in df["exercise"].unique():
        ex_df = df[df["exercise"] == ex]
        max_w = ex_df["weight"].max()
        if max_w > 0:
            prs[ex] = max_w
    return {"sessions": len(logs), "volume": round(df["volume"].sum(), 1),
            "avg": round(df["volume"].mean(), 1), "streak": streak,
            "top_ex": top, "cats": cats, "weekly": weekly, "prs": prs}

def get_streak_message(streak: int) -> tuple:
    msg, icon, color = "Just getting started", "🌱", "#6b7a99"
    for threshold, (m, i, c) in sorted(STREAK_MESSAGES.items(), reverse=True):
        if streak >= threshold:
            msg, icon, color = m, i, c
            break
    return msg, icon, color

def overload_check() -> List[dict]:
    logs = st.session_state.workout_log
    if len(logs) < 3: return []
    df = pd.DataFrame(logs)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    results = []
    for ex in df["exercise"].unique():
        sub = df[df["exercise"] == ex].tail(3)
        if len(sub) < 2: continue
        last, prev = sub.iloc[-1], sub.iloc[-2]
        diff = last["volume"] - prev["volume"]
        if diff > 0:
            results.append({"ex": ex, "icon": "🟢", "status": "progressing",
                             "msg": f"Volume ↑ {diff:.0f}kg — consider +2.5kg next session."})
        elif diff == 0:
            results.append({"ex": ex, "icon": "🟡", "status": "plateau",
                             "msg": f"Volume held at {last['volume']}kg — try +1 rep per set."})
        else:
            results.append({"ex": ex, "icon": "🔴", "status": "regressing",
                             "msg": "Volume dropped — focus on form and recovery before increasing."})
    return results

# GLOBAL PAGE CONFIG & CSS
st.set_page_config(
    page_title="FitCoach AI",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-base:        #1a2332;
    --bg-card:        #243447;
    --bg-card-hover:  #2d3f57;
    --bg-input:       #1e2d40;
    --border:         #3a5070;
    --border-hover:   #4a6488;
    --accent:         #00ff9d;
    --accent2:        #c084fc;
    --accent3:        #ccff44;
    --warn:           #fcd34d;
    --danger:         #fb7185;
    --text-primary:   #ffffff;
    --text-muted:     #94b8d4;
    --text-dim:       #c8dcea;
    --font-display:   'Syne', sans-serif;
    --font-body:      'Outfit', sans-serif;
    --font-mono:      'JetBrains Mono', monospace;
    --radius-sm:      8px;
    --radius-md:      12px;
    --radius-lg:      16px;
    --radius-xl:      24px;
    --shadow-sm:      0 2px 8px rgba(0,0,0,0.3);
    --shadow-md:      0 8px 32px rgba(0,0,0,0.4);
    --shadow-glow:    0 0 40px rgba(0,255,157,0.08);
}

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: var(--bg-base) !important;
    font-family: var(--font-body);
    color: var(--text-primary);
    -webkit-font-smoothing: antialiased;
}

[data-testid="stSidebar"] {
    background: #1e2d40 !important;
    border-right: 1px solid #3a5070 !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

.block-container {
    padding: 1.8rem 2.2rem 3rem !important;
    max-width: 1200px !important;
}

h1 {
    font-family: var(--font-display) !important;
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.04em !important;
    color: var(--text-primary) !important;
    line-height: 1.1 !important;
    margin-bottom: 0.2rem !important;
}
h2 { font-family: var(--font-display) !important; color: var(--text-primary) !important; font-size: 1.5rem !important; }
h3 { font-family: var(--font-display) !important; color: var(--text-primary) !important; }
h4, h5, h6 { font-family: var(--font-display) !important; color: var(--text-primary) !important; }
p, li { color: var(--text-dim) !important; line-height: 1.75; font-size: 16px; }

/* ── Cards ── */
.fc-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 20px 22px;
    margin-bottom: 14px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s, box-shadow 0.25s, transform 0.2s;
}
.fc-card:hover {
    border-color: var(--border-hover);
    box-shadow: var(--shadow-glow);
    transform: translateY(-1px);
}
.fc-card-active {
    border-color: rgba(57,217,138,0.3) !important;
    box-shadow: 0 0 0 1px rgba(0,229,255,0.1), var(--shadow-glow) !important;
}

/* ── Section Label ── */
.section-label {
    font-size: 10px; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--text-muted); margin: 1.5rem 0 0.8rem;
    display: flex; align-items: center; gap: 10px;
    font-family: var(--font-mono);
}
.section-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* ── RAG Badge ── */
.rag-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(57,217,138,0.04);
    border: 1px solid rgba(57,217,138,0.12);
    border-radius: 100px;
    padding: 5px 14px;
    font-size: 11px; color: var(--accent);
    font-family: var(--font-mono);
    letter-spacing: 0.04em;
    margin-bottom: 16px;
}
.rag-dot {
    width: 6px; height: 6px; background: var(--accent);
    border-radius: 50%; animation: rag-pulse 2s infinite;
}
@keyframes rag-pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.8)} }

/* ── Progress Bar ── */
.prog-bar-outer {
    background: rgba(255,255,255,0.05);
    border-radius: 100px; height: 4px;
    overflow: hidden; margin: 8px 0 4px;
}
.prog-bar-fill {
    height: 100%; border-radius: 100px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}
.prog-bar-fill.green { background: linear-gradient(90deg, var(--accent3), var(--accent)); }

/* ── Chat Bubbles ── */
.bubble-ai {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px 20px 20px 4px;
    padding: 16px 20px;
    font-size: 13.5px; line-height: 1.8;
    max-width: 82%;
    color: var(--text-primary);
    box-shadow: var(--shadow-sm);
}
.bubble-user {
    background: linear-gradient(135deg, #0a1e38, #0d1a30);
    border: 1px solid rgba(57,217,138,0.12);
    border-radius: 20px 20px 4px 20px;
    padding: 16px 20px;
    font-size: 13.5px; line-height: 1.8;
    margin-left: auto; max-width: 82%;
    color: #d0dde8;
    box-shadow: var(--shadow-sm);
}
.bubble-meta { font-size: 10px; color: var(--text-muted); margin-top: 8px; display: flex; align-items: center; gap: 6px; }
.bubble-rag-note {
    font-size: 10px; color: rgba(57,217,138,0.4);
    margin-top: 6px; padding-top: 6px;
    border-top: 1px solid rgba(255,255,255,0.04);
}
.typing-indicator { display: inline-flex; gap: 4px; align-items: center; padding: 14px 18px; }
.typing-dot {
    width: 7px; height: 7px; background: var(--accent);
    border-radius: 50%; animation: typingBounce 1.4s ease-in-out infinite; opacity: 0.5;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes typingBounce { 0%,80%,100%{transform:translateY(0);opacity:0.4} 40%{transform:translateY(-7px);opacity:1} }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 18px 16px !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stMetric"]:hover { border-color: var(--border-hover) !important; box-shadow: var(--shadow-glow) !important; }
[data-testid="stMetricValue"] { font-family: var(--font-display) !important; font-size: 1.75rem !important; color: var(--text-primary) !important; letter-spacing: -0.03em !important; }
[data-testid="stMetricLabel"] { font-size: 11px !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; color: var(--text-muted) !important; font-family: var(--font-mono) !important; }
[data-testid="stMetricDelta"] { font-size: 11px !important; }

/* ── Buttons ── */
.stButton > button {
    font-family: var(--font-body) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    border-radius: var(--radius-sm) !important;
    letter-spacing: 0.01em !important;
    transition: all 0.18s !important;
    border: none !important;
}
.stButton > button[kind="primary"] {
    background: #2563eb !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 20px rgba(37,99,235,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    background: #3b82f6 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(37,99,235,0.45) !important;
}
.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-dim) !important;
}
.stButton > button[kind="secondary"]:hover {
    background: rgba(57,217,138,0.04) !important;
    border-color: rgba(57,217,138,0.2) !important;
    color: var(--text-primary) !important;
}

/* ── Form Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stNumberInput > div > div > input {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-body) !important;
    font-size: 13px !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus,
.stNumberInput > div > div > input:focus {
    border-color: rgba(57,217,138,0.4) !important;
    box-shadow: 0 0 0 3px rgba(57,217,138,0.06) !important;
}
.stSelectbox > div > div, .stMultiSelect > div {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stNumberInput"] label,
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stSlider"] label { color: var(--text-dim) !important; font-size: 12px !important; }
[data-testid="stSlider"] > div > div > div { background: linear-gradient(90deg, var(--accent), var(--accent2)) !important; }

/* ── Expander ── */
.stExpander { border: 1px solid var(--border) !important; border-radius: var(--radius-md) !important; background: var(--bg-card) !important; }
.stExpander:hover { border-color: var(--border-hover) !important; }
details summary { color: var(--text-dim) !important; font-size: 13px !important; }

/* ── Data table ── */
.stDataFrame { border: 1px solid var(--border) !important; border-radius: var(--radius-md) !important; }
[data-testid="stDataFrame"] th { background: var(--bg-card) !important; color: var(--text-muted) !important; font-size: 11px !important; letter-spacing: 0.06em; text-transform: uppercase; }
[data-testid="stDataFrame"] td { font-size: 12px !important; color: var(--text-dim) !important; }

hr { border-color: var(--border) !important; margin: 1.6rem 0 !important; }

.stAlert { border-radius: var(--radius-md) !important; font-size: 13px !important; }
[data-testid="stInfo"] { background: rgba(57,217,138,0.06) !important; border-color: rgba(57,217,138,0.2) !important; }
[data-testid="stSuccess"] { background: rgba(168,255,62,0.05) !important; border-color: rgba(168,255,62,0.2) !important; }
[data-testid="stWarning"] { background: rgba(255,184,32,0.05) !important; border-color: rgba(255,184,32,0.2) !important; }

[data-testid="stSidebar"] .stButton > button {
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 9px 12px !important;
    width: 100% !important;
    border-radius: var(--radius-sm) !important;
}

/* ── Milestone card ── */
.milestone-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius-lg); padding: 20px 16px;
    text-align: center; position: relative; overflow: hidden; transition: all 0.2s;
}
.milestone-card.done { border-color: rgba(168,255,62,0.25) !important; background: rgba(168,255,62,0.02) !important; }
.milestone-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); }

/* ── Streak banner ── */
.streak-banner {
    border-radius: var(--radius-xl); padding: 22px 24px;
    display: flex; align-items: center; gap: 18px;
    margin-bottom: 20px; position: relative; overflow: hidden;
}

/* ── Danger zone ── */
.danger-zone {
    border: 1px solid rgba(255,92,92,0.15); border-radius: var(--radius-md);
    padding: 16px; background: rgba(255,92,92,0.02); margin-top: 16px;
}
.danger-title {
    color: var(--danger); font-size: 11px; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    margin-bottom: 10px; display: flex; align-items: center; gap: 6px;
}

/* ── Loading screen ── */
.loader-wrap {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; min-height: 85vh; text-align: center; padding: 40px;
}
.loader-wordmark {
    font-family: var(--font-display); font-size: 3.2rem; font-weight: 800;
    letter-spacing: -0.04em;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}
.loader-tagline { color: var(--text-muted); font-size: 13px; letter-spacing: 0.05em; margin-bottom: 50px; }
.loader-bar-track { width: 320px; height: 2px; background: rgba(255,255,255,0.06); border-radius: 100px; overflow: hidden; margin-bottom: 18px; }
.loader-bar-fill {
    height: 100%; border-radius: 100px;
    background: linear-gradient(90deg, var(--accent), var(--accent2), var(--accent3));
    background-size: 200%;
    animation: loadbar 2.8s cubic-bezier(0.4,0,0.2,1) forwards, shimmer 1.5s linear infinite;
}
@keyframes loadbar { 0%{width:0%} 30%{width:42%} 75%{width:80%} 100%{width:100%} }
@keyframes shimmer { 0%{background-position:0%} 100%{background-position:200%} }
.loader-status { color: var(--accent); font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.08em; margin-bottom: 40px; opacity: 0.8; }
.loader-badges { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; }
.loader-badge { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 100px; padding: 6px 16px; color: var(--text-muted); font-size: 10.5px; letter-spacing: 0.05em; font-family: var(--font-mono); }

/* ── Empty state ── */
.empty-state { text-align: center; padding: 48px 24px; background: var(--bg-card); border: 1px dashed var(--border); border-radius: var(--radius-xl); }
.empty-state-icon { font-size: 44px; margin-bottom: 16px; opacity: 0.6; }
.empty-state-title { font-family: var(--font-display); color: var(--text-primary); font-size: 16px; font-weight: 700; margin-bottom: 8px; }
.empty-state-desc { font-size: 13px; color: var(--text-muted); max-width: 300px; margin: 0 auto; }

/* ── BMI badge ── */
.bmi-badge { display: inline-flex; align-items: center; gap: 10px; padding: 10px 18px; border-radius: var(--radius-md); font-size: 13px; font-weight: 600; margin: 8px 0 0; }

/* ── Plan block ── */
.plan-block { background: var(--bg-input); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 18px; line-height: 1.9; font-size: 13.5px; color: var(--text-primary); }

/* ── PR badge ── */
.pr-badge { display: inline-flex; align-items: center; gap: 5px; background: rgba(255,184,32,0.08); border: 1px solid rgba(255,184,32,0.2); border-radius: 100px; padding: 2px 10px; font-size: 10px; color: var(--warn); font-weight: 600; font-family: var(--font-mono); vertical-align: middle; margin-left: 6px; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 100px; }

[data-testid="stForm"] { border: none !important; background: transparent !important; }
.stCaption { color: var(--text-muted) !important; font-size: 12px !important; }

@keyframes fadeUp { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }
.fade-in { animation: fadeUp 0.4s ease forwards; }

/* ════════════ HOME PAGE SPECIFIC ════════════ */

/* Hero */
.home-hero-badge {
    display: inline-flex; align-items: center; gap: 8px;
    font-family: var(--font-mono); font-size: 9.5px;
    letter-spacing: 0.12em; color: var(--accent);
    text-transform: uppercase; padding: 4px 12px;
    border: 1px solid rgba(57,217,138,0.2);
    border-radius: 100px; background: rgba(57,217,138,0.04);
    margin-bottom: 16px;
}
.home-hero-title {
    font-family: var(--font-display) !important;
    font-size: 3.2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.04em !important;
    line-height: 1.05 !important;
    color: var(--text-primary) !important;
    margin-bottom: 14px !important;
}
.home-hero-title .gradient-text {
    background: linear-gradient(135deg, var(--accent) 0%, #00b4d8 50%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.home-hero-sub {
    font-size: 15px !important;
    color: var(--text-dim) !important;
    line-height: 1.65 !important;
    max-width: 540px;
    margin-bottom: 26px !important;
}

/* Streak badge */
.home-streak-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(255,184,32,0.08); border: 1px solid rgba(255,184,32,0.2);
    border-radius: 100px; padding: 7px 16px;
    font-size: 12.5px; color: var(--warn); font-weight: 600;
    font-family: var(--font-body);
}

/* Quote strip */
.home-quote-strip {
    background: linear-gradient(135deg, rgba(0,229,255,0.03), rgba(139,92,246,0.03));
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent2);
    border-radius: 0 var(--radius-md) var(--radius-md) 0;
    padding: 14px 18px; display: flex; align-items: center; gap: 14px;
    margin-bottom: 28px;
}
.home-quote-strip .q-emoji { font-size: 22px; flex-shrink: 0; }
.home-quote-strip .q-text { font-style: italic; color: var(--text-dim); font-size: 13.5px; line-height: 1.65; }


/* Stat cards */
.home-stat-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius-md); padding: 18px 16px;
    position: relative; overflow: hidden; transition: all 0.2s;
}
.home-stat-card:hover { border-color: rgba(57,217,138,0.18); transform: translateY(-2px); box-shadow: 0 8px 32px rgba(0,0,0,0.5); }
.home-stat-card::after { content:''; position:absolute; inset:0; background:radial-gradient(ellipse at 0% 0%, rgba(0,229,255,0.03), transparent 60%); pointer-events:none; }
.home-stat-label { font-family: var(--font-mono); font-size: 9.5px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 8px; }
.home-stat-value { font-family: var(--font-display); font-size: 2.2rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1; margin-bottom: 4px; }
.home-stat-delta { font-size: 11px; color: var(--text-muted); }
.home-stat-delta.up { color: var(--accent3); }

/* Check items (home) */
.home-check-item {
    display: flex; align-items: center; gap: 12px;
    padding: 12px 14px; border-radius: var(--radius-sm);
    border: 1px solid var(--border); background: rgba(255,255,255,0.01);
    margin-bottom: 8px; transition: all 0.2s;
}
.home-check-item.done { border-color: rgba(168,255,62,0.18); background: rgba(168,255,62,0.02); }
.home-check-item:hover { border-color: var(--border-hover); }
.home-check-circle {
    width: 22px; height: 22px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; font-size: 11px; font-weight: 700;
}
.home-check-circle.done { background: var(--accent3); color: #020408; }
.home-check-circle.pending { background: rgba(255,255,255,0.04); border: 1px solid var(--border-hover); color: var(--text-muted); }
.home-check-label { font-size: 13px; font-weight: 500; color: var(--text-primary); }
.home-check-sub { font-size: 11px; color: var(--text-muted); margin-top: 1px; }
.home-check-item.done .home-check-label { color: rgba(240,244,248,0.65); }

/* Progress ring card */
.home-prog-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius-md); padding: 20px 22px;
}
.home-ring-stat { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.home-ring-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.home-ring-bar { height: 3px; background: rgba(255,255,255,0.06); border-radius: 100px; overflow: hidden; margin-top: 3px; }
.home-ring-fill { height: 100%; border-radius: 100px; }

/* Overload analysis */
.home-overload-strip { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius-md); overflow: hidden; }
.home-overload-row {
    display: flex; align-items: center; gap: 14px; padding: 12px 18px;
    border-bottom: 1px solid var(--border); font-size: 13px; transition: background 0.15s;
}
.home-overload-row:last-child { border-bottom: none; }
.home-overload-row:hover { background: rgba(255,255,255,0.015); }
.home-overload-ex { font-weight: 500; color: var(--text-primary); flex: 1; min-width: 0; }
.home-overload-msg { font-size: 12px; color: var(--text-muted); flex: 2; min-width: 0; }
.home-overload-status { font-size: 10px; font-family: var(--font-mono); letter-spacing: 0.05em; white-space: nowrap; }
.home-overload-status.green { color: var(--accent3); }
.home-overload-status.yellow { color: var(--warn); }
.home-overload-status.red { color: var(--danger); }

/* How it works */
.home-how-step {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius-md); padding: 20px 14px;
    text-align: center; position: relative; overflow: hidden; transition: all 0.2s;
}
.home-how-step::before {
    content:''; position:absolute; top:0;left:0;right:0; height:2px;
    background:linear-gradient(90deg,transparent,var(--accent),transparent);
    opacity:0; transition:opacity 0.2s;
}
.home-how-step:hover { border-color: rgba(57,217,138,0.2); transform: translateY(-3px); box-shadow: 0 12px 36px rgba(0,0,0,0.5); }
.home-how-step:hover::before { opacity: 1; }
.home-how-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 24px; height: 24px; background: rgba(57,217,138,0.08);
    border: 1px solid rgba(57,217,138,0.2); border-radius: 50%;
    font-family: var(--font-mono); font-size: 10px; font-weight: 500;
    color: var(--accent); margin: 0 auto 10px;
}
.home-how-icon { font-size: 28px; margin-bottom: 10px; display: block; }
.home-how-title { font-family: var(--font-display); font-size: 12.5px; font-weight: 700; color: var(--text-primary); margin-bottom: 6px; }
.home-how-desc { font-size: 11px; color: var(--text-muted); line-height: 1.6; }

/* CTA card */
.home-cta-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius-lg); padding: 28px;
    display: flex; align-items: center; gap: 22px;
    position: relative; overflow: hidden;
}
.home-cta-card::before {
    content:''; position:absolute; top:-60px;right:-60px;
    width:220px;height:220px;
    background:radial-gradient(circle,rgba(0,229,255,0.07),transparent 70%);
    pointer-events:none;
}
.home-cta-icon {
    width: 60px; height: 60px; background: rgba(57,217,138,0.08);
    border: 1px solid rgba(57,217,138,0.2); border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 28px; flex-shrink: 0;
}
.home-cta-title { font-family: var(--font-display); font-size: 18px; font-weight: 800; letter-spacing: -0.02em; color: var(--text-primary); margin-bottom: 6px; }
.home-cta-desc { font-size: 13px; color: var(--text-dim); line-height: 1.6; }

/* Volume bars */
.home-vol-bars {
    display: flex; align-items: flex-end; gap: 6px;
    height: 80px; background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius-md); padding: 14px 16px 10px; overflow: hidden;
}
.home-vol-bar-wrap { flex:1; display:flex; flex-direction:column; align-items:center; gap:4px; height:100%; justify-content:flex-end; }
.home-vol-bar { width:100%; border-radius:3px 3px 0 0; min-height:3px; }
.home-vol-bar-date { font-size:9px; color:var(--text-muted); font-family:var(--font-mono); }

/* PR cards */
.home-pr-card {
    background: var(--bg-card); border: 1px solid rgba(255,184,32,0.12);
    border-radius: var(--radius-md); padding: 14px; text-align: center; transition: all 0.2s;
}
.home-pr-card:hover { border-color: rgba(255,184,32,0.28); transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.4); }
.home-pr-weight { font-family: var(--font-display); font-size: 22px; font-weight: 800; letter-spacing: -0.03em; color: var(--warn); line-height: 1; margin: 4px 0; }
.home-pr-ex { font-size: 10.5px; color: var(--text-muted); line-height: 1.4; }
</style>
""", unsafe_allow_html=True)

# ── INIT ──────────────────────────────────────────────────────────────────────
init_state()

def scroll_to_top():
    st.markdown("""
        <script>
            var mainContainer = window.parent.document.querySelector('.main');
            if (mainContainer) mainContainer.scrollTo({top:0,behavior:'instant'});
            var appView = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
            if (appView) appView.scrollTo({top:0,behavior:'instant'});
            window.parent.scrollTo({top:0,behavior:'instant'});
        </script>
    """, unsafe_allow_html=True)

# ── LOADING SCREEN ────────────────────────────────────────────────────────────
if not st.session_state.app_ready:
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        st.markdown("""
        <div class="loader-wrap">
            <div style="font-size:58px;margin-bottom:18px;filter:drop-shadow(0 0 20px rgba(57,217,138,0.3))">🏋️</div>
            <div class="loader-wordmark">FitCoach AI</div>
            <div class="loader-tagline">LLM + RAG · Intelligent Fitness Coaching · Sport Science Grounded</div>
            <div class="loader-bar-track"><div class="loader-bar-fill"></div></div>
            <div class="loader-status">⚡ LOADING KNOWLEDGE BASE</div>
            <div class="loader-badges">
                <div class="loader-badge">🔬 Sport Science RAG</div>
                <div class="loader-badge">🤖 LLM Powered</div>
                <div class="loader-badge">📊 FAISS Vector Index</div>
                <div class="loader-badge">🎯 Personalised Plans</div>
                <div class="loader-badge">📈 Progress Analytics</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    rag_index, rag_chunks = build_rag_index()
    st.session_state.rag_index  = rag_index
    st.session_state.rag_chunks = rag_chunks
    st.session_state.app_ready  = True
    time.sleep(2.8)
    loading_placeholder.empty()
    st.rerun()
else:
    rag_index  = st.session_state.rag_index
    rag_chunks = st.session_state.rag_chunks

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 4px 16px">
        <div style="font-family:'Syne',sans-serif;font-size:20px;
             font-weight:800;color:#f0f4f8;letter-spacing:-0.03em">
            🏋️ FitCoach AI
        </div>
        <div style="font-size:10px;color:#3a4e65;letter-spacing:0.1em;
             text-transform:uppercase;margin-top:3px;font-family:'JetBrains Mono',monospace">
            Sport Science · LLM + RAG
        </div>
    </div>""", unsafe_allow_html=True)


    if "page" not in st.session_state:
        st.session_state.page = "🏠 Home"

    profile_done = bool(st.session_state.profile)
    plan_done    = bool(st.session_state.plan_ready)
    log_count    = len(st.session_state.workout_log)

    nav_items = [
        ("🏠", "Home",          True),
        ("👤", "Profile",       True),
        ("⚡", "Plan Generator", profile_done),
        ("📋", "Workout Log",   True),
        ("📊", "Dashboard",     log_count > 0),
        ("💬", "AI Coach",      plan_done),
    ]

    status_map = {
        "Home":           ("✅", "#10b981"),
        "Profile":        ("✅", "#10b981") if profile_done else ("·", "#3a4e65"),
        "Plan Generator": ("✅", "#10b981") if plan_done    else ("·", "#3a4e65"),
        "Workout Log":    ("✅", "#10b981") if log_count>0  else ("·", "#3a4e65"),
        "Dashboard":      ("✅", "#10b981") if log_count>0  else ("·", "#3a4e65"),
        "AI Coach":       ("✅", "#10b981") if plan_done    else ("·", "#3a4e65"),
    }

    for icon, label, enabled in nav_items:
        full = f"{icon} {label}"
        is_active = st.session_state.page == full
        if st.button(full, key=f"nav_{label}",
                     use_container_width=True,
                     type="primary" if is_active else "secondary",
                     disabled=not enabled):
            st.session_state.page = full
            st.rerun()

    st.divider()

    stats_s = get_stats()
    if not profile_done:
        st.markdown("""<div style="font-size:11px;color:#ffb820;padding:8px 10px;
        background:rgba(255,184,32,0.05);border:1px solid rgba(255,184,32,0.15);
        border-radius:8px;line-height:1.55">👆 Start with <strong>Profile</strong> to unlock your plan</div>""",
        unsafe_allow_html=True)
    elif not plan_done:
        st.markdown("""<div style="font-size:11px;color:#39d98a;padding:8px 10px;
        background:rgba(57,217,138,0.04);border:1px solid rgba(57,217,138,0.15);
        border-radius:8px;line-height:1.55">⚡ Profile set! Head to <strong>Plan Generator</strong> next</div>""",
        unsafe_allow_html=True)
    else:
        streak_msg, streak_icon, streak_color = get_streak_message(stats_s["streak"])
        st.markdown(f"""<div style="font-size:11px;color:{streak_color};padding:8px 10px;
        background:rgba(0,0,0,0.2);border:1px solid rgba(255,255,255,0.05);
        border-radius:8px;line-height:1.55">{streak_icon} <strong>{streak_msg}</strong></div>""",
        unsafe_allow_html=True)

    st.markdown(f"""<div style="margin-top:12px;font-size:10px;color:#3a4e65;
    font-family:'JetBrains Mono',monospace;letter-spacing:0.04em">
    ✅ Knowledge base · {len(rag_chunks)} vectors</div>""", unsafe_allow_html=True)

page = st.session_state.page

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME  
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    scroll_to_top()
    stats        = get_stats()
    profile_set  = bool(st.session_state.profile)
    plan_set     = bool(st.session_state.plan_ready)

    streak_msg, streak_icon, streak_color = get_streak_message(stats["streak"])

    tip_title, tip_desc = st.session_state.daily_tip
    tip_title = tip_title.replace("'", "&#39;").replace('"', "&quot;")
    tip_desc  = tip_desc.replace("'",  "&#39;").replace('"', "&quot;")

    now_hour     = datetime.now().hour
    greet        = "Good morning" if now_hour < 12 else "Good afternoon" if now_hour < 17 else "Good evening"
    profile_name = st.session_state.profile.get("name", "")
    greet_line   = f"{greet}, {profile_name}! 👋" if profile_name else f"{greet}! 👋"

    if not profile_set:
        next_step_icon   = "👤"
        next_step_text   = "Start by setting up your profile — it takes 2 minutes and shapes everything."
        next_step_color  = "#ffb820"
        next_step_bg     = "rgba(255,184,32,0.06)"
        next_step_border = "rgba(255,184,32,0.18)"
    elif not plan_set:
        next_step_icon   = "⚡"
        next_step_text   = "Profile complete! Head to Plan Generator to build your personalised 12-week plan."
        next_step_color  = "#39d98a"
        next_step_bg     = "rgba(57,217,138,0.05)"
        next_step_border = "rgba(57,217,138,0.18)"
    else:
        next_step_icon   = "💪"
        next_step_text   = f"You&#39;re on a {stats['streak']}-day streak — {streak_msg}. Keep the momentum going!"
        next_step_color  = streak_color
        next_step_bg     = "rgba(0,0,0,0.15)"
        next_step_border = f"{streak_color}30"

    # ── HERO ─────────────────────────────────────────────────────────────────
    # PART 1: Card open + background glows + eyebrow badge
    st.markdown(f"""
    <div class="fade-in" style="background:var(--bg-card);border:1px solid var(--border);
    border-radius:var(--radius-xl);padding:48px 44px;margin-bottom:20px;
    position:relative;overflow:hidden">
    <div style="position:absolute;top:-100px;right:-100px;width:400px;height:400px;
    background:radial-gradient(circle,rgba(57,217,138,0.05),transparent 65%);pointer-events:none"></div>
    <div style="position:absolute;bottom:-80px;left:-60px;width:300px;height:300px;
    background:radial-gradient(circle,rgba(139,92,246,0.04),transparent 65%);pointer-events:none"></div>
    <div style="display:inline-flex;align-items:center;gap:8px;font-family:var(--font-mono);
    font-size:9.5px;letter-spacing:0.14em;color:var(--accent);text-transform:uppercase;
    padding:4px 14px;border:1px solid rgba(57,217,138,0.2);border-radius:100px;
    background:rgba(57,217,138,0.04);margin-bottom:22px">
    <div class="rag-dot"></div>FitCoach AI · Sport Science Powered
    </div>
    <div style="font-family:var(--font-display);font-size:3rem;font-weight:800;
    letter-spacing:-0.04em;line-height:1.05;color:var(--text-primary);margin-bottom:6px">
    {greet_line}
    </div>
    <div style="font-size:16px;color:var(--text-dim);line-height:1.8;
    max-width:600px;margin-bottom:32px">
    FitCoach AI isn&#39;t a generic plan generator.<br>
    It reads <strong style="color:var(--text-primary)">peer-reviewed sport science</strong>,
    learns your body, your goals, and your limitations —
    then builds everything <em>around you</em>.
    Every workout, every macro, every recommendation is
    <strong style="color:var(--accent)">evidence-based and personalised</strong>.
    </div>
    """, unsafe_allow_html=True)

    # PART 2: 3 pillars
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:32px">
    <div style="background:rgba(57,217,138,0.04);border:1px solid rgba(57,217,138,0.12);
    border-radius:var(--radius-md);padding:18px 16px">
    <div style="font-size:22px;margin-bottom:10px">🧠</div>
    <div style="font-family:var(--font-display);font-size:15px;font-weight:700;    color:var(--text-primary);margin-bottom:6px">Knows the science</div>
    <div style="font-size:12px;color:var(--text-muted);line-height:1.7">
    RAG retrieves verified sport science for every plan and answer — not internet guesswork.
    </div></div>
    <div style="background:rgba(139,92,246,0.04);border:1px solid rgba(139,92,246,0.12);
    border-radius:var(--radius-md);padding:18px 16px">
    <div style="font-size:22px;margin-bottom:10px">🎯</div>
    <div style="font-family:var(--font-display);font-size:15px;font-weight:700;    color:var(--text-primary);margin-bottom:6px">Built around you</div>
    <div style="font-size:12px;color:var(--text-muted);line-height:1.65">
    Your age, goals, injuries, equipment and schedule shape every single recommendation.
    </div></div>
    <div style="background:rgba(255,184,32,0.04);border:1px solid rgba(255,184,32,0.12);
    border-radius:var(--radius-md);padding:18px 16px">
    <div style="font-size:22px;margin-bottom:10px">📈</div>
    <div style="font-family:var(--font-display);font-size:15px;font-weight:700;    color:var(--text-primary);margin-bottom:6px">Gets smarter over time</div>
    <div style="font-size:12px;color:var(--text-muted);line-height:1.65">
    The more you log, the more the AI adapts — detecting plateaus and updating your plan.
    </div></div>
    </div>
    """, unsafe_allow_html=True)

    # PART 3: Next step CTA + stats row
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;
    padding-top:24px;border-top:1px solid var(--border)">
    <div style="display:flex;align-items:center;gap:12px;
    background:{next_step_bg};border:1px solid {next_step_border};
    border-radius:var(--radius-lg);padding:14px 20px;flex:1;min-width:260px">
    <div style="font-size:24px">{next_step_icon}</div>
    <div>
    <div style="font-size:10px;font-family:var(--font-mono);letter-spacing:0.1em;
    text-transform:uppercase;color:var(--text-muted);margin-bottom:3px">Next step</div>
    <div style="font-size:13px;color:{next_step_color};font-weight:600;line-height:1.6">{next_step_text}</div>
    </div></div>
    <div style="display:flex;align-items:center;gap:20px">
    <div style="text-align:center">
    <div style="font-family:var(--font-display);font-size:28px;font-weight:800;
    color:{streak_color};letter-spacing:-0.03em;line-height:1">{stats['streak']}</div>
    <div style="font-size:12px;color:var(--text-muted);font-family:var(--font-mono);margin-top:2px">day streak</div>
    </div>
    <div style="width:1px;height:36px;background:var(--border)"></div>
    <div style="text-align:center">
    <div style="font-family:var(--font-display);font-size:28px;font-weight:800;
    color:var(--text-primary);letter-spacing:-0.03em;line-height:1">{stats['sessions']}</div>
    <div style="font-size:12px;color:var(--text-muted);font-family:var(--font-mono);margin-top:2px">sessions</div>
    </div>
    <div style="width:1px;height:36px;background:var(--border)"></div>
    <div style="text-align:center">
    <div style="font-family:var(--font-display);font-size:28px;font-weight:800;
    color:var(--accent3);letter-spacing:-0.03em;line-height:1">{len(rag_chunks)}</div>
    <div style="font-size:12px;color:var(--text-muted);font-family:var(--font-mono);margin-top:2px">sci. vectors</div>
    </div>
    </div></div>
    """, unsafe_allow_html=True)

    # ── HOW IT WORKS — shown first so new users know what to do ──────────────
    st.markdown('<div class="section-label">How FitCoach AI Works — Start Here</div>', unsafe_allow_html=True)
    steps = [
        ("👤", "01", "Build Profile",   "Your stats, goals & equipment",    "👤 Profile"),
        ("📚", "02", "RAG Retrieval",   "FAISS finds sport science",         None),
        ("⚡", "03", "Generate Plan",   "LLM builds your personalised plan", "⚡ Plan Generator"),
        ("📋", "04", "Log Sessions",    "Track workouts consistently",       "📋 Workout Log"),
        ("💬", "05", "Ask Coach",       "Real-time evidence guidance",       "💬 AI Coach"),
    ]
    how_cols = st.columns(5)
    for col, (icon, num, title, desc, nav_target) in zip(how_cols, steps):
        with col:
            st.markdown(f"""
            <div class="home-how-step">
                <div class="home-how-num">{num}</div>
                <span class="home-how-icon">{icon}</span>
                <div class="home-how-title">{title}</div>
                <div class="home-how-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)
            if nav_target:
                if st.button("Go →", key=f"home_how_{num}", use_container_width=True):
                    st.session_state.page = nav_target

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PROFILE  (Redesigned)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Profile":
    scroll_to_top()

    p = st.session_state.profile

    # ── Page Header ────────────────────────────────────────────────────────────
    profile_done = bool(p)
    st.markdown(f"""
    <div class="fade-in" style="margin-bottom:28px">
        <div style="display:inline-flex;align-items:center;gap:8px;
        font-family:var(--font-mono);font-size:9.5px;letter-spacing:0.12em;
        color:var(--accent2);text-transform:uppercase;padding:4px 12px;
        border:1px solid rgba(139,92,246,0.2);border-radius:100px;
        background:rgba(139,92,246,0.04);margin-bottom:16px">
            <div style="width:5px;height:5px;background:var(--accent2);border-radius:50%;animation:rag-pulse 2s infinite"></div>
            👤 Profile Setup
        </div>
        <h1 style="font-size:2.8rem!important;margin-bottom:10px!important">
            Your Profile
        </h1>
        <p style="font-size:15px!important;color:var(--text-dim)!important;max-width:520px;line-height:1.6!important;margin:0!important">
            Everything you enter shapes your personalised plan — accuracy means better results.
        </p>
    </div>""", unsafe_allow_html=True)

    # ── Info banner ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:28px">
        <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:12px;
        padding:14px 16px;display:flex;align-items:center;gap:12px">
            <div style="font-size:22px">🎯</div>
            <div>
                <div style="font-size:12px;font-weight:600;color:var(--text-primary)">Goals drive your plan</div>
                <div style="font-size:11px;color:var(--text-muted);margin-top:2px">RAG retrieves matching sport science</div>
            </div>
        </div>
        <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:12px;
        padding:14px 16px;display:flex;align-items:center;gap:12px">
            <div style="font-size:22px">🩹</div>
            <div>
                <div style="font-size:12px;font-weight:600;color:var(--text-primary)">Injuries are critical</div>
                <div style="font-size:11px;color:var(--text-muted);margin-top:2px">AI modifies exercises to keep you safe</div>
            </div>
        </div>
        <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:12px;
        padding:14px 16px;display:flex;align-items:center;gap:12px">
            <div style="font-size:22px">🔒</div>
            <div>
                <div style="font-size:12px;font-weight:600;color:var(--text-primary)">Saved locally</div>
                <div style="font-size:11px;color:var(--text-muted);margin-top:2px">Profile persists across sessions</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Form ───────────────────────────────────────────────────────────────────
    with st.form("profile_form"):
        # Body Stats section
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;margin:0 0 16px">
            <div style="width:28px;height:28px;background:rgba(57,217,138,0.08);
            border:1px solid rgba(57,217,138,0.2);border-radius:8px;
            display:flex;align-items:center;justify-content:center;font-size:14px">📏</div>
            <div style="font-family:var(--font-mono);font-size:10px;font-weight:700;
            letter-spacing:0.12em;text-transform:uppercase;color:var(--text-muted)">Body Stats</div>
            <div style="flex:1;height:1px;background:var(--border)"></div>
        </div>""", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        age    = c1.number_input("Age", min_value=10, max_value=90, value=int(p.get("age", 22)))
        weight = c2.number_input("Weight (kg)", min_value=30.0, max_value=300.0,
                                 value=float(p.get("weight", 70.0)), step=0.5)
        height = c3.number_input("Height (cm)", min_value=100.0, max_value=250.0,
                                 value=float(p.get("height", 170.0)), step=0.5)
        gender = c4.selectbox("Gender",
                    ["Male","Female","Non-binary","Prefer not to say"],
                    index=["Male","Female","Non-binary","Prefer not to say"].index(p.get("gender","Male")))

        # Live BMI
        bmi = round(weight / ((height / 100) ** 2), 1)
        bmi_cat = ("Underweight" if bmi < 18.5 else "Normal range" if bmi < 25 else "Overweight" if bmi < 30 else "Obese")
        bmi_color = ("#39d98a" if bmi < 18.5 else "#a8ff3e" if bmi < 25 else "#ffb820" if bmi < 30 else "#ff5c5c")
        bmi_bar_pct = min(100, int((bmi / 40) * 100))
        bmi_notes = {
            "Underweight": "Plan will prioritise building strength and healthy mass.",
            "Normal range": "Great baseline — plan will target your specific goals.",
            "Overweight":   "Plan will include a healthy calorie deficit and progressive training.",
            "Obese":        "Plan will start conservatively and build safely over time.",
        }
        st.markdown(f"""
        <div style="background:var(--bg-card);border:1px solid {bmi_color}22;border-radius:12px;
        padding:16px 20px;margin:12px 0 4px;display:flex;align-items:center;gap:20px">
            <div style="text-align:center;min-width:64px">
                <div style="font-family:var(--font-display);font-size:2rem;font-weight:800;
                color:{bmi_color};letter-spacing:-0.03em;line-height:1">{bmi}</div>
                <div style="font-size:10px;font-family:var(--font-mono);color:var(--text-muted);
                letter-spacing:0.06em;margin-top:3px">BMI</div>
            </div>
            <div style="width:1px;height:40px;background:var(--border)"></div>
            <div style="flex:1">
                <div style="font-size:13px;font-weight:600;color:{bmi_color};margin-bottom:4px">{bmi_cat}</div>
                <div style="font-size:11px;color:var(--text-muted);margin-bottom:8px">{bmi_notes[bmi_cat]}</div>
                <div style="height:4px;background:rgba(255,255,255,0.06);border-radius:100px;overflow:hidden">
                    <div style="width:{bmi_bar_pct}%;height:100%;background:{bmi_color};border-radius:100px;opacity:0.7"></div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)
        st.divider()

        # Goals & Training section
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;margin:0 0 16px">
            <div style="width:28px;height:28px;background:rgba(168,255,62,0.08);
            border:1px solid rgba(168,255,62,0.2);border-radius:8px;
            display:flex;align-items:center;justify-content:center;font-size:14px">🎯</div>
            <div style="font-family:var(--font-mono);font-size:10px;font-weight:700;
            letter-spacing:0.12em;text-transform:uppercase;color:var(--text-muted)">Goals & Training</div>
            <div style="flex:1;height:1px;background:var(--border)"></div>
        </div>""", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            level = st.selectbox("Fitness Level",
                ["Beginner","Intermediate","Advanced","Athlete"],
                index=["Beginner","Intermediate","Advanced","Athlete"].index(p.get("level","Beginner")),
                help="Be honest — the AI tailors intensity to your level.")
            days = st.select_slider("Training Days / Week",
                options=[2,3,4,5,6], value=int(p.get("days", 4)),
                help="How many days can you realistically commit to?")
            goals = st.multiselect("Fitness Goals", [
                "Lose weight","Build muscle","Improve endurance","Increase flexibility",
                "Athletic performance","General health","Stress relief"
            ], default=p.get("goals", ["General health"]))
            target = st.text_input("Specific Target", value=p.get("target",""),
                placeholder='e.g. "Lose 5kg in 3 months" or "Run 5km non-stop"',
                help="Concrete targets help the AI set realistic milestones.")
        with col_b:
            equipment = st.multiselect("Available Equipment", [
                "No equipment","Dumbbells","Barbell & plates","Resistance bands",
                "Kettlebells","Pull-up bar","Full gym access","Treadmill"
            ], default=p.get("equipment", ["No equipment"]),
            help="Only select what you actually have access to.")
            diet = st.multiselect("Dietary Preferences", [
                "No restrictions","Vegetarian","Vegan","Keto","Halal","Gluten-free"
            ], default=p.get("diet", ["No restrictions"]))
            injuries = st.text_area("Injuries / Limitations",
                value=p.get("injuries",""),
                placeholder="e.g. knee pain, lower back issues, shoulder impingement...\nLeave blank if none.",
                height=118,
                help="The AI modifies exercises to keep you training safely.")

        st.divider()
        submitted = st.form_submit_button("💾  Save Profile", use_container_width=True, type="primary")

    if submitted:
        st.session_state.profile = {
            "age": age, "weight": weight, "height": height, "gender": gender,
            "level": level, "days": days, "goals": goals, "target": target,
            "equipment": equipment, "diet": diet, "injuries": injuries, "bmi": bmi
        }
        save_profile()
        st.markdown("""
        <div style="background:rgba(168,255,62,0.04);border:1px solid rgba(168,255,62,0.2);
        border-radius:12px;padding:14px 18px;display:flex;align-items:center;gap:12px;margin-bottom:16px">
            <div style="font-size:20px">✅</div>
            <div>
                <div style="font-size:15px;font-weight:600;color:var(--accent3)">Profile saved successfully!</div>
                <div style="font-size:14px;color:var(--text-muted);margin-top:2px">Your data is stored locally and will persist across sessions.</div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Summary card
        goal_chips = "".join([f'<span style="display:inline-block;padding:3px 10px;background:rgba(57,217,138,0.06);border:1px solid rgba(57,217,138,0.15);border-radius:100px;font-size:11px;color:var(--accent);margin:2px">{g}</span>' for g in goals])
        equip_chips = "".join([f'<span style="display:inline-block;padding:3px 10px;background:rgba(168,255,62,0.06);border:1px solid rgba(168,255,62,0.15);border-radius:100px;font-size:11px;color:var(--accent3);margin:2px">{e}</span>' for e in equipment])
        st.markdown(f"""
        <div style="background:var(--bg-card);border:1px solid rgba(57,217,138,0.15);border-radius:14px;
        padding:20px 22px;margin-bottom:20px">
            <div style="font-family:var(--font-display);font-size:15px;font-weight:700;            color:var(--accent3);margin-bottom:16px;letter-spacing:-0.01em">📋 Profile Summary</div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:16px">
                <div style="background:rgba(255,255,255,0.02);border-radius:8px;padding:12px">
                    <div style="font-size:10px;color:var(--text-muted);font-family:var(--font-mono);
                    letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px">Age / Gender</div>
                    <div style="font-size:15px;font-weight:700;color:var(--text-primary)">{age} · {gender}</div>
                </div>
                <div style="background:rgba(255,255,255,0.02);border-radius:8px;padding:12px">
                    <div style="font-size:10px;color:var(--text-muted);font-family:var(--font-mono);
                    letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px">Body</div>
                    <div style="font-size:15px;font-weight:700;color:var(--text-primary)">{weight}kg · {height}cm</div>
                </div>
                <div style="background:rgba(255,255,255,0.02);border-radius:8px;padding:12px">
                    <div style="font-size:10px;color:var(--text-muted);font-family:var(--font-mono);
                    letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px">BMI · Level</div>
                    <div style="font-size:15px;font-weight:700;color:{bmi_color}">{bmi}</div>
                    <div style="font-size:11px;color:var(--text-dim);margin-top:2px">{level} · {days}×/week</div>
                </div>
            </div>
            <div style="margin-bottom:10px">
                <div style="font-size:10px;color:var(--text-muted);font-family:var(--font-mono);
                letter-spacing:0.08em;text-transform:uppercase;margin-bottom:6px">Goals</div>
                {goal_chips}
            </div>
            <div>
                <div style="font-size:10px;color:var(--text-muted);font-family:var(--font-mono);
                letter-spacing:0.08em;text-transform:uppercase;margin-bottom:6px">Equipment</div>
                {equip_chips}
            </div>
            {f'<div style="margin-top:12px;padding:10px 14px;background:rgba(255,184,32,0.05);border:1px solid rgba(255,184,32,0.15);border-radius:8px;font-size:12px;color:var(--warn)">⚠️ Injuries noted: {injuries}</div>' if injuries.strip() else ''}
        </div>""", unsafe_allow_html=True)

        col_cta, col_b2, _ = st.columns([1.5, 1.5, 2])
        with col_cta:
            if st.button("⚡ Generate My Plan →", type="primary", use_container_width=True):
                st.session_state.page = "⚡ Plan Generator"; (); 
        with col_b2:
            if st.button("📋 Log a Workout", use_container_width=True):
                st.session_state.page = "📋 Workout Log"; 

    # ── Danger Zone 
    st.divider()
    st.markdown("""
    <div class="danger-zone">
        <div class="danger-title">⚠ Danger Zone</div>
        <div style="font-size:12px;color:var(--text-muted);margin-bottom:12px">
        Resetting your profile will also clear your saved plan. Workout logs are kept.
        </div>
    </div>""", unsafe_allow_html=True)

    if "confirm_reset_profile" not in st.session_state:
        st.session_state.confirm_reset_profile = False

    if st.session_state.confirm_reset_profile:
        st.warning("⚠️ This will delete your profile and plan. Are you sure?")
        c1, c2 = st.columns([1, 3])
        if c1.button("✅ Yes, Reset", type="primary", use_container_width=True):
            st.session_state.profile = {}
            st.session_state.plan = ""
            st.session_state.plan_ready = False
            if PROFILE_FILE.exists(): PROFILE_FILE.unlink()
            if PLAN_FILE.exists(): PLAN_FILE.unlink()
            st.session_state.confirm_reset_profile = False
            st.success("Profile and plan cleared.")
            st.rerun()
        if c2.button("❌ Cancel", use_container_width=True):
            st.session_state.confirm_reset_profile = False
            st.rerun()
    else:
        if st.button("🗑️ Reset Profile & Plan", use_container_width=True):
            st.session_state.confirm_reset_profile = True
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PLAN GENERATOR  (Redesigned)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Plan Generator":
    scroll_to_top()

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="fade-in" style="margin-bottom:28px">
        <div style="display:inline-flex;align-items:center;gap:8px;
        font-family:var(--font-mono);font-size:9.5px;letter-spacing:0.12em;
        color:var(--accent);text-transform:uppercase;padding:4px 12px;
        border:1px solid rgba(57,217,138,0.2);border-radius:100px;
        background:rgba(57,217,138,0.04);margin-bottom:16px">
            <div style="width:5px;height:5px;background:var(--accent);border-radius:50%;animation:rag-pulse 2s infinite"></div>
            ⚡ LLM + RAG · Sport Science Grounded
        </div>
        <h1 style="font-size:2.8rem!important;margin-bottom:10px!important">Plan Generator</h1>
        <p style="font-size:15px!important;color:var(--text-dim)!important;max-width:520px;line-height:1.6!important;margin:0!important">
            Your personalised 12-week fitness plan — built from your profile and verified sport science.
        </p>
    </div>""", unsafe_allow_html=True)

    if not st.session_state.profile:
        st.markdown("""
        <div style="text-align:center;padding:60px 24px;background:var(--bg-card);
        border:1px dashed var(--border);border-radius:20px">
            <div style="font-size:48px;margin-bottom:16px;opacity:0.5">👤</div>
            <div style="font-family:var(--font-display);font-size:18px;font-weight:700;
            color:var(--text-primary);margin-bottom:8px">Profile Required</div>
            <div style="font-size:13px;color:var(--text-muted);max-width:300px;margin:0 auto 20px">
            Fill in your profile first so the AI can create a plan that's truly personalised to you.</div>
        </div>""", unsafe_allow_html=True)
        if st.button("👤 Set Up Profile →", type="primary"):
            st.session_state.page = "👤 Profile";
        st.stop()

    profile = st.session_state.profile

    # ── Profile + RAG context in side-by-side expanders ───────────────────────
    col_snap, col_rag = st.columns(2)
    with col_snap:
        with st.expander("📋 Active profile snapshot", expanded=False):
            c1, c2 = st.columns(2)
            c1.metric("Age / Gender",    f"{profile.get('age','?')} · {profile.get('gender','?')}")
            c2.metric("Level",           profile.get("level","—"))
            c3, c4 = st.columns(2)
            c3.metric("Weight / Height", f"{profile.get('weight','?')}kg · {profile.get('height','?')}cm")
            c4.metric("BMI",             profile.get("bmi","—"))
            goal_str = ", ".join(profile.get("goals",[]))
            st.markdown(f'<div style="font-size:12px;color:var(--text-dim);margin-top:8px">🎯 {goal_str}</div>', unsafe_allow_html=True)
            if profile.get("injuries"):
                st.markdown(f'<div style="margin-top:8px;padding:8px 12px;background:rgba(255,184,32,0.05);border:1px solid rgba(255,184,32,0.15);border-radius:8px;font-size:12px;color:var(--warn)">⚠️ {profile["injuries"]}</div>', unsafe_allow_html=True)

    query   = f"{' '.join(profile.get('goals',[]))} {profile.get('level','')} training plan"
    rag_ctx = retrieve(query, rag_index, rag_chunks)

    with col_rag:
        with st.expander("📚 RAG knowledge retrieved", expanded=False):
            st.caption("These sport science excerpts ground your plan.")
            st.code(rag_ctx[:800] + "\n...", language="text")

    st.divider()

    # ── Generate button row ────────────────────────────────────────────────────
    btn_label = "⚡  Regenerate Plan" if st.session_state.plan_ready else "⚡  Generate My Personalised Plan"

    if st.session_state.plan_ready:
        # Show plan status banner
        st.markdown("""
        <div style="background:rgba(168,255,62,0.03);border:1px solid rgba(168,255,62,0.15);
        border-radius:12px;padding:14px 18px;display:flex;align-items:center;
        justify-content:space-between;margin-bottom:16px">
            <div style="display:flex;align-items:center;gap:12px">
                <div style="font-size:20px">✅</div>
                <div>
                    <div style="font-size:13px;font-weight:600;color:var(--accent3)">Plan ready</div>
                    <div style="font-size:11px;color:var(--text-muted)">Your personalised plan is saved and active. Regenerate anytime.</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    col_gen, col_dl, col_clear = st.columns([3,1,1])
    with col_gen:
        if st.button(btn_label, type="primary", use_container_width=True):
            with st.spinner("🤖  Analysing profile and building your plan..."):
                plan = generate_plan(profile, rag_ctx)
            st.session_state.plan = plan
            st.session_state.plan_ready = True
            save_plan()
            st.success("✅ Your plan is ready!")
            st.balloons()

    if st.session_state.plan_ready and st.session_state.plan:
        with col_dl:
            st.download_button("💾 Download",
                data=st.session_state.plan.encode(),
                file_name=f"fitcoach_plan_{date.today()}.txt",
                mime="text/plain", use_container_width=True)
        with col_clear:
            if st.button("🗑️ Delete", use_container_width=True):
                st.session_state.plan = ""
                st.session_state.plan_ready = False
                if PLAN_FILE.exists(): PLAN_FILE.unlink()
                st.rerun()

    # ── Plan sections ──────────────────────────────────────────────────────────
    if st.session_state.plan_ready and st.session_state.plan:
        plan_text = st.session_state.plan
        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

        SECTIONS = [
            ("## WEEKLY WORKOUT PLAN",           "🏋️", "Weekly Workout Plan",        "#39d98a",  True),
            ("## PROGRESSIVE OVERLOAD STRATEGY", "📈", "Progressive Overload",        "#a8ff3e",  False),
            ("## NUTRITION GUIDE",               "🥗", "Nutrition Guide",             "#8b5cf6",  True),
            ("## RECOVERY PROTOCOL",             "😴", "Recovery Protocol",           "#39d98a",  False),
            ("## 12-WEEK MILESTONES",            "🎯", "12-Week Milestones",          "#ffb820",  False),
            ("## COACH'S NOTES",                 "💡", "Coach's Notes",               "#a8ff3e",  False),
        ]
        DESCS = {
            "Weekly Workout Plan":   "Day-by-day training schedule with exercises, sets, reps, and coaching cues.",
            "Progressive Overload":  "How to increase intensity week by week so you keep making progress.",
            "Nutrition Guide":       "Calorie targets, macro breakdown, meal timing, and a sample day.",
            "Recovery Protocol":     "Sleep targets, stretching routine, and active recovery tips.",
            "12-Week Milestones":    "Concrete targets at weeks 2, 4, 8, and 12 to keep you on track.",
            "Coach's Notes":         "Personalised tips written specifically for your profile and goals.",
        }

        found_any = False
        for header, icon, title, color, expanded in SECTIONS:
            pat = re.compile(re.escape(header) + r"([\s\S]*?)(?=## |\Z)", re.I)
            m   = pat.search(plan_text)
            if m:
                found_any = True
                # Custom styled expander header
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:8px;margin-top:6px;padding:10px 14px;
                background:var(--bg-card);border:1px solid {color}18;border-radius:10px 10px 0 0;
                border-bottom:none">
                    <span style="font-size:16px">{icon}</span>
                    <span style="font-family:var(--font-display);font-size:13.5px;font-weight:700;
                    color:var(--text-primary)">{title}</span>
                    <span style="font-size:11px;color:var(--text-muted);margin-left:4px">— {DESCS[title]}</span>
                </div>""", unsafe_allow_html=True)
                with st.expander("", expanded=expanded):
                    import re as _re
                    section_text = m.group(1).strip()
                    lines = section_text.split("\n")

                    # Separate table lines from non-table lines
                    table_lines = []
                    pre_lines   = []
                    post_lines  = []
                    in_table    = False
                    after_table = False

                    for line in lines:
                        stripped = line.strip()
                        is_table_row = stripped.startswith("|") or _re.match(r"^\|?[-:]+\|", stripped)
                        if is_table_row:
                            in_table    = True
                            after_table = False
                            table_lines.append(line)
                        elif in_table and not after_table:
                            after_table = True
                            post_lines.append(line)
                        elif not in_table:
                            pre_lines.append(line)
                        else:
                            post_lines.append(line)

                    # Render non-table text above the table
                    if pre_lines:
                        pre_text = "\n".join(pre_lines).replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
                        st.markdown(pre_text)

                    # Convert markdown table → HTML table with <br> as real line breaks
                    if table_lines:
                        # Strip separator row (---|---) for parsing
                        data_rows = [r for r in table_lines if not _re.match(r"^\s*\|?\s*[-:]+\s*\|", r)]

                        html_rows = ""
                        for row_idx, row in enumerate(data_rows):
                            # Split by | and clean up
                            cells = [c.strip() for c in row.strip().strip("|").split("|")]
                            tag = "th" if row_idx == 0 else "td"
                            html_cells = ""
                            for cell in cells:
                                # Replace <br> variants with actual HTML <br> tag
                                cell_html = (cell
                                .replace("<br />", "<br>")
                                .replace("<br/>", "<br>")
                                )
                                # Split numbered items separated by " · " back into per-line
                                cell_html = _re.sub(r"\s*·\s*(\d+\.)", r"<br>\1", cell_html)
                                # Convert markdown bold (**text**) to HTML <strong>
                                cell_html = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", cell_html)
                                # Convert markdown italic (*text*) to HTML <em>
                                cell_html = _re.sub(r"\*(.+?)\*", r"<em>\1</em>", cell_html)
                                html_cells += (
                                    f'<{tag} style="padding:10px 14px;border:1px solid #131d2e;'
                                    f'vertical-align:top;font-size:13px;color:{"#f0f4f8" if row_idx==0 else "#7a8fa8"};'
                                    f'font-weight:{"700" if row_idx==0 else "400"};'
                                    f'background:{"#0c1220" if row_idx==0 else "#080d17"};'
                                    f'line-height:1.8">{cell_html}</{tag}>'
                                )
                            html_rows += f"<tr>{html_cells}</tr>"

                        html_table = (
                            '<div style="overflow-x:auto;margin:12px 0">'
                            '<table style="width:100%;border-collapse:collapse;'
                            'font-family:Outfit,sans-serif;border:1px solid #131d2e;border-radius:10px;overflow:hidden">'
                            f"{html_rows}"
                            "</table></div>"
                        )
                        st.markdown(html_table, unsafe_allow_html=True)

                    # Render any text that came after the table
                    if post_lines:
                        post_text = "\n".join(post_lines).replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
                        st.markdown(post_text)

        if not found_any:
            st.markdown(f'<div class="plan-block">{plan_text}</div>', unsafe_allow_html=True)

        st.divider()

        # ── Next step CTA ──────────────────────────────────────────────────────
        st.markdown("""
        <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:14px;
        padding:22px 24px;display:flex;align-items:center;gap:20px;margin-bottom:16px">
            <div style="font-size:32px">🚀</div>
            <div style="flex:1">
                <div style="font-family:var(--font-display);font-size:15px;font-weight:700;
                color:var(--text-primary);margin-bottom:4px">Plan ready — time to execute</div>
                <div style="font-size:13px;color:var(--text-dim)">
                Start logging workouts to track your progress, or ask your AI coach any questions about the plan.
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
        col_chat, col_log = st.columns(2)
        with col_chat:
            if st.button("💬 Ask your coach about this plan", type="primary", use_container_width=True):
                st.session_state.page = "💬 AI Coach"; 
        with col_log:
            if st.button("📋 Start logging workouts", use_container_width=True):
                st.session_state.page = "📋 Workout Log";

    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 24px;background:var(--bg-card);
        border:1px dashed var(--border);border-radius:20px;margin-top:8px">
            <div style="font-size:48px;margin-bottom:16px;opacity:0.5">⚡</div>
            <div style="font-family:var(--font-display);font-size:18px;font-weight:700;
            color:var(--text-primary);margin-bottom:8px">Ready to build your plan?</div>
            <div style="font-size:13px;color:var(--text-muted);max-width:340px;margin:0 auto">
            Click the button above. The AI will retrieve relevant sport science and build a fully personalised 12-week plan.</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: WORKOUT LOG  (Redesigned)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Workout Log":
    scroll_to_top()

    stats = get_stats()
    streak_msg, streak_icon, streak_color = get_streak_message(stats["streak"])

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="fade-in" style="margin-bottom:24px">
        <div style="display:inline-flex;align-items:center;gap:8px;
        font-family:var(--font-mono);font-size:9.5px;letter-spacing:0.12em;
        color:#a8ff3e;text-transform:uppercase;padding:4px 12px;
        border:1px solid rgba(168,255,62,0.2);border-radius:100px;
        background:rgba(168,255,62,0.04);margin-bottom:16px">
            <div style="width:5px;height:5px;background:#a8ff3e;border-radius:50%"></div>
            📋 Session Tracker
        </div>
        <h1 style="font-size:2.8rem!important;margin-bottom:10px!important">Workout Log</h1>
        <p style="font-size:15px!important;color:var(--text-dim)!important;max-width:520px;line-height:1.6!important;margin:0!important">
            Every session logged is data that makes your AI recommendations smarter.
        </p>
    </div>""", unsafe_allow_html=True)

    # ── Last logged banner ─────────────────────────────────────────────────────
    last = st.session_state.last_logged
    if last:
        if last.get("category") == "Cardio":
            last_str = f"{last['exercise']} · {last['duration']}min · {last['distance']}km"
        else:
            last_str = f"{last['exercise']} · {last['sets']}×{last['reps']} @ {last['weight']}kg · {last['volume']}kg vol"
        st.markdown(f"""
        <div style="background:rgba(0,229,255,0.03);border:1px solid rgba(57,217,138,0.12);
        border-radius:10px;padding:11px 16px;display:flex;align-items:center;gap:12px;margin-bottom:20px">
            <div style="font-size:16px">⚡</div>
            <div style="font-size:12px;color:var(--text-dim)">
                <span style="color:var(--accent);font-weight:600">Last logged:</span> {last_str}
                <span style="color:var(--text-muted);margin-left:8px">{last['date']}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Log form ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:14px;
    padding:18px 20px 10px;margin-bottom:4px">
        <div style="font-family:var(--font-display);font-size:16px;font-weight:700;
        color:var(--text-primary);margin-bottom:3px">What did you crush today? 🔥</div>
        <div style="font-size:12px;color:var(--text-muted);margin-bottom:14px">
        Select a category, pick an exercise, and log your numbers.</div>
    </div>""", unsafe_allow_html=True)

    tab_s, tab_c, tab_f, tab_r = st.tabs(["🏋️ Strength", "🏃 Cardio", "🧘 Flexibility", "😴 Recovery"])

    def log_success_banner(entry):
        if entry["category"] == "Cardio":
            st.success(f"🏃 Logged: **{entry['exercise']}** — {entry['duration']} min | {entry['distance']} km. Nice work!")
        else:
            st.success(f"💪 Logged: **{entry['exercise']}** — {entry['sets']}×{entry['reps']} @ {entry['weight']}kg | Volume: **{entry['volume']}kg**. Keep it up!")

    with tab_s:
        st.markdown("**Quick-select an exercise:**")
        quick_cols = st.columns(8)
        for i, ex in enumerate(EXERCISE_QUICK_LOG["Strength"]):
            if quick_cols[i % 8].button(ex, key=f"qs_{ex}", use_container_width=True):
                st.session_state["_quick_ex"] = ex
        with st.form("log_strength", clear_on_submit=True):
            default_ex = st.session_state.get("_quick_ex", "")
            exercise = st.text_input("Exercise Name", value=default_ex,
                                     placeholder="e.g. Bench Press, Squat, Deadlift...")
            col_s, col_r, col_w, col_n = st.columns([1,1,1,2])
            sets   = col_s.number_input("Sets",        min_value=1, max_value=30, value=3)
            reps   = col_r.number_input("Reps",        min_value=1, max_value=100, value=10)
            weight = col_w.number_input("Weight (kg)", min_value=0.0, max_value=500.0, value=0.0, step=0.5, help="0 = bodyweight")
            notes  = col_n.text_input("Notes", placeholder="PR? Form note?")
            if st.form_submit_button("➕  Log Strength Session", type="primary", use_container_width=True):
                if not exercise.strip():
                    st.warning("⚠️ Please enter an exercise name.")
                else:
                    entry = add_entry(exercise.strip(), sets, reps, weight, "Strength", notes)
                    log_success_banner(entry)
                    if "_quick_ex" in st.session_state: del st.session_state["_quick_ex"]
                    st.rerun()

    with tab_c:
        st.markdown("**Quick-select an exercise:**")
        qc_cols = st.columns(7)
        for i, ex in enumerate(EXERCISE_QUICK_LOG["Cardio"]):
            if qc_cols[i % 7].button(ex, key=f"qc_{ex}", use_container_width=True):
                st.session_state["_quick_ex_c"] = ex
        with st.form("log_cardio", clear_on_submit=True):
            default_ex_c = st.session_state.get("_quick_ex_c", "")
            exercise_c = st.text_input("Exercise Name", value=default_ex_c,
                                       placeholder="e.g. Running, Cycling, Jump Rope...")
            col_d, col_dist, col_pace, col_n2 = st.columns([1,1,1,2])
            duration  = col_d.number_input("Duration (min)", min_value=0, max_value=300, value=30)
            distance  = col_dist.number_input("Distance (km)", min_value=0.0, max_value=200.0, value=0.0, step=0.1)
            pace_note = col_pace.text_input("Pace / Intensity", placeholder="e.g. Zone 2")
            notes_c   = col_n2.text_input("Notes", placeholder="How did it feel?")
            if st.form_submit_button("➕  Log Cardio Session", type="primary", use_container_width=True):
                if not exercise_c.strip():
                    st.warning("⚠️ Please enter an exercise name.")
                else:
                    full_notes = f"{pace_note} | {notes_c}".strip(" |") if pace_note else notes_c
                    entry = add_entry(exercise_c.strip(), 0, 0, 0.0, "Cardio", full_notes, duration=duration, distance=distance)
                    log_success_banner(entry)
                    if "_quick_ex_c" in st.session_state: del st.session_state["_quick_ex_c"]
                    st.rerun()

    with tab_f:
        with st.form("log_flex", clear_on_submit=True):
            col_fx1, col_fx2, col_fx3 = st.columns([3,1,2])
            exercise_f = col_fx1.text_input("Session Name", placeholder="e.g. Full body stretch, Hip mobility...")
            dur_f      = col_fx2.number_input("Duration (min)", min_value=0, max_value=120, value=20)
            notes_f    = col_fx3.text_input("Notes", placeholder="Focus areas...")
            if st.form_submit_button("➕  Log Flexibility Session", type="primary", use_container_width=True):
                if not exercise_f.strip():
                    st.warning("⚠️ Please enter a session name.")
                else:
                    add_entry(exercise_f.strip(), 0, 0, 0.0, "Flexibility", notes_f, duration=dur_f)
                    st.success(f"🧘 Logged: **{exercise_f}** — {dur_f} min. Great for recovery!")
                    st.rerun()

    with tab_r:
        with st.form("log_recovery", clear_on_submit=True):
            col_rv1, col_rv2, col_rv3 = st.columns([3,1,2])
            exercise_rv = col_rv1.text_input("Recovery Activity", placeholder="e.g. Active Recovery Walk, Ice Bath...")
            dur_rv      = col_rv2.number_input("Duration (min)", min_value=0, max_value=120, value=15)
            notes_rv    = col_rv3.text_input("Notes", placeholder="How are you feeling?")
            if st.form_submit_button("➕  Log Recovery Session", type="primary", use_container_width=True):
                if not exercise_rv.strip():
                    st.warning("⚠️ Please enter an activity name.")
                else:
                    add_entry(exercise_rv.strip(), 0, 0, 0.0, "Recovery", notes_rv, duration=dur_rv)
                    st.success(f"😴 Logged: **{exercise_rv}** — {dur_rv} min. Recovery is training too!")
                    st.rerun()

    st.divider()

    # ── Progressive Overload Analysis ─────────────────────────────────────────
    st.markdown('<div class="section-label">Progressive Overload Analysis</div>', unsafe_allow_html=True)
    checks = overload_check()
    if not checks:
        st.markdown("""
        <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:12px;
        padding:20px 24px;display:flex;align-items:center;gap:14px">
            <div style="font-size:28px;opacity:0.4">📊</div>
            <div>
                <div style="font-size:13px;font-weight:500;color:var(--text-primary)">Not enough data yet</div>
                <div style="font-size:12px;color:var(--text-muted);margin-top:2px">Log at least 3 sessions of the same exercise to see overload analysis.</div>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        status_css = {"progressing": "green", "plateau": "yellow", "regressing": "red"}
        status_lbl = {"progressing": "PROGRESSING ↑", "plateau": "PLATEAU —", "regressing": "REGRESSING ↓"}
        status_bg  = {
            "progressing": "rgba(168,255,62,0.03)",
            "plateau":     "rgba(255,184,32,0.03)",
            "regressing":  "rgba(255,92,92,0.03)",
        }
        status_border = {
            "progressing": "rgba(168,255,62,0.15)",
            "plateau":     "rgba(255,184,32,0.15)",
            "regressing":  "rgba(255,92,92,0.15)",
        }
        rows_html = ""
        for c in checks[:6]:
            css = status_css.get(c["status"], "green")
            lbl = status_lbl.get(c["status"], c["status"].upper())
            bg  = status_bg.get(c["status"], "rgba(168,255,62,0.03)")
            bd  = status_border.get(c["status"], "rgba(168,255,62,0.15)")
            rows_html += f"""
            <div class="home-overload-row" style="background:{bg};border-left:2px solid {bd}">
                <span style="font-size:14px">{c['icon']}</span>
                <div class="home-overload-ex">{c['ex']}</div>
                <div class="home-overload-msg">{c['msg']}</div>
                <div class="home-overload-status {css}">{lbl}</div>
            </div>"""
        st.markdown(f'<div class="home-overload-strip">{rows_html}</div>', unsafe_allow_html=True)

    st.divider()

    # ── Exercise History ───────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Exercise History</div>', unsafe_allow_html=True)
    df = pd.DataFrame(st.session_state.workout_log)

    if df.empty:
        st.markdown("""
        <div style="text-align:center;padding:48px 24px;background:var(--bg-card);
        border:1px dashed var(--border);border-radius:20px">
            <div style="font-size:44px;margin-bottom:16px;opacity:0.5">📋</div>
            <div style="font-family:var(--font-display);font-size:16px;font-weight:700;
            color:var(--text-primary);margin-bottom:8px">No sessions logged yet</div>
            <div style="font-size:13px;color:var(--text-muted)">Use the forms above to log your first workout. Every session counts! 💪</div>
        </div>""", unsafe_allow_html=True)
    else:
        col_f1, col_f2, col_f3 = st.columns([2,2,1])
        cats       = df["category"].unique().tolist()
        cat_filter = col_f1.multiselect("Filter by category", cats, default=cats)
        ex_filter  = col_f2.text_input("Search exercise", placeholder="Type to filter...")
        col_f3.download_button("📥 Export CSV",
            data=df.to_csv(index=False).encode(),
            file_name=f"workout_log_{date.today()}.csv", mime="text/csv")

        filt = df[df["category"].isin(cat_filter)]
        if ex_filter:
            filt = filt[filt["exercise"].str.contains(ex_filter, case=False, na=False)]
        for col in ["duration","distance"]:
            if col not in filt.columns: filt[col] = 0

        st.dataframe(
            filt[["date","exercise","category","sets","reps","weight","volume",
                  "duration","distance","notes"]].sort_index(ascending=False),
            use_container_width=True, hide_index=True,
            column_config={
                "volume":   st.column_config.NumberColumn("Volume (kg)",    format="%.1f"),
                "weight":   st.column_config.NumberColumn("Weight (kg)",    format="%.1f"),
                "duration": st.column_config.NumberColumn("Duration (min)", format="%d"),
                "distance": st.column_config.NumberColumn("Distance (km)",  format="%.1f"),
            }
        )

        # Delete entry
        st.markdown("""
        <div style="border:1px solid rgba(255,92,92,0.15);border-radius:12px;
        padding:16px 18px;background:rgba(255,92,92,0.02);margin-top:16px">
            <div style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
            color:var(--danger);margin-bottom:10px;display:flex;align-items:center;gap:6px">
                ⚠ Remove an Entry
            </div>
            <div style="font-size:12px;color:var(--text-muted);margin-bottom:12px">
            Logged the wrong weight or exercise? Select and remove it here. This cannot be undone.
            </div>""", unsafe_allow_html=True)

        if "confirm_delete_id" not in st.session_state:
            st.session_state.confirm_delete_id = None

        deletable = filt[filt["id"].notna()].reset_index(drop=True) if "id" in filt.columns else filt.copy().reset_index(drop=True)
        if not deletable.empty:
            def make_label(row):
                if row.get("category") == "Cardio":
                    return f"#{int(row['id'])} · {row['date']} · {row['exercise']} · {row['duration']}min"
                return f"#{int(row['id'])} · {row['date']} · {row['exercise']} · {row['sets']}×{row['reps']} @ {row['weight']}kg"
            options = {make_label(row): row["id"] for _, row in deletable.iterrows()}
            selected_label = st.selectbox("Select entry to remove", options=list(options.keys()),
                                          index=None, placeholder="Choose an entry...")
            if selected_label:
                selected_id = options[selected_label]
                if st.session_state.confirm_delete_id == selected_id:
                    c_yes, c_no = st.columns([1,3])
                    if c_yes.button("✅ Confirm Delete", type="primary", use_container_width=True):
                        st.session_state.workout_log = [e for e in st.session_state.workout_log if e.get("id") != selected_id]
                        save_log()
                        st.session_state.confirm_delete_id = None
                        st.success(f"Entry #{int(selected_id)} removed.")
                        st.rerun()
                    if c_no.button("❌ Cancel", use_container_width=True):
                        st.session_state.confirm_delete_id = None
                        st.rerun()
                else:
                    if st.button(f"🗑 Remove Entry #{int(selected_id)}", use_container_width=True):
                        st.session_state.confirm_delete_id = selected_id
                        st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        # Volume chart
        if len(df) >= 2:
            st.divider()
            st.markdown('<div class="section-label">Volume per Session (Last 20)</div>', unsafe_allow_html=True)
            cdf = df.tail(20).copy()
            cdf["label"] = cdf["exercise"] + " (" + cdf["date"] + ")"
            fig = px.bar(cdf, x="label", y="volume", color="category",
                color_discrete_map={"Strength":"#39d98a","Cardio":"#a8ff3e","Flexibility":"#8b5cf6","Recovery":"#ffb820"},
                template="plotly_dark")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(2,4,8,0.9)",
                height=300, xaxis_title="", yaxis_title="Volume (kg)",
                xaxis_tickangle=-35, margin=dict(t=10),
                legend=dict(orientation="h", y=1.12),
                font=dict(family="Outfit", size=11, color="#3a4e65")
            )
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD  (Redesigned)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    scroll_to_top()

    stats = get_stats()
    df    = pd.DataFrame(st.session_state.workout_log)
    streak_msg, streak_icon, streak_color = get_streak_message(stats["streak"])

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="fade-in" style="margin-bottom:24px">
        <div style="display:inline-flex;align-items:center;gap:8px;
        font-family:var(--font-mono);font-size:9.5px;letter-spacing:0.12em;
        color:var(--accent2);text-transform:uppercase;padding:4px 12px;
        border:1px solid rgba(139,92,246,0.2);border-radius:100px;
        background:rgba(139,92,246,0.04);margin-bottom:16px">
            <div style="width:5px;height:5px;background:var(--accent2);border-radius:50%"></div>
            📊 Progress Analytics
        </div>
        <h1 style="font-size:2.8rem!important;margin-bottom:10px!important">Dashboard</h1>
        <p style="font-size:15px!important;color:var(--text-dim)!important;max-width:520px;line-height:1.6!important;margin:0!important">
            Your fitness journey at a glance — the more you log, the smarter the insights.
        </p>
    </div>""", unsafe_allow_html=True)


    # ── Metrics row ────────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Sessions",      stats["sessions"])
    m2.metric("Total Volume",  f"{stats['volume']:,.0f} kg" if stats['volume'] >= 1000 else f"{stats['volume']} kg")
    m3.metric("Avg / Session", f"{stats['avg']} kg")
    m4.metric("Best Exercise", stats["top_ex"] if stats["top_ex"] != "—" else "—")
    m5.metric("PRs Tracked",   len(stats.get("prs", {})))

    if df.empty:
        st.markdown("""
        <div style="text-align:center;padding:48px 24px;background:var(--bg-card);
        border:1px dashed var(--border);border-radius:20px;margin-top:20px">
            <div style="font-size:44px;margin-bottom:16px;opacity:0.5">📊</div>
            <div style="font-family:var(--font-display);font-size:16px;font-weight:700;
            color:var(--text-primary);margin-bottom:8px">No workout data yet</div>
            <div style="font-size:13px;color:var(--text-muted)">Head to Workout Log and start tracking your sessions.</div>
        </div>""", unsafe_allow_html=True)
        if st.button("📋 Go to Workout Log →", type="primary"):
            st.session_state.page = "📋 Workout Log";
        st.stop()
        
    # ── Charts: Volume over time + Training split ──────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-label">Volume Over Time</div>', unsafe_allow_html=True)
        vdf = df.copy()
        vdf["date"]   = pd.to_datetime(vdf["date"])
        vdf["volume"] = pd.to_numeric(vdf["volume"], errors="coerce").fillna(0)
        daily = vdf.groupby("date")["volume"].sum().reset_index()
        fig1  = px.area(daily, x="date", y="volume", template="plotly_dark",
                        color_discrete_sequence=["#39d98a"])
        fig1.update_traces(fillcolor="rgba(57,217,138,0.06)", line_width=2)
        fig1.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(2,4,8,0.9)",
            height=250, margin=dict(t=10,b=0),
            xaxis_title="", yaxis_title="Volume (kg)",
            font=dict(family="Outfit", size=11, color="#3a4e65"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-label">Training Split</div>', unsafe_allow_html=True)
        cat_df = pd.DataFrame(list(stats["cats"].items()), columns=["Category","Count"])
        fig2   = px.pie(cat_df, names="Category", values="Count", template="plotly_dark",
                        color_discrete_map={"Strength":"#39d98a","Cardio":"#a8ff3e","Flexibility":"#8b5cf6","Recovery":"#ffb820"},
                        hole=0.5)
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", height=250, margin=dict(t=10,b=0),
            font=dict(family="Outfit", size=11, color="#3a4e65"),
            legend=dict(orientation="h", y=-0.12)
        )
        fig2.update_traces(textfont_size=11)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Top exercises bar chart ────────────────────────────────────────────────
    st.markdown('<div class="section-label">Top Exercises by Total Volume</div>', unsafe_allow_html=True)
    exdf = df.groupby("exercise")["volume"].sum().sort_values(ascending=False).head(8).reset_index()
    exdf.columns = ["Exercise","Total Volume (kg)"]
    fig3 = px.bar(exdf, x="Total Volume (kg)", y="Exercise", orientation="h",
                  template="plotly_dark", color="Total Volume (kg)",
                  color_continuous_scale=[[0, "#131d2e"], [1, "#39d98a"]])
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(2,4,8,0.9)",
        height=280, margin=dict(t=10,b=0), coloraxis_showscale=False,
        font=dict(family="Outfit", size=11, color="#3a4e65"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
    )
    fig3.update_traces(marker_line_width=0)
    st.plotly_chart(fig3, use_container_width=True)

    # ── Monthly heatmap ────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">🔥 Monthly Workout Calendar</div>', unsafe_allow_html=True)
    heat_df = df.copy()
    if not heat_df.empty and "date" in heat_df.columns:
        import calendar
        heat_df["date"] = pd.to_datetime(heat_df["date"])
        today_dt = datetime.now()
        year, month = today_dt.year, today_dt.month
        month_df = heat_df[(heat_df["date"].dt.year == year) & (heat_df["date"].dt.month == month)].copy()
        daily_volume = month_df.groupby(month_df["date"].dt.day)["volume"].sum().to_dict()
        cal_grid = calendar.monthcalendar(year, month)
        heat_matrix, text_matrix = [], []
        for week in cal_grid:
            heat_row, text_row = [], []
            for day in week:
                if day == 0:
                    heat_row.append(None); text_row.append("")
                else:
                    volume = daily_volume.get(day, 0)
                    heat_row.append(volume)
                    text_row.append(f"{day}<br>{int(volume)}kg" if volume > 0 else str(day))
            heat_matrix.append(heat_row); text_matrix.append(text_row)
        fig_heat = px.imshow(heat_matrix, aspect="auto",
            color_continuous_scale=[[0.0,"#0c1220"],[0.3,"#0047AB"],[0.6,"#39d98a"],[0.85,"#a8ff3e"],[1.0,"#ffb820"]])
        fig_heat.update_traces(text=text_matrix, texttemplate="%{text}",
                               hovertemplate="Volume: %{z} kg<extra></extra>")
        fig_heat.update_xaxes(tickvals=list(range(7)), ticktext=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
        fig_heat.update_yaxes(showticklabels=False)
        fig_heat.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(2,4,8,0.9)",
            height=320, margin=dict(t=30,b=10),
            coloraxis_colorbar=dict(title="Volume (kg)", tickfont=dict(size=10)),
            title=dict(text=f"{calendar.month_name[month]} {year}", font=dict(family="Syne", size=13, color="#7a8fa8")),
            font=dict(family="Outfit", size=11, color="#3a4e65")
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("Brighter = higher training volume. Hover for exact figures.")

    # ── Adaptive AI Report ─────────────────────────────────────────────────────
    st.markdown('<div class="section-label">AI Adaptive Recommendation</div>', unsafe_allow_html=True)

    if stats["sessions"] < 3:
        pct_unlock = int((stats["sessions"] / 3) * 100)
        st.markdown(f"""
        <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:14px;
        padding:22px 24px;display:flex;align-items:center;gap:20px">
            <div style="font-size:36px;opacity:0.5">🔒</div>
            <div style="flex:1">
                <div style="font-size:15px;font-weight:600;color:var(--text-primary);margin-bottom:4px">
                    Unlock adaptive recommendations
                </div>
                <div style="font-size:14px;color:var(--text-muted);margin-bottom:12px">
                    Log {3 - stats['sessions']} more session{'s' if 3 - stats['sessions'] != 1 else ''} to unlock AI-powered adaptive analysis.
                </div>
                <div style="height:4px;background:rgba(255,255,255,0.06);border-radius:100px;overflow:hidden;max-width:280px">
                    <div style="width:{pct_unlock}%;height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2));border-radius:100px"></div>
                </div>
                <div style="font-size:14px;color:var(--text-muted);margin-top:6px;font-family:var(--font-mono)">{stats['sessions']}/3 sessions</div>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:var(--bg-card);border:1px solid rgba(57,217,138,0.12);
        border-radius:14px;padding:20px 24px;margin-bottom:16px">
            <div style="display:flex;align-items:flex-start;gap:14px">
                <div style="font-size:28px">🤖</div>
                <div style="flex:1">
                    <div style="font-family:var(--font-display);font-size:14px;font-weight:700;
                    color:var(--text-primary);margin-bottom:6px">Smart Plan Update</div>
                    <div style="font-size:14px;color:var(--text-muted);line-height:1.7">
                    The AI will analyse your real workout history and provide data-driven recommendations —
                    grounded in your actual performance and sport science principles.
                    </div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button("🔄  Generate Adaptive Update", type="primary"):
            if not st.session_state.profile:
                st.warning("Set up your profile first.")
            else:
                with st.spinner("🤖 Analysing your workout history..."):
                    rag_ctx = retrieve("progressive overload adaptation progression", rag_index, rag_chunks, k=3)
                    report  = generate_adaptive_update(st.session_state.profile, st.session_state.workout_log, rag_ctx)
                    st.session_state.adaptive_report = report

    if st.session_state.adaptive_report:
        section_meta = {
            "## PROGRESS ANALYSIS":        ("📊", "Progress Analysis"),
            "## ADAPTIVE RECOMMENDATIONS": ("🎯", "Adaptive Recommendations"),
            "## NEXT WEEK TARGETS":        ("📅", "Next Week Targets"),
        }
        found_any = False
        for section, (icon, title) in section_meta.items():
            pat = re.compile(re.escape(section) + r"([\s\S]*?)(?=## |\Z)", re.I)
            m   = pat.search(st.session_state.adaptive_report)
            if m:
                found_any = True
                with st.expander(f"{icon}  {title}", expanded=True):
                    st.write(m.group(1).strip())

        if not found_any:
            st.markdown(f'<div class="plan-block">{st.session_state.adaptive_report}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AI COACH
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💬 AI Coach":
    scroll_to_top()

    profile  = st.session_state.profile
    has_plan = bool(st.session_state.plan_ready)

    # ── Header ─────────────────────────────────────────────────────────────────
    online_dot = '<div style="width:7px;height:7px;background:#a8ff3e;border-radius:50%;animation:rag-pulse 2s infinite"></div>'
    st.markdown(f"""
    <div class="fade-in" style="margin-bottom:24px">
        <div style="display:inline-flex;align-items:center;gap:8px;
        font-family:var(--font-mono);font-size:9.5px;letter-spacing:0.12em;
        color:#a8ff3e;text-transform:uppercase;padding:4px 12px;
        border:1px solid rgba(168,255,62,0.2);border-radius:100px;
        background:rgba(168,255,62,0.04);margin-bottom:16px">
            {online_dot} Coach Online · RAG Active
        </div>
        <h1 style="font-size:2.8rem!important;margin-bottom:10px!important">AI Coach</h1>
        <p style="font-size:15px!important;color:var(--text-dim)!important;max-width:520px;line-height:1.6!important;margin:0!important">
            Context-aware, evidence-based, always available. Every answer grounded in sport science.
        </p>
    </div>""", unsafe_allow_html=True)

    # ── Context status row ─────────────────────────────────────────────────────
    ctx_items = [
        ("👤", "Profile",  "✅ Set"     if profile  else "❌ Not set",  bool(profile)),
        ("⚡", "Plan",     "✅ Ready"   if has_plan else "⚠️ Not yet",  has_plan),
        ("📋", "Workouts", f"{len(st.session_state.workout_log)} sessions", len(st.session_state.workout_log) > 0),
        ("🤖", "Coach",    "✅ Online",  True),
    ]
    ctx_html = ""
    for icon, label, val, ok in ctx_items:
        bg    = "rgba(168,255,62,0.03)" if ok else "rgba(255,92,92,0.03)"
        bd    = "rgba(168,255,62,0.15)" if ok else "rgba(255,92,92,0.12)"
        vcol  = "var(--accent3)"        if ok else "var(--danger)"
        ctx_html += f"""
        <div style="background:{bg};border:1px solid {bd};border-radius:10px;
        padding:12px 16px;text-align:center">
            <div style="font-size:20px;margin-bottom:6px">{icon}</div>
            <div style="font-size:11px;font-family:var(--font-mono);color:var(--text-muted);
            letter-spacing:0.06em;text-transform:uppercase;margin-bottom:3px">{label}</div>
            <div style="font-size:12px;font-weight:600;color:{vcol}">{val}</div>
        </div>"""
    st.markdown(f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:24px">{ctx_html}</div>',
                unsafe_allow_html=True)

    # ── Suggested questions ────────────────────────────────────────────────────
    with st.expander("💡 Tap a question to ask instantly", expanded=False):
        q_groups = {
            "📈 Progression": ["How do I know when to increase my weights?","I hit a plateau — what should I change?","What is progressive overload and how do I apply it?"],
            "🥗 Nutrition":   ["What should I eat before and after training?","How many calories should I eat on rest days?","How much protein do I need per day?"],
            "😴 Recovery":    ["How many rest days do I need per week?","Should I do cardio before or after weights?","Why do my muscles feel sore 2 days after training?"],
            "🩹 Modifications":["Can you modify exercises for bad knees?","What exercises are safe with lower back pain?","How do I train with a sore shoulder?"],
        }
        for group_label, questions in q_groups.items():
            st.markdown(f"**{group_label}**")
            q_cols = st.columns(3)
            for i, q in enumerate(questions):
                if q_cols[i % 3].button(q, key=f"sq_{group_label}_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role":"user","content":q})
                    st.rerun()

    st.divider()

    # ── Chat window ────────────────────────────────────────────────────────────
    if not st.session_state.chat_history:
        now = datetime.now().strftime("%H:%M")
        st.markdown(f"""
        <div style="background:var(--bg-card);border:1px solid var(--border);
        border-radius:20px 20px 20px 4px;padding:20px 24px;max-width:85%;margin-bottom:8px" class="fade-in">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">
                <div style="width:32px;height:32px;background:linear-gradient(135deg,var(--accent),var(--accent2));
                border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:16px">🤖</div>
                <div>
                    <div style="font-family:var(--font-display);font-size:14px;font-weight:700;color:var(--text-primary)">FitCoach AI</div>
                    <div style="font-size:10px;color:var(--text-muted);font-family:var(--font-mono)">{now} · RAG Active</div>
                </div>
            </div>
            <div style="font-size:14px;color:var(--text-dim);line-height:1.8">
                Hey! I'm your AI-powered personal trainer. Every answer I give is grounded in
                <strong style="color:var(--accent)">verified sport science</strong> — not generic internet advice.<br><br>
                I have full context about your profile, plan, and workout history. Ask me anything about:
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:14px">
                <div style="background:rgba(57,217,138,0.04);border:1px solid rgba(0,229,255,0.1);
                border-radius:8px;padding:8px 12px;font-size:12px;color:var(--text-dim)">
                    🏋️ Exercise form & programming
                </div>
                <div style="background:rgba(57,217,138,0.04);border:1px solid rgba(0,229,255,0.1);
                border-radius:8px;padding:8px 12px;font-size:12px;color:var(--text-dim)">
                    🥗 Nutrition, macros & meal timing
                </div>
                <div style="background:rgba(57,217,138,0.04);border:1px solid rgba(0,229,255,0.1);
                border-radius:8px;padding:8px 12px;font-size:12px;color:var(--text-dim)">
                    📈 Breaking plateaus & overload
                </div>
                <div style="background:rgba(57,217,138,0.04);border:1px solid rgba(0,229,255,0.1);
                border-radius:8px;padding:8px 12px;font-size:12px;color:var(--text-dim)">
                    🩹 Injury-safe modifications
                </div>
            </div>
            <div style="margin-top:12px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.04);
            font-size:10px;color:rgba(57,217,138,0.4);font-family:var(--font-mono)">
                Evidence-grounded · LLM + RAG · Sport science based
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            now_t = datetime.now().strftime("%H:%M")
            if msg["role"] == "user":
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#0a1e38,#0d1a30);
                border:1px solid rgba(57,217,138,0.12);border-radius:20px 20px 4px 20px;
                padding:16px 20px;font-size:13.5px;line-height:1.8;
                margin-left:auto;max-width:82%;color:#d0dde8;margin-bottom:8px" class="fade-in">
                    {msg["content"]}
                    <div style="font-size:10px;color:var(--text-muted);margin-top:8px;
                    display:flex;align-items:center;gap:6px;justify-content:flex-end">
                        <span>{now_t}</span><span>·</span><span>You</span>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:var(--bg-card);border:1px solid var(--border);
                border-radius:20px 20px 20px 4px;padding:16px 20px;font-size:13.5px;
                line-height:1.8;max-width:82%;color:var(--text-primary);margin-bottom:8px" class="fade-in">
                    {msg["content"]}
                    <div style="font-size:10px;color:var(--text-muted);margin-top:8px;
                    display:flex;align-items:center;gap:6px">
                        <span>{now_t}</span><span>·</span>
                        <span style="color:var(--accent)">FitCoach AI</span>
                    </div>
                    <div style="font-size:10px;color:rgba(57,217,138,0.35);margin-top:5px;
                    padding-top:6px;border-top:1px solid rgba(255,255,255,0.04);
                    font-family:var(--font-mono)">
                        Evidence-grounded · LLM + RAG
                    </div>
                </div>""", unsafe_allow_html=True)

    # Typing indicator + LLM call
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        st.markdown("""
        <div style="background:var(--bg-card);border:1px solid var(--border);
        border-radius:20px 20px 20px 4px;padding:14px 20px;display:inline-block;margin-bottom:8px">
            <div class="typing-indicator">
                <div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>
            </div>
        </div>""", unsafe_allow_html=True)
        last_q  = st.session_state.chat_history[-1]["content"]
        rag_ctx = retrieve(last_q, rag_index, rag_chunks, k=3)
        reply   = chat_response(
            history=st.session_state.chat_history,
            profile=profile or {},
            plan=st.session_state.plan or "",
            log=st.session_state.workout_log,
            rag_context=rag_ctx
        )
        st.session_state.chat_history.append({"role":"assistant","content":reply})
        st.rerun()

    # ── Input bar ──────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    <div style="font-size:11px;color:var(--text-muted);margin-bottom:8px;
    font-family:var(--font-mono);letter-spacing:0.04em">
        ASK YOUR COACH — answers grounded in sport science
    </div>""", unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        col_in, col_btn = st.columns([5,1])
        user_msg = col_in.text_input(
            "Ask your coach...", label_visibility="collapsed",
            placeholder="e.g. Should I increase my squat this week? How much protein do I need?")
        send = col_btn.form_submit_button("Send →", type="primary", use_container_width=True)

    if send and user_msg.strip():
        st.session_state.chat_history.append({"role":"user","content":user_msg.strip()})
        st.rerun()

    # Clear chat
    if st.session_state.chat_history:
        col_clr, _ = st.columns([1,5])
        with col_clr:
            if "confirm_clear_chat" not in st.session_state:
                st.session_state.confirm_clear_chat = False
            if st.session_state.confirm_clear_chat:
                c1, c2 = st.columns(2)
                if c1.button("✅ Yes, clear", use_container_width=True, type="primary"):
                    st.session_state.chat_history = []
                    st.session_state.confirm_clear_chat = False
                    st.rerun()
                if c2.button("❌ Cancel", use_container_width=True):
                    st.session_state.confirm_clear_chat = False
                    st.rerun()
            else:
                if st.button("🗑 Clear Chat", use_container_width=True):
                    st.session_state.confirm_clear_chat = True
                    st.rerun()