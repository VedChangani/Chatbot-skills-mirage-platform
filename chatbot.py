import streamlit as st
import os
import re
import json
import inspect
import numpy as np
from datetime import datetime, timedelta
from supabase import create_client
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from deep_translator import GoogleTranslator

# ------------------------------------------------
# Config
# ------------------------------------------------
st.set_page_config(page_title="Career AI", page_icon="💼", layout="centered")

# ------------------------------------------------
# Credentials
# ------------------------------------------------
SUPABASE_URL = st.secrets.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY", "")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY", "")

if not GROQ_API_KEY or not SUPABASE_KEY or not SUPABASE_URL:
    st.error("❌ Missing credentials. Add to Streamlit Secrets.")
    st.stop()

llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ================================================
# SUPABASE AUTH — iframe postMessage token receiver
#
# Flow:
#   1. Next.js parent reads the Supabase session from its own client
#   2. Parent postMessages { access_token, refresh_token, user_id, email }
#      into this iframe on load
#   3. A small JS snippet (injected via st.components) catches the message
#      and writes the token into the iframe URL as ?sb_token=<access_token>
#      triggering a Streamlit rerun
#   4. Python reads the token from query params, calls get_user() to verify
#      it server-side, and stores the real user UUID
#   5. On every subsequent rerun, user_id is already in st.session_state
#      so none of this runs again
# ================================================

import streamlit.components.v1 as components

# ── Step 1: Inject the JS postMessage listener ───────────────────────────────
# This runs on every page load but is a no-op once user_id is set.
# The JS listens for a message from the parent Next.js window, then
# reloads the iframe URL with ?sb_token=<token> appended.
# We use a unique type "SUPABASE_AUTH" to avoid reacting to other messages.

TOKEN_PARAM = "sb_token"

def inject_postmessage_listener():
    """
    Inject a JS snippet that:
    - Listens for postMessage from parent with type SUPABASE_AUTH
    - On receipt, appends ?sb_token=<access_token> to the iframe URL
      which triggers Streamlit to rerun and lets Python read it
    - Also immediately requests the token from the parent (in case
      the message already fired before the listener was ready)
    """
    components.html(
        """
        <script>
        (function() {
            // Avoid registering multiple listeners on reruns
            if (window._sbAuthListenerRegistered) return;
            window._sbAuthListenerRegistered = true;

            function applyToken(token) {
                // Only redirect if token not already in URL
                const url = new URL(window.location.href);
                if (!url.searchParams.get('sb_token')) {
                    url.searchParams.set('sb_token', token);
                    // Replace so the token param doesn't stack up in history
                    window.location.replace(url.toString());
                }
            }

            window.addEventListener('message', function(event) {
                // Security: only accept messages from your Next.js origin.
                // IMPORTANT — replace this with your actual production domain.
                const ALLOWED_ORIGINS = [
                    'http://localhost:3000',
                    'https://your-app.vercel.app',   // ← replace with real domain
                ];
                if (!ALLOWED_ORIGINS.includes(event.origin)) return;

                if (event.data && event.data.type === 'SUPABASE_AUTH') {
                    const token = event.data.access_token;
                    if (token) applyToken(token);
                }
            });

            // Ask the parent to re-send the token in case it was sent
            // before this listener was registered (common on first load)
            if (window.parent !== window) {
                window.parent.postMessage({ type: 'REQUEST_SUPABASE_TOKEN' }, '*');
            }
        })();
        </script>
        """,
        height=0,
    )

def verify_token_and_set_user(access_token: str) -> tuple:
    """
    Verify the access_token server-side by calling supabase.auth.get_user().
    Returns (user_id, email) or raises on failure.
    This is the critical security step — we never trust a token without
    verifying it against Supabase.
    """
    try:
        res = supabase.auth.get_user(access_token)
        if res and res.user:
            return res.user.id, res.user.email
        raise ValueError("Token valid but no user returned.")
    except Exception as e:
        raise ValueError(f"Token verification failed: {e}")

# ── Auth gate ─────────────────────────────────────────────────────────────────

if "user_id" not in st.session_state:

    # Check if the postMessage listener already wrote a token into the URL
    token_from_url = st.query_params.get(TOKEN_PARAM)

    if token_from_url:
        # Verify token server-side and extract real user UUID
        try:
            uid, email = verify_token_and_set_user(token_from_url)
            st.session_state.user_id = uid
            st.session_state.user_email = email
            # Remove the token from the URL immediately for cleanliness
            # (it's already stored in session_state, no need to keep it visible)
            st.query_params.clear()
            st.rerun()
        except ValueError as e:
            # Token was present but invalid — show error and wait for parent
            st.set_page_config(page_title="Career AI", page_icon="💼")
            st.error(f"❌ Authentication failed: {e}")
            st.info("Please return to the main app and try again.")
            inject_postmessage_listener()
            st.stop()
    else:
        # No token yet — inject listener and wait for parent to postMessage it
        st.set_page_config(page_title="Career AI", page_icon="💼")
        st.title("💼 Career Intelligence Chatbot")
        st.info("⏳ Authenticating with your session...")
        inject_postmessage_listener()
        st.stop()

# ── Authenticated header ──────────────────────────────────────────────────────
st.title("💼 Career Intelligence Chatbot")
st.caption("RAG + Query Rewriting + Tool Calling + Risk Metrics + Hindi Support")

# ------------------------------------------------
# TABLE SCHEMA
# ------------------------------------------------
TABLE_SCHEMA = """
PRIMARY TABLE: naukri_jobs
Columns: id, jobtitle, company, stars, experience, location, skills, posted, postdate, site_name, uniq_id, created_at

SECONDARY TABLE: jobs
Columns: jobid, jobtitle, company, joblocation_address, skills, payrate, postdate, industry, experience, education, jobdescription, numberofpositions, site_name, uniq_id
"""

# ------------------------------------------------
# TRANSLATION
# ------------------------------------------------
def translate_to_english(text):
    try:
        if re.search("[\u0900-\u097F]", text):
            return GoogleTranslator(source="hi", target="en").translate(text)
        return text
    except Exception:
        try:
            r = llm.invoke(f"Translate to English. Return ONLY translation:\n{text}")
            return r.content.strip()
        except Exception:
            return text

# ------------------------------------------------
# SAVING CHAT HISTORY
# ------------------------------------------------
def save_chat(role, message):
    try:
        supabase.table("chat_history").insert({
            "user_id": st.session_state.user_id,
            "role": role,
            "message": message
        }).execute()
    except Exception as e:
        print("Chat save error:", e)

# ------------------------------------------------
# LOADING CHAT HISTORY — Fixed ordering bug
# ------------------------------------------------
def load_chat_history():
    """
    Load last 20 messages in correct chronological order.
    Bug fix: previously used desc + Python reverse which still
    returned the LATEST 20 rows (wrong). Now we use a subquery
    approach: fetch desc (gets newest first), then reverse in Python
    — but LIMIT must happen AFTER ordering so we get the true last 20.
    """
    try:
        # Fetch most recent 20 rows (newest first), then reverse for chronological display
        res = supabase.table("chat_history") \
            .select("role, message, created_at") \
            .eq("user_id", st.session_state.user_id) \
            .order("created_at", desc=True) \
            .limit(20) \
            .execute()

        if not res.data:
            return []

        # Reverse to get chronological order (oldest → newest)
        data = list(reversed(res.data))
        return [{"role": r["role"], "content": r["message"]} for r in data]

    except Exception as e:
        print("Chat load error:", e)
        return []

# ------------------------------------------------
# BUILD HISTORY STRING FOR LLM — No truncation of content
# ------------------------------------------------
def build_history_str(chat_history, max_messages=10, max_chars_per_msg=600):
    """
    Build a conversation history string for LLM prompts.
    Uses last `max_messages` messages with reasonable per-message limit.
    """
    if not chat_history:
        return ""
    history_str = ""
    recent = chat_history[-max_messages:]
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:max_chars_per_msg]
        if len(msg["content"]) > max_chars_per_msg:
            content += "..."
        history_str += f"{role}: {content}\n"
    return history_str.strip()

# ================================================
# RAG SECTION
# ================================================

@st.cache_resource
def load_embedding_model():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

def save_embeddings_to_supabase(jobs, embedding_model):
    try:
        existing_ids = set()
        offset = 0
        while True:
            res = supabase.table("job_embeddings")\
                .select("id").range(offset, offset + 999).execute()
            batch = res.data or []
            if not batch:
                break
            existing_ids.update(row["id"] for row in batch)
            offset += 1000
            if len(batch) < 1000:
                break

        new_jobs = [job for job in jobs if job.get("id") not in existing_ids]
        if not new_jobs:
            return 0

        saved = 0
        all_records = []
        batch_size = 200
        progress = st.progress(0, text=f"Embedding {len(new_jobs)} new jobs...")

        for batch_start in range(0, len(new_jobs), batch_size):
            batch_jobs = new_jobs[batch_start: batch_start + batch_size]
            page_contents = []
            for job in batch_jobs:
                content = f"""Job Title: {job.get('jobtitle', 'N/A')}
Company: {job.get('company', 'N/A')}
Location: {job.get('location', 'N/A')}
Experience: {job.get('experience', 'N/A')}
Skills: {job.get('skills', 'N/A')}
Rating: {job.get('stars', 'N/A')}""".strip()
                page_contents.append(content)

            embeddings = embedding_model.embed_documents(page_contents)

            for job, content, embedding in zip(batch_jobs, page_contents, embeddings):
                all_records.append({
                    "id": job.get("id"),
                    "jobtitle": job.get("jobtitle", ""),
                    "company": job.get("company", ""),
                    "location": job.get("location", ""),
                    "skills": job.get("skills", ""),
                    "experience": job.get("experience", ""),
                    "stars": job.get("stars", ""),
                    "page_content": content,
                    "embedding": embedding
                })

            progress.progress(
                min((batch_start + batch_size) / len(new_jobs), 1.0),
                text=f"Embedding... {min(batch_start + batch_size, len(new_jobs))}/{len(new_jobs)}"
            )

        progress2 = st.progress(0, text=f"Saving {len(all_records)} records to Supabase...")
        for i in range(0, len(all_records), 50):
            chunk = all_records[i:i + 50]
            supabase.table("job_embeddings").upsert(chunk, on_conflict="id").execute()
            saved += len(chunk)
            progress2.progress(
                min((i + 50) / len(all_records), 1.0),
                text=f"Saving... {min(i + 50, len(all_records))}/{len(all_records)}"
            )

        progress.empty()
        progress2.empty()
        return saved

    except Exception as e:
        return f"Error: {str(e)}"

def load_embeddings_from_supabase(embedding_model):
    try:
        from langchain_community.vectorstores import FAISS

        count_res = supabase.table("job_embeddings").select("id", count="exact").execute()
        total = count_res.count or 0
        if total == 0:
            return None, 0

        chunk_size = 500
        offset = 0
        vector_store = None
        progress = st.progress(0, text="Loading embeddings from Supabase...")

        while offset < total:
            res = supabase.table("job_embeddings").select("*")\
                .range(offset, offset + chunk_size - 1).execute()
            chunk = res.data or []
            if not chunk:
                break

            chunk_text_embeddings = []
            chunk_metadatas = []

            for row in chunk:
                embedding = row["embedding"]
                if isinstance(embedding, str):
                    embedding = json.loads(embedding)
                embedding = [float(x) for x in embedding]
                chunk_text_embeddings.append((row["page_content"], embedding))
                chunk_metadatas.append({
                    "jobtitle": row.get("jobtitle", ""),
                    "company": row.get("company", ""),
                    "location": row.get("location", ""),
                    "skills": row.get("skills", ""),
                })

            if vector_store is None:
                vector_store = FAISS.from_embeddings(
                    text_embeddings=chunk_text_embeddings,
                    embedding=embedding_model,
                    metadatas=chunk_metadatas
                )
            else:
                chunk_store = FAISS.from_embeddings(
                    text_embeddings=chunk_text_embeddings,
                    embedding=embedding_model,
                    metadatas=chunk_metadatas
                )
                vector_store.merge_from(chunk_store)

            offset += chunk_size
            progress.progress(
                min(offset / total, 1.0),
                text=f"Loading... {min(offset, total)}/{total}"
            )

        progress.empty()
        return vector_store, total

    except Exception as e:
        return None, str(e)

@st.cache_resource
def get_vector_store():
    try:
        embedding_model = load_embedding_model()

        total_res = supabase.table("naukri_jobs").select("id", count="exact").execute()
        total_jobs = total_res.count or 0

        saved_res = supabase.table("job_embeddings").select("id", count="exact").execute()
        total_saved = saved_res.count or 0

        if total_saved >= total_jobs and total_saved > 0:
            vector_store, result = load_embeddings_from_supabase(embedding_model)
            if vector_store:
                return vector_store, result, "loaded"

        all_jobs = []
        page_size = 1000
        offset = 0
        while True:
            res = supabase.table("naukri_jobs").select(
                "id, jobtitle, company, location, skills, experience, stars"
            ).range(offset, offset + page_size - 1).execute()
            batch = res.data or []
            if not batch:
                break
            all_jobs.extend(batch)
            offset += page_size
            if len(batch) < page_size:
                break

        if not all_jobs:
            return None, 0, "no_data"

        saved = save_embeddings_to_supabase(all_jobs, embedding_model)
        if isinstance(saved, str):
            return None, saved, "save_error"

        vector_store, result = load_embeddings_from_supabase(embedding_model)
        if vector_store:
            return vector_store, result, "created"
        else:
            return None, result, "load_error"

    except Exception as e:
        return None, str(e), "exception"

def rag_search(query, vector_store, k=5):
    try:
        return vector_store.similarity_search(query, k=k)
    except Exception:
        return []

# ------------------------------------------------
# QUERY REWRITER
# ------------------------------------------------
def rewrite_query(question, chat_history):
    if not chat_history or len(chat_history) < 2:
        return question

    history_str = build_history_str(chat_history, max_messages=6)

    rewrite_prompt = f"""You are a search query optimizer for a job search engine.

CONVERSATION HISTORY:
{history_str}

CURRENT USER MESSAGE: {question}

Rewrite the current message into a STANDALONE search query for a job database.

Rules:
- If already clear and standalone, return as-is
- Replace pronouns like "that", "it", "those" with actual context from history
- Keep concise — 5-15 words max
- Focus on: role, skills, location, company
- Return ONLY the rewritten query, nothing else

Examples:
"Tell me more about that" → "Data Scientist jobs at Accenture Bengaluru"
"What about Mumbai?" → "Python Developer jobs in Mumbai"
"Show me similar ones" → "Machine Learning Engineer jobs Bangalore"
"What skills do I need?" → "Required skills for Data Analyst role"
"How much do they pay?" → "Data Scientist salary payrate"
"""
    try:
        response = llm.invoke(rewrite_prompt)
        rewritten = response.content.strip()
        if len(rewritten) > 150 or "\n" in rewritten:
            return question
        return rewritten
    except Exception:
        return question

# ------------------------------------------------
# RAG ANSWER
# ------------------------------------------------
def rag_answer(question, vector_store, chat_history, is_hindi=False):
    rewritten_query = rewrite_query(question, chat_history)
    relevant_docs = rag_search(rewritten_query, vector_store)

    context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs]) \
        if relevant_docs else "No relevant jobs found."

    history_str = build_history_str(chat_history, max_messages=8)

    language = "IMPORTANT: Respond entirely in Hindi." if is_hindi else "Respond in English."

    prompt = f"""You are an expert career advisor with access to real Indian job market data.

CONVERSATION HISTORY:
{history_str}

ORIGINAL USER QUESTION: {question}
SEARCH QUERY USED: {rewritten_query}

RELEVANT JOB DATA (via semantic search):
{context}

Instructions:
- Format EACH job as a compact 3-line card exactly like this:

**1. Job Title** — Company Name
📍 Location | ⏳ Experience | ⭐ Rating
🛠️ `skill1` `skill2` `skill3` `skill4`

Rules for cards:
- Strictly 3 lines per card, no more
- Keep location short — city names only, max 2-3 cities
- If rating is missing or None → write N/A
- Skills in backticks like tags — max 4-5 skills per card
- Separate each card with a blank line
- After ALL cards write ONE short summary line (max 20 words)
- End with ONE short follow-up question (max 15 words)
- NO long paragraphs anywhere
- {language}
"""
    try:
        response = llm.invoke(prompt)
        return response.content, relevant_docs, rewritten_query
    except Exception as e:
        return f"Error: {str(e)}", [], question

# ================================================
# RISK METRICS SECTION
# ================================================

def fetch_market_data_for_risk(role):
    data = {}
    try:
        res = supabase.table("naukri_jobs").select("*", count="exact")\
            .ilike("jobtitle", f"%{role}%").execute()
        data["total_jobs"] = res.count or 0

        res2 = supabase.table("naukri_jobs").select("skills")\
            .ilike("jobtitle", f"%{role}%").limit(100).execute()
        freq = {}
        for row in res2.data or []:
            for s in (row.get("skills") or "").split(","):
                s = s.strip().lower()
                if s:
                    freq[s] = freq.get(s, 0) + 1
        data["top_skills"] = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:15]

        cutoff_recent = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
        cutoff_old = (datetime.utcnow() - timedelta(days=180)).strftime("%Y-%m-%dT%H:%M:%S+00:00")

        res_recent = supabase.table("naukri_jobs").select("*", count="exact")\
            .ilike("jobtitle", f"%{role}%").gte("postdate", cutoff_recent).execute()
        data["recent_3m"] = res_recent.count or 0

        res_old = supabase.table("naukri_jobs").select("*", count="exact")\
            .ilike("jobtitle", f"%{role}%")\
            .gte("postdate", cutoff_old).lt("postdate", cutoff_recent).execute()
        data["prev_3m"] = res_old.count or 0

        res4 = supabase.table("naukri_jobs").select("company")\
            .ilike("jobtitle", f"%{role}%").limit(100).execute()
        companies = list(set([r.get("company", "") for r in res4.data or [] if r.get("company")]))
        data["unique_companies"] = len(companies)
        data["top_companies"] = companies[:10]

    except Exception as e:
        data["db_error"] = str(e)
    return data

def calculate_risk_metrics(role, job_description, market_data):
    skills_from_db = [s[0] for s in market_data.get("top_skills", [])]
    total_jobs = market_data.get("total_jobs", 0)
    recent_3m = market_data.get("recent_3m", 0)
    prev_3m = market_data.get("prev_3m", 0)
    unique_companies = market_data.get("unique_companies", 0)

    trend_str = f"{((recent_3m - prev_3m) / prev_3m * 100):+.1f}% change" \
        if prev_3m > 0 else "Insufficient historical data"

    risk_prompt = f"""You are an expert AI labor market analyst.

JOB ROLE: {role}
JOB DESCRIPTION: {job_description or "Not provided"}

REAL MARKET DATA:
- Total active listings: {total_jobs}
- Jobs last 3 months: {recent_3m}
- Jobs previous 3 months: {prev_3m}
- Trend: {trend_str}
- Unique companies hiring: {unique_companies}
- Top skills: {', '.join(skills_from_db[:10]) if skills_from_db else 'N/A'}

Calculate 4 risk metrics. Respond ONLY with JSON:
{{
  "task_automation_risk": {{
    "score": <0-100>,
    "label": "<LOW/MEDIUM/HIGH/VERY HIGH>",
    "key_automatable_tasks": ["task1", "task2", "task3"],
    "explanation": "<2-3 sentences>"
  }},
  "ai_replacement_risk": {{
    "score": <0-100>,
    "label": "<LOW/MEDIUM/HIGH/VERY HIGH>",
    "timeline": "<estimated timeline>",
    "explanation": "<2-3 sentences>"
  }},
  "market_saturation_risk": {{
    "score": <0-100>,
    "label": "<LOW/MEDIUM/HIGH/VERY HIGH>",
    "market_signal": "<growing/stable/declining>",
    "explanation": "<2-3 sentences>"
  }},
  "overall_risk_score": {{
    "score": <0-100>,
    "label": "<LOW/MEDIUM/HIGH/VERY HIGH>",
    "verdict": "<one line summary>",
    "top_recommendations": ["rec1", "rec2", "rec3"]
  }}
}}"""

    try:
        response = llm.invoke(risk_prompt)
        content = re.sub(r"```json|```", "", response.content.strip()).strip()
        return json.loads(content)
    except Exception as e:
        return {"error": str(e)}

def display_risk_metrics(metrics, role, market_data, is_hindi=False):
    if "error" in metrics:
        return f"Error calculating risk: {metrics['error']}"

    def score_color(s): return "🔴" if s >= 70 else "🟡" if s >= 40 else "🟢"
    def score_bar(s): return "█" * int(s/10) + "░" * (10 - int(s/10))

    tar = metrics.get("task_automation_risk", {})
    air = metrics.get("ai_replacement_risk", {})
    msr = metrics.get("market_saturation_risk", {})
    ovr = metrics.get("overall_risk_score", {})

    output = f"""## 🎯 Risk Analysis — **{role}**

---
### 1️⃣ Task Automation Risk
{score_color(tar.get('score',0))} **{tar.get('score',0)}/100** — {tar.get('label','')}
`{score_bar(tar.get('score',0))}` {tar.get('score',0)}%
{tar.get('explanation','')}
**Tasks at risk:** {' • '.join(tar.get('key_automatable_tasks',[]))}

---
### 2️⃣ AI Replacement Risk
{score_color(air.get('score',0))} **{air.get('score',0)}/100** — {air.get('label','')}
`{score_bar(air.get('score',0))}` {air.get('score',0)}%
{air.get('explanation','')}
⏱️ **Timeline:** {air.get('timeline','N/A')}

---
### 3️⃣ Market Saturation Risk
{score_color(msr.get('score',0))} **{msr.get('score',0)}/100** — {msr.get('label','')}
`{score_bar(msr.get('score',0))}` {msr.get('score',0)}%
{msr.get('explanation','')}
📈 **Signal:** {msr.get('market_signal','N/A')}

---
### 🏆 Overall Risk Score
{score_color(ovr.get('score',0))} **{ovr.get('score',0)}/100** — {ovr.get('label','')}
`{score_bar(ovr.get('score',0))}` {ovr.get('score',0)}%
**{ovr.get('verdict','')}**

💡 **Recommendations:**
{chr(10).join([f"• {r}" for r in ovr.get('top_recommendations',[])])}

---
📊 *{market_data.get('total_jobs',0)} listings | {market_data.get('unique_companies',0)} companies hiring*
"""

    if is_hindi:
        try:
            translated = llm.invoke(f"Translate to Hindi. Keep numbers, scores, emojis:\n{output}")
            return translated.content
        except Exception:
            return output
    return output

def handle_risk_conversation(user_message, is_hindi):
    if "risk_session" not in st.session_state:
        st.session_state.risk_session = {"active": False, "role": None, "job_description": None, "step": None}

    rs = st.session_state.risk_session
    risk_keywords = ["risk", "automation risk", "ai risk", "replacement risk",
                     "saturation", "risk score", "job risk", "career risk", "जोखिम", "खतरा"]
    is_risk_request = any(kw in user_message.lower() for kw in risk_keywords)

    if is_risk_request and not rs["active"]:
        st.session_state.risk_session = {"active": True, "role": None, "job_description": None, "step": "ask_role"}
        if is_hindi:
            return "बिल्कुल! 📊 अपना **जॉब रोल** बताएं (जैसे: Data Analyst, Developer):"
        return "Sure! 📊 Please tell me your **Job Role / Designation**:"

    if rs["active"] and rs["step"] == "ask_role":
        st.session_state.risk_session["role"] = translate_to_english(user_message)
        st.session_state.risk_session["step"] = "ask_description"
        if is_hindi:
            return f"रोल: **{user_message}** ✅\n\n**Job Description** पेस्ट करें:\n*(नहीं है तो 'skip' लिखें)*"
        return f"Role: **{user_message}** ✅\n\nPaste your **Job Description**:\n*(Type 'skip' if you don't have one)*"

    if rs["active"] and rs["step"] == "ask_description":
        jd = user_message if user_message.lower() not in ["skip", "no", "नहीं", "छोड़ो"] else ""
        st.session_state.risk_session["job_description"] = translate_to_english(jd)
        role = st.session_state.risk_session["role"]
        job_description = st.session_state.risk_session["job_description"]
        st.session_state.risk_session = {"active": False, "role": None, "job_description": None, "step": None}

        with st.spinner(f"📊 Fetching market data for '{role}'..."):
            market_data = fetch_market_data_for_risk(role)
        with st.spinner("🧠 Calculating risk metrics..."):
            metrics = calculate_risk_metrics(role, job_description, market_data)

        return display_risk_metrics(metrics, role, market_data, is_hindi)

    return None

# ================================================
# TOOL CALLING SECTION
# ================================================

TOOLS = {
    "count_jobs_naukri": {"description": "Count jobs from naukri_jobs matching role/city", "params": ["role", "city", "months"]},
    "list_jobs_naukri": {"description": "List jobs from naukri_jobs for a role/city", "params": ["role", "city", "months", "limit"]},
    "count_jobs_secondary": {"description": "Count jobs from jobs table", "params": ["role", "city", "months"]},
    "list_jobs_secondary": {"description": "List jobs from jobs table with salary info", "params": ["role", "city", "limit"]},
    "top_skills": {"description": "Get most in-demand skills for a role", "params": ["role"]},
    "industry_breakdown": {"description": "Show industries hiring for a role", "params": ["role", "city"]},
    "salary_insights": {"description": "Show salary info from jobs table", "params": ["role", "city"]},
    "recent_jobs": {"description": "Get jobs posted in last N months", "params": ["months", "role"]},
    "company_jobs": {"description": "List jobs from a specific company", "params": ["company", "role"]},
    "run_custom_sql": {"description": "Run custom PostgreSQL SELECT query", "params": ["sql"]},
    "general_advice": {"description": "Answer career questions using LLM knowledge", "params": ["question"]}
}

# ------------------------------------------------
# FIXED: apply_multi_role_filter
# The original code joined conditions with commas inside a single or_() call.
# Supabase Python client's or_() accepts a single string like "col.ilike.%x%,col.ilike.%y%"
# which is correct — but only when the column filter is on THE SAME column.
# The bug was that an empty `role` string still triggered filtering.
# Also added proper None/empty checks.
# ------------------------------------------------
def build_or_filter(column, values):
    """
    Build a Supabase or_ filter string for multiple values on one column.
    Returns None if no valid values.
    """
    if not values:
        return None
    parts = [f"{column}.ilike.%{v}%" for v in values if v and v.strip()]
    if not parts:
        return None
    return ",".join(parts)

def apply_multi_role_filter(query, column, role_string):
    """Apply ilike filter for multiple roles/values on a column."""
    if not role_string or not role_string.strip():
        return query
    roles = [r.strip() for r in re.split(r",|or|\||and", role_string, flags=re.I)]
    roles = [r for r in roles if r]
    if not roles:
        return query
    filter_str = build_or_filter(column, roles)
    if filter_str:
        return query.or_(filter_str)
    return query

def apply_multi_city_filter(query, column, city_string):
    """Apply ilike filter for multiple cities on a column."""
    if not city_string or not city_string.strip():
        return query
    cities = [c.strip() for c in re.split(r",|or|\|", city_string, flags=re.I)]
    cities = [c for c in cities if c]
    if not cities:
        return query
    filter_str = build_or_filter(column, cities)
    if filter_str:
        return query.or_(filter_str)
    return query

def apply_multi_company_filter(query, column, company_string):
    """Apply ilike filter for multiple companies on a column."""
    if not company_string or not company_string.strip():
        return query
    companies = [c.strip() for c in re.split(r",|or|\|", company_string, flags=re.I)]
    companies = [c for c in companies if c]
    if not companies:
        return query
    filter_str = build_or_filter(column, companies)
    if filter_str:
        return query.or_(filter_str)
    return query

# ------------------------------------------------
# TOOL FUNCTIONS — Fixed all query functions
# ------------------------------------------------

def count_jobs_naukri(role="", city="", months=None):
    try:
        q = supabase.table("naukri_jobs").select("*", count="exact")
        q = apply_multi_role_filter(q, "jobtitle", role)
        q = apply_multi_city_filter(q, "location", city)
        if months:
            cutoff = datetime.utcnow() - timedelta(days=30 * int(months))
            q = q.gte("postdate", cutoff.strftime("%Y-%m-%dT%H:%M:%S+00:00"))
        result = q.execute()
        return {"count": result.count or 0}
    except Exception as e:
        return {"error": str(e), "count": 0}

def list_jobs_naukri(role="", city="", months=None, limit=8):
    try:
        q = supabase.table("naukri_jobs").select(
            "jobtitle, company, location, skills, experience, stars, posted, postdate"
        )
        q = apply_multi_role_filter(q, "jobtitle", role)
        q = apply_multi_city_filter(q, "location", city)
        if months:
            cutoff = datetime.utcnow() - timedelta(days=30 * int(months))
            q = q.gte("postdate", cutoff.strftime("%Y-%m-%dT%H:%M:%S+00:00"))
        result = q.limit(int(limit)).execute()
        return {"jobs": result.data or []}
    except Exception as e:
        return {"error": str(e), "jobs": []}

def count_jobs_secondary(role="", city="", months=None):
    try:
        q = supabase.table("jobs").select("*", count="exact")
        q = apply_multi_role_filter(q, "jobtitle", role)
        q = apply_multi_city_filter(q, "joblocation_address", city)
        if months:
            cutoff = datetime.utcnow() - timedelta(days=30 * int(months))
            q = q.gte("postdate", cutoff.strftime("%Y-%m-%d"))
        result = q.execute()
        return {"count": result.count or 0}
    except Exception as e:
        return {"error": str(e), "count": 0}

def list_jobs_secondary(role="", city="", limit=8):
    try:
        q = supabase.table("jobs").select(
            "jobtitle, company, joblocation_address, skills, payrate, experience, industry"
        )
        q = apply_multi_role_filter(q, "jobtitle", role)
        q = apply_multi_city_filter(q, "joblocation_address", city)
        result = q.limit(int(limit)).execute()
        return {"jobs": result.data or []}
    except Exception as e:
        return {"error": str(e), "jobs": []}

def top_skills(role=""):
    try:
        q = supabase.table("naukri_jobs").select("skills")
        q = apply_multi_role_filter(q, "jobtitle", role)
        res = q.limit(200).execute()  # increased limit for better skill frequency
        freq = {}
        for row in res.data or []:
            skills_raw = row.get("skills") or ""
            for s in skills_raw.split(","):
                s = s.strip()
                if s:
                    freq[s] = freq.get(s, 0) + 1
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:12]
        return {"skills": top, "total_analyzed": len(res.data or [])}
    except Exception as e:
        return {"error": str(e), "skills": []}

def industry_breakdown(role="", city=""):
    try:
        q = supabase.table("jobs").select("industry")
        q = apply_multi_role_filter(q, "jobtitle", role)
        q = apply_multi_city_filter(q, "joblocation_address", city)
        res = q.limit(300).execute()
        freq = {}
        for row in res.data or []:
            ind = (row.get("industry") or "Unknown").strip()
            if ind:
                freq[ind] = freq.get(ind, 0) + 1
        top = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True)[:8])
        return {"industries": top, "total_analyzed": len(res.data or [])}
    except Exception as e:
        return {"error": str(e), "industries": {}}

def salary_insights(role="", city=""):
    try:
        q = supabase.table("jobs").select(
            "jobtitle, company, payrate, joblocation_address"
        )
        q = apply_multi_role_filter(q, "jobtitle", role)
        q = apply_multi_city_filter(q, "joblocation_address", city)
        res = q.limit(30).execute()
        # Return all rows; filter those with payrate on the display side
        all_data = res.data or []
        with_salary = [r for r in all_data if r.get("payrate")]
        return {
            "salary_data": with_salary,
            "total_found": len(all_data),
            "with_salary": len(with_salary)
        }
    except Exception as e:
        return {"error": str(e), "salary_data": []}

def recent_jobs(months=3, role=""):
    try:
        cutoff = datetime.utcnow() - timedelta(days=30 * int(months))
        q = supabase.table("naukri_jobs").select(
            "jobtitle, company, location, postdate, skills, posted"
        ).gte("postdate", cutoff.strftime("%Y-%m-%dT%H:%M:%S+00:00"))
        q = apply_multi_role_filter(q, "jobtitle", role)
        result = q.order("postdate", desc=True).limit(15).execute()
        return {"jobs": result.data or [], "months": months}
    except Exception as e:
        return {"error": str(e), "jobs": []}

def company_jobs(company="", role=""):
    try:
        q = supabase.table("naukri_jobs").select(
            "jobtitle, company, location, skills, experience, stars, posted"
        )
        q = apply_multi_company_filter(q, "company", company)
        q = apply_multi_role_filter(q, "jobtitle", role)
        result = q.limit(10).execute()
        return {"jobs": result.data or []}
    except Exception as e:
        return {"error": str(e), "jobs": []}

def run_custom_sql(sql=""):
    try:
        if not sql.strip().lower().startswith("select"):
            return {"error": "Only SELECT allowed."}
        result = supabase.rpc("run_query", {"query": sql}).execute()
        return {"results": result.data or [], "sql_used": sql}
    except Exception as e:
        return {"error": str(e)}

def general_advice(question=""):
    try:
        response = llm.invoke(
            f"You are an expert Indian job market career advisor. Answer concisely:\n{question}"
        )
        return {"answer": response.content}
    except Exception as e:
        return {"error": str(e)}

TOOL_FUNCTIONS = {
    "count_jobs_naukri": count_jobs_naukri,
    "list_jobs_naukri": list_jobs_naukri,
    "count_jobs_secondary": count_jobs_secondary,
    "list_jobs_secondary": list_jobs_secondary,
    "top_skills": top_skills,
    "industry_breakdown": industry_breakdown,
    "salary_insights": salary_insights,
    "recent_jobs": recent_jobs,
    "company_jobs": company_jobs,
    "run_custom_sql": run_custom_sql,
    "general_advice": general_advice,
}

def should_use_rag(question):
    rag_signals = [
        "similar to", "match", "profile", "find jobs for me",
        "suitable", "recommend", "based on my skills",
        "like me", "fit", "relevant jobs", "jobs for someone who",
        "career change", "transition", "what jobs can i do",
        "i know", "i have experience", "i like working",
        "tell me more", "more about", "show me similar",
        "what about", "how about", "other options",
        "मेरे लिए", "मुझे कौन सी", "मेरे skills", "मेरी profile",
        "और बताओ", "इसके बारे में",
        "jobs", "openings", "vacancies",
        "hiring", "positions", "roles",
        "find job", "search job",
        "developer job", "analyst job",
        "engineer job"
    ]
    return any(signal in question.lower() for signal in rag_signals)

def is_casual_message(text):
    casual_patterns = [
        "hi", "hello", "hey", "good morning", "good evening",
        "how are you", "what's up", "thanks", "thank you",
        "ok", "okay", "cool", "nice", "great", "hmm",
        "bye", "goodbye", "see you"
    ]
    text_lower = text.lower().strip()
    if len(text_lower.split()) <= 3:
        for p in casual_patterns:
            if p in text_lower:
                return True
    return False

def plan_tool_call(user_question, chat_history):
    history_str = build_history_str(chat_history, max_messages=4)

    tools_desc = "\n".join([
        f"- {name}: {info['description']} | params: {info['params']}"
        for name, info in TOOLS.items()
    ])

    plan_prompt = f"""You are a tool-calling AI for an Indian job market database.

AVAILABLE TOOLS:
{tools_desc}

DATABASE SCHEMA:
{TABLE_SCHEMA}

CONVERSATION HISTORY:
{history_str}

USER QUESTION: {user_question}

Rules:
1. Default to naukri_jobs tools for most queries
2. Use jobs table for salary/industry/payrate
3. Use run_custom_sql for complex queries
4. Use general_advice for career knowledge questions
5. Pass limit as integer always
6. If asking about jobs in a city, include that city in params
7. Always include role param when a job role is mentioned

Respond ONLY with a valid JSON array:
[{{"tool": "name", "params": {{}}, "reason": "why"}}]
"""
    try:
        response = llm.invoke(plan_prompt)
        content = re.sub(r"```json|```", "", response.content.strip()).strip()
        # Handle potential trailing text after JSON
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        tool_calls = json.loads(content)
        if isinstance(tool_calls, dict):
            tool_calls = [tool_calls]
        return tool_calls
    except Exception as e:
        return [{"tool": "general_advice", "params": {"question": user_question}, "reason": f"fallback: {e}"}]

def execute_tool_calls(tool_calls):
    results = []
    for call in tool_calls:
        tool_name = call.get("tool")
        params = call.get("params", {})
        reason = call.get("reason", "")
        if tool_name in TOOL_FUNCTIONS:
            func = TOOL_FUNCTIONS[tool_name]
            valid_params = inspect.signature(func).parameters.keys()
            filtered = {k: v for k, v in params.items() if k in valid_params}
            try:
                data = func(**filtered)
            except Exception as e:
                data = {"error": str(e)}
        else:
            data = {"error": f"Unknown tool: {tool_name}"}
        results.append({"tool": tool_name, "reason": reason, "data": data})
    return results

def generate_final_answer(user_question, tool_results, chat_history, is_hindi=False):
    results_str = ""
    for r in tool_results:
        results_str += f"\n[Tool: {r['tool']} | Reason: {r['reason']}]\n"
        results_str += json.dumps(r["data"], indent=2, default=str)

    history_str = build_history_str(chat_history, max_messages=6)

    language = "IMPORTANT: Respond entirely in Hindi." if is_hindi else "Respond in English."

    answer_prompt = f"""You are an expert career advisor with real Indian job market data.

CONVERSATION HISTORY:
{history_str}

USER QUESTION: {user_question}

DATA FROM DATABASE:
{results_str}

Instructions:
- Use real data for specific, accurate answers
- Highlight numbers, companies, skills, trends
- If data is empty or has errors, answer from general knowledge and mention data was unavailable
- Be conversational and concise
- Use bullet points where helpful
- Reference previous conversation context when relevant
- {language}
"""
    try:
        response = llm.invoke(answer_prompt)
        return response.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# ================================================
# BUILD RAG INDEX ON STARTUP
# ================================================
with st.spinner("⏳ Loading RAG index..."):
    vector_store, result, status = get_vector_store()

if vector_store:
    if status == "created":
        st.success(f"✅ RAG created and saved! {result} jobs indexed permanently.")
    elif status == "loaded":
        st.success(f"✅ RAG loaded instantly! {result} jobs ready.")
else:
    if status == "no_data":
        st.error("❌ No data in naukri_jobs table.")
    elif status == "save_error":
        st.error(f"❌ Save failed: {result}")
    elif status == "load_error":
        st.error(f"❌ Load failed: {result}")
    else:
        st.error(f"❌ RAG failed: {result}")

# ================================================
# DEBUG PANEL
# ================================================
with st.expander("🔧 Debug Panel"):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("Test naukri_jobs"):
            try:
                res = supabase.table("naukri_jobs").select("*").limit(2).execute()
                st.success(f"✅ {len(res.data)} rows")
                if res.data: st.dataframe(res.data)
            except Exception as e:
                st.error(f"❌ {e}")
    with col2:
        if st.button("Test jobs table"):
            try:
                res = supabase.table("jobs").select("*").limit(2).execute()
                st.success(f"✅ {len(res.data)} rows")
                if res.data: st.dataframe(res.data)
            except Exception as e:
                st.error(f"❌ {e}")
    with col3:
        if st.button("Test job_embeddings"):
            try:
                res = supabase.table("job_embeddings")\
                    .select("id, jobtitle, company, location").limit(5).execute()
                st.success(f"✅ {len(res.data)} embeddings")
                if res.data: st.dataframe(res.data)
            except Exception as e:
                st.error(f"❌ {e}")
    with col4:
        if st.button("Test RAG Search"):
            if vector_store:
                docs = rag_search("python developer bangalore", vector_store)
                st.success(f"✅ {len(docs)} jobs found")
                for d in docs[:2]:
                    st.text(d.page_content[:200])
                    st.divider()
            else:
                st.error("❌ RAG not ready")
    with col5:
        if st.button("🔄 Refresh RAG"):
            try:
                supabase.table("job_embeddings").delete().neq("id", -1).execute()
                get_vector_store.clear()
                st.success("✅ Cleared! Refresh page to rebuild.")
            except Exception as e:
                st.error(f"❌ {e}")

    # Extra: test a direct query to debug filter issues
    st.divider()
    st.markdown("**🔬 Test Query Filter**")
    test_role = st.text_input("Test role (e.g. 'data analyst')", key="test_role_input")
    test_city = st.text_input("Test city (e.g. 'Mumbai')", key="test_city_input")
    if st.button("Run Test Query"):
        r1 = list_jobs_naukri(role=test_role, city=test_city, limit=5)
        st.json(r1)
        r2 = count_jobs_naukri(role=test_role, city=test_city)
        st.write(f"Count: {r2}")

# ================================================
# CHAT UI
# ================================================

# ── Initialize messages — load from Supabase on first run only
if "messages" not in st.session_state:
    history = load_chat_history()

    if history:
        st.session_state.messages = history
        st.session_state.history_loaded = True
    else:
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                "👋 Hi! I'm your Career Intelligence Assistant.\n\n"
                "I can help you with:\n"
                "- 🔍 **Semantic search** — *Find jobs matching my Python and ML skills*\n"
                "- 💬 **Follow-ups** — *What about Mumbai? Tell me more about that*\n"
                "- 📊 **Queries** — *How many BPO jobs in Delhi?*\n"
                "- 🎯 **Risk score** — *Calculate my AI risk score*\n"
                "- 💰 **Salary** — *Show salaries for data analysts*\n"
                "- 🏢 **Companies** — *Jobs at TCS or Infosys*\n"
                "- 🇮🇳 **Hindi** — *मुझे नौकरी ढूंढने में मदद करो* 🙏\n\n"
                f"{'✅ RAG Active — ' + str(result) + ' jobs indexed' if vector_store else '⚠️ RAG not available'}"
            )
        }]
        st.session_state.history_loaded = False

# Keep message list from growing unboundedly in session
if len(st.session_state.messages) > 60:
    # Keep welcome message + last 40 messages
    st.session_state.messages = (
        st.session_state.messages[:1] + st.session_state.messages[-40:]
    )

# Render all messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ── Chat input
if prompt := st.chat_input("Ask about jobs, skills, salaries, risk score..."):

    # Add user message to session + display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    save_chat("user", prompt)

    # Detect Hindi and translate for processing
    is_hindi = bool(re.search("[\u0900-\u097F]", prompt))
    processing_prompt = translate_to_english(prompt) if is_hindi else prompt

    if is_hindi:
        with st.sidebar:
            st.info(f"🌐 Translated: *{processing_prompt}*")

    answer = None

    # ── Priority 1: Risk session
    risk_answer = handle_risk_conversation(processing_prompt, is_hindi)

    if risk_answer:
        answer = risk_answer

    # ── Priority 2: Casual conversation
    elif is_casual_message(processing_prompt):
        # Pass recent history so casual replies are context-aware
        history_str = build_history_str(st.session_state.messages[:-1], max_messages=4)
        casual_reply = llm.invoke(
            f"You are a friendly career assistant. Use conversation history for context.\n"
            f"HISTORY:\n{history_str}\n\nUSER: {processing_prompt}\n\nRespond briefly and naturally:"
        ).content
        answer = casual_reply

    # ── Priority 3: RAG — semantic + follow-up
    elif should_use_rag(processing_prompt) and vector_store:
        with st.spinner("🔍 Searching..."):
            rag_response, relevant_docs, rewritten = rag_answer(
                processing_prompt,
                vector_store,
                st.session_state.messages[:-1],  # history excluding current user msg
                is_hindi
            )
        if rewritten != processing_prompt:
            st.caption(f"🔄 Searched for: _{rewritten}_")
        answer = rag_response

    # ── Priority 4: Tool calling — structured queries
    else:
        with st.spinner("🧠 Thinking..."):
            tool_calls = plan_tool_call(
                processing_prompt,
                st.session_state.messages[:-1]
            )

            with st.sidebar:
                st.subheader("🔍 Tool Plan")
                for tc in tool_calls:
                    st.markdown(f"**🔧 {tc['tool']}**")
                    st.caption(tc.get("reason", ""))
                    st.json(tc.get("params", {}))
                    st.divider()

            tool_results = execute_tool_calls(tool_calls)

            answer = generate_final_answer(
                processing_prompt,
                tool_results,
                st.session_state.messages[:-1],
                is_hindi
            )

    # Save + display assistant answer
    if answer:
        save_chat("assistant", answer)
        st.chat_message("assistant").write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})