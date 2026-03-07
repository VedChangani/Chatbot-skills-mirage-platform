import streamlit as st
import re
import json
import inspect
import os
from datetime import datetime, timedelta
from supabase import create_client
from langchain_groq import ChatGroq
from deep_translator import GoogleTranslator

# ------------------------------------------------
# Config
# ------------------------------------------------
st.set_page_config(page_title="Career AI", page_icon="💼", layout="centered")
st.title("💼 Career Intelligence Chatbot")
st.caption("Understands your question → picks the right tool → fetches real data")

# ------------------------------------------------
# Credentials
# ------------------------------------------------
SUPABASE_URL = st.secrets.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY", "")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY", "")

if not GROQ_API_KEY or not SUPABASE_KEY or not SUPABASE_URL:
    st.error("❌ Missing credentials. Add SUPABASE_URL, SUPABASE_KEY, GROQ_API_KEY to Streamlit Secrets.")
    st.stop()

llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ------------------------------------------------
# TABLE SCHEMA
# ------------------------------------------------
TABLE_SCHEMA = """
PRIMARY TABLE: naukri_jobs
Columns:
  - id (bigint): unique ID
  - jobtitle (text): job role/title
  - company (text): company name
  - stars (text): company rating
  - experience (text): required experience
  - location (text): city/location
  - skills (text): required skills (comma separated)
  - posted (text): posted time e.g. "3 days ago"
  - postdate (timestamp with time zone): exact post date
  - site_name (text): source portal
  - uniq_id (text): unique identifier
  - created_at (timestamp with time zone): record created time

SECONDARY TABLE: jobs
Columns:
  - jobid (bigint): unique job ID
  - jobtitle (text): job role/title
  - company (text): company name
  - joblocation_address (text): city/location
  - skills (text): required skills (comma separated)
  - payrate (text): salary or pay info
  - postdate (timestamp): date job was posted
  - industry (text): industry sector
  - experience (text): required experience
  - education (text): required education
  - jobdescription (text): full job description
  - numberofpositions (integer): number of openings
  - site_name (text): job portal source
  - uniq_id (text): unique identifier
"""

# ------------------------------------------------
# TRANSLATION
# ------------------------------------------------
def translate_to_english(text):
    try:
        if re.search("[\u0900-\u097F]", text):
            translated = GoogleTranslator(source="hi", target="en").translate(text)
            return translated
        return text
    except Exception:
        try:
            r = llm.invoke(f"Translate this to English. Return ONLY the translation:\n{text}")
            return r.content.strip()
        except Exception:
            return text

# ------------------------------------------------
# RISK METRICS ENGINE
# ------------------------------------------------

def fetch_market_data_for_risk(role):
    """Fetch real DB data to power risk calculations."""
    data = {}
    try:
        # Total jobs for this role
        res = supabase.table("naukri_jobs").select("*", count="exact").ilike("jobtitle", f"%{role}%").execute()
        data["total_jobs"] = res.count or 0

        # Skills demanded
        res2 = supabase.table("naukri_jobs").select("skills").ilike("jobtitle", f"%{role}%").limit(100).execute()
        freq = {}
        for row in res2.data or []:
            for s in (row.get("skills") or "").split(","):
                s = s.strip().lower()
                if s:
                    freq[s] = freq.get(s, 0) + 1
        data["top_skills"] = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:15]

        # Recent job postings trend (last 3 months vs previous 3 months)
        cutoff_recent = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
        cutoff_old = (datetime.utcnow() - timedelta(days=180)).strftime("%Y-%m-%dT%H:%M:%S+00:00")

        res_recent = supabase.table("naukri_jobs").select("*", count="exact")\
            .ilike("jobtitle", f"%{role}%")\
            .gte("postdate", cutoff_recent).execute()
        data["recent_3m"] = res_recent.count or 0

        res_old = supabase.table("naukri_jobs").select("*", count="exact")\
            .ilike("jobtitle", f"%{role}%")\
            .gte("postdate", cutoff_old)\
            .lt("postdate", cutoff_recent).execute()
        data["prev_3m"] = res_old.count or 0

        # Experience distribution
        res3 = supabase.table("naukri_jobs").select("experience")\
            .ilike("jobtitle", f"%{role}%").limit(100).execute()
        exp_list = [r.get("experience", "") for r in res3.data or []]
        data["experience_data"] = exp_list

        # Company diversity
        res4 = supabase.table("naukri_jobs").select("company")\
            .ilike("jobtitle", f"%{role}%").limit(100).execute()
        companies = list(set([r.get("company", "") for r in res4.data or [] if r.get("company")]))
        data["unique_companies"] = len(companies)
        data["top_companies"] = companies[:10]

    except Exception as e:
        data["db_error"] = str(e)

    return data


def calculate_risk_metrics(role, job_description, market_data):
    """
    Use LLM + real market data to calculate all 4 risk metrics.
    Returns structured risk scores with explanations.
    """

    skills_from_db = [s[0] for s in market_data.get("top_skills", [])]
    total_jobs = market_data.get("total_jobs", 0)
    recent_3m = market_data.get("recent_3m", 0)
    prev_3m = market_data.get("prev_3m", 0)
    unique_companies = market_data.get("unique_companies", 0)

    # Calculate trend
    if prev_3m > 0:
        trend = ((recent_3m - prev_3m) / prev_3m) * 100
        trend_str = f"{trend:+.1f}% change in job postings (last 3 months vs previous 3 months)"
    else:
        trend_str = "Insufficient historical data for trend"

    risk_prompt = f"""You are an expert AI labor market analyst. Calculate the following 4 risk metrics for a job professional.

JOB ROLE: {role}

JOB DESCRIPTION PROVIDED BY USER:
{job_description if job_description else "Not provided - use role name and DB data to infer"}

REAL MARKET DATA FROM DATABASE:
- Total active job listings for this role: {total_jobs}
- Jobs posted last 3 months: {recent_3m}
- Jobs posted previous 3 months: {prev_3m}
- Trend: {trend_str}
- Unique companies hiring: {unique_companies}
- Top skills demanded in market: {', '.join(skills_from_db[:10]) if skills_from_db else 'Not available'}

INSTRUCTIONS:
Use the real market data as PRIMARY source. Use your AI knowledge as SECONDARY source to fill gaps.

Calculate these 4 metrics on a scale of 0-100:

1. TASK AUTOMATION RISK (0-100):
   - How many tasks in this role can be automated by AI/software?
   - Higher score = more tasks automatable
   - Use job description to identify specific automatable tasks
   - Consider: data entry, repetitive analysis, report generation, customer queries, etc.

2. AI REPLACEMENT RISK (0-100):
   - How likely is AI to fully replace this role in 3-5 years?
   - Higher score = higher replacement risk
   - Consider: market trend data, skill demands, AI capability growth
   - Use hiring trend (recent vs previous) as signal

3. MARKET SATURATION RISK (0-100):
   - How saturated/competitive is this job market?
   - Higher score = more saturated/harder to find job
   - Use: total_jobs vs unique_companies ratio, posting trends
   - More companies = less saturated, fewer companies = more saturated

4. OVERALL RISK SCORE (0-100):
   - Weighted average: Task(25%) + AI Replacement(40%) + Market Saturation(35%)
   - This represents total risk of losing job / career instability

Respond ONLY with this exact JSON (no markdown, no explanation):
{{
  "task_automation_risk": {{
    "score": <0-100>,
    "label": "<LOW/MEDIUM/HIGH/VERY HIGH>",
    "key_automatable_tasks": ["task1", "task2", "task3"],
    "explanation": "<2-3 sentences using real data>"
  }},
  "ai_replacement_risk": {{
    "score": <0-100>,
    "label": "<LOW/MEDIUM/HIGH/VERY HIGH>",
    "timeline": "<estimated timeline for significant impact>",
    "explanation": "<2-3 sentences using real data>"
  }},
  "market_saturation_risk": {{
    "score": <0-100>,
    "label": "<LOW/MEDIUM/HIGH/VERY HIGH>",
    "market_signal": "<growing/stable/declining>",
    "explanation": "<2-3 sentences using real data>"
  }},
  "overall_risk_score": {{
    "score": <0-100>,
    "label": "<LOW/MEDIUM/HIGH/VERY HIGH>",
    "verdict": "<one line summary>",
    "top_recommendations": ["rec1", "rec2", "rec3"]
  }}
}}
"""

    try:
        response = llm.invoke(risk_prompt)
        content = re.sub(r"```json|```", "", response.content.strip()).strip()
        return json.loads(content)
    except Exception as e:
        return {"error": str(e)}


def display_risk_metrics(metrics, role, market_data, is_hindi=False):
    """Display risk metrics as a nicely formatted Streamlit UI."""

    if "error" in metrics:
        return f"Error calculating risk: {metrics['error']}"

    def score_color(score):
        if score >= 70:
            return "🔴"
        elif score >= 40:
            return "🟡"
        else:
            return "🟢"

    def score_bar(score):
        filled = int(score / 10)
        empty = 10 - filled
        return "█" * filled + "░" * empty

    tar = metrics.get("task_automation_risk", {})
    air = metrics.get("ai_replacement_risk", {})
    msr = metrics.get("market_saturation_risk", {})
    ovr = metrics.get("overall_risk_score", {})

    output = f"""
## 🎯 Risk Analysis for **{role}**

---

### 1️⃣ Task Automation Risk
{score_color(tar.get('score', 0))} **Score: {tar.get('score', 0)}/100** — {tar.get('label', '')}
`{score_bar(tar.get('score', 0))}` {tar.get('score', 0)}%

{tar.get('explanation', '')}

**Automatable tasks:**
{chr(10).join([f"• {t}" for t in tar.get('key_automatable_tasks', [])])}

---

### 2️⃣ AI Replacement Risk
{score_color(air.get('score', 0))} **Score: {air.get('score', 0)}/100** — {air.get('label', '')}
`{score_bar(air.get('score', 0))}` {air.get('score', 0)}%

{air.get('explanation', '')}
⏱️ **Timeline:** {air.get('timeline', 'N/A')}

---

### 3️⃣ Market Saturation Risk
{score_color(msr.get('score', 0))} **Score: {msr.get('score', 0)}/100** — {msr.get('label', '')}
`{score_bar(msr.get('score', 0))}` {msr.get('score', 0)}%

{msr.get('explanation', '')}
📈 **Market Signal:** {msr.get('market_signal', 'N/A')}

---

### 🏆 Overall Risk Score
{score_color(ovr.get('score', 0))} **{ovr.get('score', 0)}/100** — {ovr.get('label', '')}
`{score_bar(ovr.get('score', 0))}` {ovr.get('score', 0)}%

**{ovr.get('verdict', '')}**

**💡 Recommendations:**
{chr(10).join([f"• {r}" for r in ovr.get('top_recommendations', [])])}

---
📊 *Based on {market_data.get('total_jobs', 0)} real job listings | {market_data.get('unique_companies', 0)} companies hiring*
"""

    if is_hindi:
        try:
            translated = llm.invoke(
                f"Translate this career risk report to Hindi. Keep numbers, scores, and emojis as-is:\n{output}"
            )
            return translated.content
        except Exception:
            return output

    return output


# ------------------------------------------------
# RISK SESSION STATE MANAGEMENT
# ------------------------------------------------

def handle_risk_conversation(user_message, is_hindi):
    """
    Multi-turn conversation handler for risk assessment.
    Asks for job role and description if not provided.
    """

    # Initialize risk session
    if "risk_session" not in st.session_state:
        st.session_state.risk_session = {
            "active": False,
            "role": None,
            "job_description": None,
            "step": None
        }

    rs = st.session_state.risk_session

    # Check if user is asking for risk metrics
    risk_keywords = [
        "risk", "automation risk", "ai risk", "replacement risk",
        "saturation", "risk score", "job risk", "career risk",
        "जोखिम", "खतरा", "नौकरी जोखिम"
    ]
    is_risk_request = any(kw in user_message.lower() for kw in risk_keywords)

    # Start risk flow
    if is_risk_request and not rs["active"]:
        st.session_state.risk_session = {
            "active": True,
            "role": None,
            "job_description": None,
            "step": "ask_role"
        }
        if is_hindi:
            return "बिल्कुल! मैं आपके लिए जोखिम विश्लेषण करूंगा। 📊\n\nकृपया अपना **जॉब रोल/पद** बताएं (जैसे: Data Analyst, BPO Executive, Software Developer):"
        return "Sure! I'll calculate your career risk metrics. 📊\n\nPlease tell me your **Job Role / Designation** (e.g., Data Analyst, BPO Executive, Software Developer):"

    # Step 1: Collect job role
    if rs["active"] and rs["step"] == "ask_role":
        st.session_state.risk_session["role"] = translate_to_english(user_message)
        st.session_state.risk_session["step"] = "ask_description"
        if is_hindi:
            return f"धन्यवाद! आपका रोल है: **{user_message}** ✅\n\nअब कृपया अपनी **Job Description** पेस्ट करें (आपके दैनिक कार्य, जिम्मेदारियां, उपयोग किए जाने वाले टूल्स):\n\n*(अगर आपके पास JD नहीं है तो 'skip' टाइप करें)*"
        return f"Got it! Role: **{user_message}** ✅\n\nNow please paste your **Job Description** (your daily tasks, responsibilities, tools you use):\n\n*(Type 'skip' if you don't have a JD)*"

    # Step 2: Collect job description then calculate
    if rs["active"] and rs["step"] == "ask_description":
        jd = user_message if user_message.lower() not in ["skip", "no", "नहीं", "छोड़ो"] else ""
        st.session_state.risk_session["job_description"] = translate_to_english(jd)
        st.session_state.risk_session["step"] = "calculating"

        role = st.session_state.risk_session["role"]
        job_description = st.session_state.risk_session["job_description"]

        # Fetch real market data
        with st.spinner(f"📊 Fetching market data for '{role}'..."):
            market_data = fetch_market_data_for_risk(role)

        # Calculate risk metrics
        with st.spinner("🧠 Calculating risk metrics..."):
            metrics = calculate_risk_metrics(role, job_description, market_data)

        # Reset session
        st.session_state.risk_session = {
            "active": False,
            "role": None,
            "job_description": None,
            "step": None
        }

        return display_risk_metrics(metrics, role, market_data, is_hindi)

    return None


# ------------------------------------------------
# TOOLS REGISTRY
# ------------------------------------------------
TOOLS = {
    "count_jobs_naukri": {
        "description": "Count jobs from naukri_jobs table matching role and/or city",
        "params": ["role", "city", "months"]
    },
    "list_jobs_naukri": {
        "description": "List job postings from naukri_jobs with details for a role/city",
        "params": ["role", "city", "months", "limit"]
    },
    "count_jobs_secondary": {
        "description": "Count jobs from jobs table matching role and/or city",
        "params": ["role", "city", "months"]
    },
    "list_jobs_secondary": {
        "description": "List job postings from jobs table with salary/industry info",
        "params": ["role", "city", "limit"]
    },
    "top_skills": {
        "description": "Get most in-demand skills for a job role from naukri_jobs",
        "params": ["role"]
    },
    "industry_breakdown": {
        "description": "Show which industries are hiring for a role from jobs table",
        "params": ["role", "city"]
    },
    "salary_insights": {
        "description": "Show pay/salary info for a role from jobs table",
        "params": ["role", "city"]
    },
    "recent_jobs": {
        "description": "Get jobs posted in last N months from naukri_jobs",
        "params": ["months", "role"]
    },
    "company_jobs": {
        "description": "List jobs from a specific company from naukri_jobs",
        "params": ["company", "role"]
    },
    "run_custom_sql": {
        "description": "Run a custom PostgreSQL SELECT query on either table when no other tool fits",
        "params": ["sql"]
    },
    "general_advice": {
        "description": "Answer career/certification/AI risk questions using LLM knowledge only",
        "params": ["question"]
    }
}

# ------------------------------------------------
# TOOL IMPLEMENTATIONS
# ------------------------------------------------

def count_jobs_naukri(role="", city="", months=None):
    try:
        q = supabase.table("naukri_jobs").select("*", count="exact")
        if role:
            q = q.ilike("jobtitle", f"%{role}%")
        if city:
            q = q.ilike("location", f"%{city}%")
        if months:
            cutoff = datetime.utcnow() - timedelta(days=30 * int(months))
            q = q.gte("postdate", cutoff.strftime("%Y-%m-%dT%H:%M:%S+00:00"))
        res = q.execute()
        return {"count": res.count or 0, "table": "naukri_jobs", "role": role, "city": city}
    except Exception as e:
        return {"error": str(e)}


def list_jobs_naukri(role="", city="", months=None, limit=8):
    try:
        q = supabase.table("naukri_jobs").select(
            "jobtitle, company, location, skills, experience, stars, posted, postdate"
        )
        if role:
            q = q.ilike("jobtitle", f"%{role}%")
        if city:
            q = q.ilike("location", f"%{city}%")
        if months:
            cutoff = datetime.utcnow() - timedelta(days=30 * int(months))
            q = q.gte("postdate", cutoff.strftime("%Y-%m-%dT%H:%M:%S+00:00"))
        res = q.limit(int(limit)).execute()
        return {"jobs": res.data or [], "table": "naukri_jobs"}
    except Exception as e:
        return {"error": str(e)}


def count_jobs_secondary(role="", city="", months=None):
    try:
        q = supabase.table("jobs").select("*", count="exact")
        if role:
            q = q.ilike("jobtitle", f"%{role}%")
        if city:
            q = q.ilike("joblocation_address", f"%{city}%")
        if months:
            cutoff = datetime.utcnow() - timedelta(days=30 * int(months))
            q = q.gte("postdate", cutoff.strftime("%Y-%m-%d"))
        res = q.execute()
        return {"count": res.count or 0, "table": "jobs", "role": role, "city": city}
    except Exception as e:
        return {"error": str(e)}


def list_jobs_secondary(role="", city="", limit=8):
    try:
        q = supabase.table("jobs").select(
            "jobtitle, company, joblocation_address, skills, payrate, experience, industry"
        )
        if role:
            q = q.ilike("jobtitle", f"%{role}%")
        if city:
            q = q.ilike("joblocation_address", f"%{city}%")
        res = q.limit(int(limit)).execute()
        return {"jobs": res.data or [], "table": "jobs"}
    except Exception as e:
        return {"error": str(e)}


def top_skills(role=""):
    try:
        q = supabase.table("naukri_jobs").select("skills")
        if role:
            q = q.ilike("jobtitle", f"%{role}%")
        res = q.limit(150).execute()
        freq = {}
        for row in res.data or []:
            for s in (row.get("skills") or "").split(","):
                s = s.strip()
                if s:
                    freq[s] = freq.get(s, 0) + 1
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:12]
        return {"skills": top}
    except Exception as e:
        return {"error": str(e)}


def industry_breakdown(role="", city=""):
    try:
        q = supabase.table("jobs").select("industry")
        if role:
            q = q.ilike("jobtitle", f"%{role}%")
        if city:
            q = q.ilike("joblocation_address", f"%{city}%")
        res = q.limit(200).execute()
        freq = {}
        for row in res.data or []:
            ind = row.get("industry") or "Unknown"
            freq[ind] = freq.get(ind, 0) + 1
        top = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True)[:8])
        return {"industries": top}
    except Exception as e:
        return {"error": str(e)}


def salary_insights(role="", city=""):
    try:
        q = supabase.table("jobs").select("jobtitle, company, payrate, joblocation_address")
        if role:
            q = q.ilike("jobtitle", f"%{role}%")
        if city:
            q = q.ilike("joblocation_address", f"%{city}%")
        res = q.limit(20).execute()
        jobs = [r for r in (res.data or []) if r.get("payrate")]
        return {"salary_data": jobs}
    except Exception as e:
        return {"error": str(e)}


def recent_jobs(months=3, role=""):
    try:
        cutoff = datetime.utcnow() - timedelta(days=30 * int(months))
        q = supabase.table("naukri_jobs").select(
            "jobtitle, company, location, postdate, skills, posted"
        ).gte("postdate", cutoff.strftime("%Y-%m-%dT%H:%M:%S+00:00"))
        if role:
            q = q.ilike("jobtitle", f"%{role}%")
        res = q.limit(15).execute()
        return {"jobs": res.data or [], "months": months}
    except Exception as e:
        return {"error": str(e)}


def company_jobs(company="", role=""):
    try:
        q = supabase.table("naukri_jobs").select(
            "jobtitle, company, location, skills, experience, stars, posted"
        )
        if company:
            q = q.ilike("company", f"%{company}%")
        if role:
            q = q.ilike("jobtitle", f"%{role}%")
        res = q.limit(10).execute()
        return {"jobs": res.data or []}
    except Exception as e:
        return {"error": str(e)}


def run_custom_sql(sql=""):
    try:
        clean_sql = sql.strip().lower()
        if not clean_sql.startswith("select"):
            return {"error": "Only SELECT queries are allowed for safety."}
        res = supabase.rpc("run_query", {"query": sql}).execute()
        return {"results": res.data or [], "sql_used": sql}
    except Exception as e:
        return {"error": str(e), "sql_attempted": sql}


def general_advice(question=""):
    try:
        response = llm.invoke(f"""
You are an expert career advisor specializing in the Indian job market.
Answer this question with specific, actionable advice:
{question}
Be concise, practical, and data-informed.
""")
        return {"answer": response.content}
    except Exception as e:
        return {"error": str(e)}


# ------------------------------------------------
# TOOL FUNCTION MAP
# ------------------------------------------------
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

# ------------------------------------------------
# BRAIN
# ------------------------------------------------

def plan_tool_call(user_question, chat_history):
    history_str = ""
    if chat_history:
        for msg in chat_history[-4:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content'][:200]}\n"

    tools_desc = "\n".join([
        f"- {name}: {info['description']} | params: {info['params']}"
        for name, info in TOOLS.items()
    ])

    plan_prompt = f"""You are a tool-calling AI assistant for an Indian job market database.

AVAILABLE TOOLS:
{tools_desc}

DATABASE SCHEMA:
{TABLE_SCHEMA}

CONVERSATION HISTORY:
{history_str}

USER QUESTION: {user_question}

Rules:
1. Default to naukri_jobs tools for most job listing queries
2. Use jobs table tools when salary/industry/education/payrate data is needed
3. Use both tables when comparing or getting a complete picture
4. If no tool fits AND DB data is needed → use run_custom_sql with valid PostgreSQL SELECT
5. For general knowledge or advice → use general_advice
6. You can call multiple tools
7. When user says "at most N" or "limit N" → pass limit as integer
8. Never pass limit as string

Respond ONLY with a JSON array. No markdown, no explanation:
[
  {{
    "tool": "tool_name",
    "params": {{}},
    "reason": "one line why"
  }}
]
"""

    try:
        response = llm.invoke(plan_prompt)
        content = re.sub(r"```json|```", "", response.content.strip()).strip()
        tool_calls = json.loads(content)
        if isinstance(tool_calls, dict):
            tool_calls = [tool_calls]
        return tool_calls
    except Exception:
        return [{"tool": "general_advice", "params": {"question": user_question}, "reason": "fallback"}]


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
            data = func(**filtered)
        else:
            data = {"error": f"Unknown tool: {tool_name}"}
        results.append({"tool": tool_name, "reason": reason, "data": data})
    return results


def generate_final_answer(user_question, tool_results, chat_history, is_hindi=False):
    results_str = ""
    for r in tool_results:
        results_str += f"\n[Tool: {r['tool']} | Reason: {r['reason']}]\n"
        results_str += json.dumps(r["data"], indent=2, default=str)
        results_str += "\n"

    history_str = ""
    for msg in (chat_history or [])[-4:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content'][:300]}\n"

    language_instruction = (
        "IMPORTANT: The user asked in Hindi. You MUST respond entirely in Hindi."
        if is_hindi else "Respond in English."
    )

    answer_prompt = f"""You are an expert career advisor with access to real Indian job market data.

CONVERSATION HISTORY:
{history_str}

USER QUESTION: {user_question}

DATA FETCHED FROM DATABASE:
{results_str}

Instructions:
- Use real data to give specific, accurate, helpful answers
- Highlight key numbers, companies, skills, trends
- If data is empty or has errors, say so and answer from general knowledge
- Be conversational, not robotic
- Use bullet points or tables where helpful
- Keep it concise but complete
- {language_instruction}
"""
    try:
        response = llm.invoke(answer_prompt)
        return response.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"


# ------------------------------------------------
# Debug Panel
# ------------------------------------------------
with st.expander("🔧 Debug Panel"):
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Test naukri_jobs"):
            try:
                res = supabase.table("naukri_jobs").select("*").limit(2).execute()
                st.success(f"✅ {len(res.data)} rows")
                if res.data:
                    st.dataframe(res.data)
                else:
                    st.warning("Empty")
            except Exception as e:
                st.error(f"❌ {e}")
    with col2:
        if st.button("Test jobs table"):
            try:
                res = supabase.table("jobs").select("*").limit(2).execute()
                st.success(f"✅ {len(res.data)} rows")
                if res.data:
                    st.dataframe(res.data)
                else:
                    st.warning("Empty")
            except Exception as e:
                st.error(f"❌ {e}")
    with col3:
        if st.button("Test Translation"):
            sample = "मुंबई में नौकरियां"
            translated = translate_to_english(sample)
            st.success(f"Input: {sample}")
            st.info(f"Translated: {translated}")


# ------------------------------------------------
# Chat UI
# ------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "👋 Hi! I'm your Career Intelligence Assistant.\n\n"
            "I search across **Naukri + Jobs** databases and can calculate **AI Risk Metrics**.\n\n"
            "Try asking:\n"
            "- *Calculate my risk score* 🎯\n"
            "- *What is my AI replacement risk?*\n"
            "- *Give me latest job postings for data analyst at most 10*\n"
            "- *How many BPO jobs are in Delhi?*\n"
            "- *What skills are needed for data analyst roles?*\n"
            "- *मेरा जोखिम स्कोर क्या है?* 🙏"
        )
    }]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask about jobs, skills, salaries, or calculate your risk score..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    is_hindi = bool(re.search("[\u0900-\u097F]", prompt))
    processing_prompt = translate_to_english(prompt) if is_hindi else prompt

    if is_hindi:
        with st.sidebar:
            st.info(f"🌐 Translated: *{processing_prompt}*")

    # ✅ Check if this is a risk metrics conversation
    risk_answer = handle_risk_conversation(processing_prompt, is_hindi)

    if risk_answer:
        st.session_state.messages.append({"role": "assistant", "content": risk_answer})
        st.chat_message("assistant").write(risk_answer)
    else:
        with st.spinner("🧠 Thinking..."):
            tool_calls = plan_tool_call(processing_prompt, st.session_state.messages[:-1])

            with st.sidebar:
                st.subheader("🔍 Tool Plan")
                st.caption("What the AI decided to do:")
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
                is_hindi=is_hindi
            )

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)