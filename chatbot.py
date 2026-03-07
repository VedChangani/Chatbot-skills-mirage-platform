import streamlit as st
import re
import json
import inspect
import os
from datetime import datetime, timedelta
from supabase import create_client
from langchain_groq import ChatGroq

# ------------------------------------------------
# Config
# ------------------------------------------------
st.set_page_config(page_title="Career AI", page_icon="💼", layout="centered")
st.title("💼 Career Intelligence Chatbot")
st.caption("Understands your question → picks the right tool → fetches real data")

# ------------------------------------------------
# Credentials - reads from Streamlit Secrets or env
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
Table: jobs
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
# TOOLS REGISTRY
# ------------------------------------------------
TOOLS = {
    "count_jobs": {
        "description": "Count total number of jobs matching role and/or city",
        "params": ["role", "city", "months"]
    },
    "list_jobs": {
        "description": "List job postings with details for a role/city",
        "params": ["role", "city", "months", "limit"]
    },
    "top_skills": {
        "description": "Get most in-demand skills for a job role",
        "params": ["role"]
    },
    "industry_breakdown": {
        "description": "Show which industries are hiring for a role",
        "params": ["role", "city"]
    },
    "salary_insights": {
        "description": "Show pay/salary info for a role",
        "params": ["role", "city"]
    },
    "recent_jobs": {
        "description": "Get jobs posted in last N months",
        "params": ["months", "role"]
    },
    "company_jobs": {
        "description": "List jobs from a specific company",
        "params": ["company", "role"]
    },
    "run_custom_sql": {
        "description": "Run a custom PostgreSQL SELECT query for anything not covered by other tools",
        "params": ["sql"]
    },
    "general_advice": {
        "description": "Answer career/certification/AI risk questions using LLM knowledge only, no DB needed",
        "params": ["question"]
    }
}

# ------------------------------------------------
# TOOL IMPLEMENTATIONS
# ------------------------------------------------

def count_jobs(role="", city="", months=None):
    try:
        q = supabase.table("jobs").select("*", count="exact")
        if role:
            q = q.ilike("jobtitle", f"%{role}%")
        if city:
            q = q.ilike("joblocation_address", f"%{city}%")
        if months:
            cutoff = (datetime.utcnow() - timedelta(days=30 * months)).strftime("%Y-%m-%d")
            q = q.gte("postdate", cutoff)
        res = q.execute()
        return {"count": res.count or 0}
    except Exception as e:
        return {"error": str(e)}


def list_jobs(role="", city="", months=None, limit=8):
    try:
        q = supabase.table("jobs").select(
            "jobtitle, company, joblocation_address, skills, payrate, postdate, experience"
        )
        if role:
            q = q.ilike("jobtitle", f"%{role}%")
        if city:
            q = q.ilike("joblocation_address", f"%{city}%")
        if months:
            cutoff = (datetime.utcnow() - timedelta(days=30 * months)).strftime("%Y-%m-%d")
            q = q.gte("postdate", cutoff)
        res = q.limit(limit).execute()
        return {"jobs": res.data or []}
    except Exception as e:
        return {"error": str(e)}


def top_skills(role=""):
    try:
        q = supabase.table("jobs").select("skills")
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
        cutoff = (datetime.utcnow() - timedelta(days=30 * months)).strftime("%Y-%m-%d")
        q = supabase.table("jobs").select(
            "jobtitle, company, joblocation_address, postdate, skills"
        ).gte("postdate", cutoff)
        if role:
            q = q.ilike("jobtitle", f"%{role}%")
        res = q.limit(15).execute()
        return {"jobs": res.data or [], "months": months}
    except Exception as e:
        return {"error": str(e)}


def company_jobs(company="", role=""):
    try:
        q = supabase.table("jobs").select(
            "jobtitle, company, joblocation_address, skills, payrate, postdate"
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
    "count_jobs": count_jobs,
    "list_jobs": list_jobs,
    "top_skills": top_skills,
    "industry_breakdown": industry_breakdown,
    "salary_insights": salary_insights,
    "recent_jobs": recent_jobs,
    "company_jobs": company_jobs,
    "run_custom_sql": run_custom_sql,
    "general_advice": general_advice,
}

# ------------------------------------------------
# BRAIN: LLM plans which tools to call
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

    plan_prompt = f"""You are a tool-calling AI assistant for a job market database.

AVAILABLE TOOLS:
{tools_desc}

DATABASE SCHEMA:
{TABLE_SCHEMA}

CONVERSATION HISTORY:
{history_str}

USER QUESTION: {user_question}

Your job:
1. Understand what the user wants
2. Pick the best tool(s) to answer it
3. If no predefined tool fits AND database data is needed → use "run_custom_sql" with a valid PostgreSQL SELECT query
4. If it is general knowledge or advice → use "general_advice"
5. You can call multiple tools if needed

Respond ONLY with a JSON array. No markdown, no explanation, no extra text:
[
  {{
    "tool": "tool_name",
    "params": {{
      "param1": "value1"
    }},
    "reason": "one line why"
  }}
]

For run_custom_sql, write valid PostgreSQL. Table name is "jobs". Only SELECT is allowed.
"""

    try:
        response = llm.invoke(plan_prompt)
        content = response.content.strip()
        content = re.sub(r"```json|```", "", content).strip()
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


def generate_final_answer(user_question, tool_results, chat_history):
    results_str = ""
    for r in tool_results:
        results_str += f"\n[Tool: {r['tool']} | Reason: {r['reason']}]\n"
        results_str += json.dumps(r["data"], indent=2, default=str)
        results_str += "\n"

    history_str = ""
    for msg in (chat_history or [])[-4:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content'][:300]}\n"

    answer_prompt = f"""You are an expert career advisor with access to real Indian job market data.

CONVERSATION HISTORY:
{history_str}

USER QUESTION: {user_question}

DATA FETCHED FROM DATABASE:
{results_str}

Instructions:
- Use the real data above to give specific, accurate, helpful answers
- Highlight key numbers, companies, skills, trends from the data
- If data is empty or has errors, say so and answer from general knowledge
- Be conversational, not robotic
- Use bullet points or tables where helpful
- If user asked in Hindi, respond in Hindi
- Keep it concise but complete
"""
    try:
        response = llm.invoke(answer_prompt)
        return response.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"


# ------------------------------------------------
# Translation
# ------------------------------------------------
def translate_to_english(text):
    try:
        r = llm.invoke(f"Translate to English, return ONLY the translation, nothing else:\n{text}")
        return r.content.strip()
    except Exception:
        return text


# ------------------------------------------------
# Debug Panel
# ------------------------------------------------
with st.expander("🔧 Debug: Test Supabase Connection"):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run Test Query"):
            try:
                res = supabase.table("jobs").select("*").limit(3).execute()
                st.success(f"✅ Got {len(res.data)} rows")
                if res.data:
                    st.write("Columns:", list(res.data[0].keys()))
                    st.dataframe(res.data)
                else:
                    st.warning("⚠️ Empty — check table name or RLS")
            except Exception as e:
                st.error(f"❌ {e}")
    with col2:
        if st.button("Test Tool Planner"):
            test = "How many developer jobs are in Mumbai?"
            plan = plan_tool_call(test, [])
            st.json(plan)


# ------------------------------------------------
# Chat UI
# ------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "👋 Hi! I'm your Career Intelligence Assistant.\n\n"
            "I understand your question, pick the right tool, and fetch real data.\n\n"
            "Try asking:\n"
            "- *How many BPO jobs are in Delhi?*\n"
            "- *What skills are needed for data analyst roles?*\n"
            "- *Which companies are hiring the most?*\n"
            "- *Show me jobs posted in the last 2 months*\n"
            "- *What is the AI risk for call center jobs?*\n"
            "- *Which city has the most openings?*\n\n"
            "You can also ask in **Hindi**! 🙏"
        )
    }]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask anything about jobs, skills, salaries, risk..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    is_hindi = bool(re.search("[\u0900-\u097F]", prompt))
    processing_prompt = translate_to_english(prompt) if is_hindi else prompt

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
            st.session_state.messages[:-1]
        )

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)