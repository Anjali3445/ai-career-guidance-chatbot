"""Core LLM logic for the AI-Powered Career Guidance Chatbot."""
import json
import os
import re
import time
from typing import Dict, Generator, List

from google import genai
from google.genai import types

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
# Backup models used automatically if the main one is busy or unavailable
MODELS = list(dict.fromkeys([MODEL, "gemini-flash-latest", "gemini-3.5-flash-lite", "gemini-2.5-flash"]))
RETRIES_PER_MODEL = 2
REQUEST_TIMEOUT_MS = 60000  # never wait more than 60s per request
BUSY_MSG = ("Gemini is busy or your free quota is used up. Wait a minute and try again. "
            "If it keeps failing, run: python test_api.py")

# Language shown in the UI -> description given to the LLM
LANGUAGES = {
    "English": "English",
    "हिंदी (Hindi)": "Hindi (Devanagari script)",
    "Hinglish": "Hinglish (Hindi written in English letters)",
}


# ----------------------------------------------------------------- helpers
def _retryable(e: Exception) -> bool:
    m = str(e).upper()
    return any(k in m for k in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED",
                                "OVERLOADED", "HIGH DEMAND", "500", "INTERNAL", "DEADLINE",
                                "TIMEOUT", "TIMED OUT", "DISCONNECT", "CONNECTION", "REMOTEPROTOCOL",
                                "RESET", "READERROR", "CONNECTERROR"))


def _model_missing(e: Exception) -> bool:
    m = str(e).upper()
    return "404" in m or "NOT_FOUND" in m or "NOT FOUND" in m


def _config(model: str, **kw):
    """Build generation config. Thinking is switched off on 2.5 models so replies are much faster."""
    if model.startswith("gemini-2.5"):
        kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    return types.GenerateContentConfig(**kw)


_BAD_MODELS = set()  # models that returned 404 for this key; skipped on later calls
_SKIP_WORDS = ("image", "tts", "live", "audio", "embed", "exp", "robotics", "computer", "veo", "imagen")


def _discover_models(client) -> List[str]:
    """Ask the API which Flash models this key can use (fallback when the hard-coded names are gone)."""
    try:
        names = []
        for m in client.models.list():
            n = (m.name or "").replace("models/", "")
            acts = getattr(m, "supported_actions", None) or []
            if "flash" in n and "generateContent" in acts and not any(w in n for w in _SKIP_WORDS):
                names.append(n)
        return sorted(set(names), reverse=True)[:4]
    except Exception:
        return []


def _candidates(client) -> List[str]:
    models = [m for m in MODELS if m not in _BAD_MODELS]
    if not models:
        models = [m for m in _discover_models(client) if m not in _BAD_MODELS]
    return models


def _final_error(errors: Dict[str, Exception]) -> RuntimeError:
    """Turn the per-model errors into one clear message."""
    joined = " ".join(str(e).upper() for e in errors.values())
    if any(k in joined for k in ("DISCONNECT", "CONNECTION", "CONNECTERROR")):
        return RuntimeError("Connection to Gemini failed. Check your internet / VPN / firewall and try again.")
    if errors and all(_model_missing(e) for e in errors.values()):
        return RuntimeError("None of the Gemini models are available for your API key. Run: python test_api.py "
                            "to see which models work, then set GEMINI_MODEL in your .env file.")
    if any(_retryable(e) for e in errors.values()):
        return RuntimeError(BUSY_MSG)
    detail = "; ".join(f"{m}: {str(e)[:120]}" for m, e in errors.items())
    return RuntimeError(f"Could not get a valid response. {detail}")


def get_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key,
                        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))


def lang_rule(profile: Dict) -> str:
    """Extra instruction so JSON text values come back in the user's language."""
    lang = (profile or {}).get("language", "English")
    if lang == "English":
        return ""
    return (f"\nLANGUAGE: Write all human-readable text values in {lang}. Keep JSON keys, proper nouns "
            f"(course names, platforms, technologies) and these exact values in English: "
            f"priority (High/Medium/Low), cost (Free/Paid), type (Course/Book/Project/Docs).")


def chat_lang_rule(profile: Dict) -> str:
    lang = (profile or {}).get("language", "English")
    return "" if lang == "English" else f"\nAlways reply in {lang}."


def profile_to_text(p: Dict, max_resume: int = 5000) -> str:
    text = (
        f"Name: {p.get('name') or 'Not given'}\n"
        f"Education: {p['education']} | Branch/Field: {p['branch']} | Year/Status: {p['year']}\n"
        f"Experience level: {p['experience']}\n"
        f"Current skills: {', '.join(p['skills']) or 'None listed'}\n"
        f"Interests: {', '.join(p['interests']) or 'None listed'}\n"
        f"Career goal / dream: {p.get('goal') or 'Not sure yet'}\n"
        f"Work preference: {p['work_style']}\n"
        f"Hours per week available for learning: {p['hours']}"
    )
    if p.get("resume_text"):
        text += f"\n\nRESUME TEXT (from uploaded PDF):\n{p['resume_text'][:max_resume]}"
    return text


def extract_pdf_text(file_obj, max_chars: int = 8000) -> str:
    """Read text from an uploaded PDF resume."""
    from pypdf import PdfReader

    reader = PdfReader(file_obj)
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:max_chars]


def _extract_json(text: str) -> Dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Model did not return JSON.")
    return json.loads(text[start: end + 1])


def _json_call(client, system: str, prompt: str, temperature: float = 0.4,
               max_tokens: int = 8000) -> Dict:
    """Call the LLM and return parsed JSON. Streams (keeps connection alive), retries, falls back."""
    errors: Dict[str, Exception] = {}
    for _round in range(2):  # round 2 = auto-discovered models if every listed model is gone
        models = _candidates(client)
        if not models:
            break
        all_missing = True
        for model in models:
            for attempt in range(RETRIES_PER_MODEL):
                try:
                    stream = client.models.generate_content_stream(
                        model=model,
                        contents=prompt,
                        config=_config(model,
                                       system_instruction=system,
                                       temperature=temperature,
                                       max_output_tokens=max_tokens,
                                       response_mime_type="application/json"),
                    )
                    text = "".join(c.text for c in stream if c.text)
                    return _extract_json(text)
                except (ValueError, json.JSONDecodeError, TypeError) as e:  # bad JSON -> try again
                    errors[model] = e
                    all_missing = False
                except Exception as e:
                    errors[model] = e
                    if _model_missing(e):
                        _BAD_MODELS.add(model)
                        break  # try next model
                    all_missing = False
                    if not _retryable(e):
                        raise
                    time.sleep(2 * (attempt + 1))
        if not all_missing:
            break
    raise _final_error(errors)


# ----------------------------------------------------------------- roadmap
ROADMAP_SYSTEM = """You are an expert career counsellor for students and early-career
professionals, with deep knowledge of the Indian and global job market.
Given a user profile (and resume text if present), produce a realistic, personalised career guidance report.
Rules:
- Respond with ONE valid JSON object only. No markdown, no commentary.
- Be honest: base match scores on the profile, do not flatter.
- Do NOT invent URLs. Give resource title + platform name only.
- Salary figures are rough estimates; state them as ranges in INR LPA.
"""

ROADMAP_SCHEMA = """{
  "summary": "2-3 sentence personalised summary of the profile",
  "career_paths": [
    {"title": "", "match_score": 0, "why": "", "entry_roles": ["", ""],
     "salary_range_lpa": "e.g. 4-8", "growth_outlook": "High/Medium/Low + one line"}
  ],
  "skill_gap": [
    {"skill": "", "priority": "High/Medium/Low", "already_have": false, "note": ""}
  ],
  "resources": [
    {"skill": "", "title": "", "platform": "", "type": "Course/Book/Project/Docs", "cost": "Free/Paid"}
  ],
  "roadmap_90_days": [
    {"phase": "Days 1-30", "focus": "", "actions": ["", ""]}
  ],
  "next_step": "One concrete action to take this week"
}"""


def generate_roadmap(client, profile: Dict) -> Dict:
    """Structured career report (3 paths, skill gap, resources, 90-day plan)."""
    prompt = (
        f"USER PROFILE:\n{profile_to_text(profile)}\n\n"
        f"Return JSON matching this schema exactly (3 career_paths, 6-10 skill_gap items, "
        f"6-8 resources, 3 roadmap phases):\n{ROADMAP_SCHEMA}"
    )
    return _json_call(client, ROADMAP_SYSTEM + lang_rule(profile), prompt)


# ----------------------------------------------------------------- resume
RESUME_SYSTEM = """You are a senior recruiter and resume coach. Review the candidate's resume text
for the target role. Be specific, honest and practical. Base everything ONLY on the resume text given.
Respond with ONE valid JSON object only. No markdown, no commentary."""

RESUME_SCHEMA = """{
  "resume_score": 0,
  "summary": "2-3 sentence overall verdict",
  "skills_found": ["skills actually present in the resume"],
  "strengths": ["3-5 specific strengths"],
  "improvements": [{"issue": "what is weak or missing", "fix": "exact way to fix it"}],
  "missing_keywords": ["keywords/skills recruiters expect for the target role but are absent"],
  "ats_tips": ["3-5 tips to pass Applicant Tracking Systems"]
}"""


def analyze_resume(client, profile: Dict, target_role: str) -> Dict:
    prompt = (
        f"TARGET ROLE: {target_role or 'General entry-level role'}\n\n"
        f"CANDIDATE:\n{profile_to_text(profile, max_resume=8000)}\n\n"
        f"Return JSON matching this schema (4-6 improvements, 5-10 missing_keywords):\n{RESUME_SCHEMA}"
    )
    return _json_call(client, RESUME_SYSTEM + lang_rule(profile), prompt, temperature=0.3)


# ----------------------------------------------------------------- mock interview
INTERVIEW_SYSTEM = """You are a friendly but rigorous interviewer for entry-level roles.
Respond with ONE valid JSON object only. No markdown, no commentary."""


def generate_questions(client, profile: Dict, role: str, level: str, n: int) -> List[Dict]:
    schema = ('{"questions": [{"question": "", "type": "Technical/HR/Behavioural", '
              '"what_they_look_for": "one short line"}]}')
    prompt = (
        f"Create {n} interview questions for the role '{role}' at {level} difficulty.\n"
        f"Candidate profile:\n{profile_to_text(profile, max_resume=2500)}\n\n"
        f"Mix technical, behavioural and HR questions (mostly technical). Questions must be in English.\n"
        f"Return JSON like:\n{schema}"
    )
    data = _json_call(client, INTERVIEW_SYSTEM, prompt, temperature=0.7, max_tokens=3000)
    qs = [q for q in data.get("questions", []) if isinstance(q, dict) and q.get("question")]
    if not qs:
        raise RuntimeError("No interview questions were generated. Please try again.")
    return qs[:n]


def evaluate_answer(client, profile: Dict, role: str, level: str, question: str, answer: str) -> Dict:
    schema = ('{"score": 0, "feedback": "2-3 sentences", "strengths": ["..."], '
              '"improve": ["..."], "model_answer": "short outline of an ideal answer"}')
    prompt = (
        f"Role: {role} | Difficulty: {level}\nQuestion: {question}\n"
        f"Candidate answer: {answer[:3000]}\n\n"
        f"Score the answer from 0 to 10 (be fair but strict; vague or empty answers score low). "
        f"Return JSON like:\n{schema}"
    )
    return _json_call(client, INTERVIEW_SYSTEM + lang_rule(profile), prompt, temperature=0.3, max_tokens=2000)


def interview_summary(client, profile: Dict, role: str, level: str, results: List[Dict]) -> Dict:
    qa = "\n".join(
        f"Q{i + 1}: {r['question']}\nAnswer: {str(r['answer'])[:800]}\nScore: {r.get('score', 0)}/10"
        for i, r in enumerate(results)
    )
    schema = ('{"overall_score": 0, "readiness": "Not ready / Almost ready / Interview ready", '
              '"summary": "3 sentences", "strengths": ["..."], "improvements": ["..."], "next_steps": ["..."]}')
    prompt = (
        f"Role: {role} | Difficulty: {level}\nInterview transcript:\n{qa}\n\n"
        f"Give a final interview report. overall_score is 0-100. Return JSON like:\n{schema}"
    )
    return _json_call(client, INTERVIEW_SYSTEM + lang_rule(profile), prompt, temperature=0.3, max_tokens=2000)


# ----------------------------------------------------------------- chat
CHAT_SYSTEM = """You are CareerBuddy, a friendly and honest AI career guidance assistant.
Use the user's profile and the generated roadmap (below) to answer follow-up questions
about careers, skills, interview prep, resumes and learning plans.
Keep answers clear and practical (short paragraphs or bullets). If unsure, say so.
Never invent URLs or guarantee jobs/salaries. Stay on career-related topics.{lang}

USER PROFILE:
{profile}

GENERATED ROADMAP (JSON):
{roadmap}
"""


def chat_stream(client, profile: Dict, roadmap: Dict, history: List[Dict]) -> Generator[str, None, None]:
    """Stream a follow-up answer token by token (with retry/fallback before streaming starts)."""
    system = CHAT_SYSTEM.format(
        lang=chat_lang_rule(profile),
        profile=profile_to_text(profile, max_resume=2500),
        roadmap=json.dumps(roadmap, indent=1) if roadmap else "Not generated yet.",
    )
    contents = [
        types.Content(
            role="model" if m["role"] == "assistant" else "user",
            parts=[types.Part(text=m["content"])],
        )
        for m in history[-12:]  # keep last 12 turns for context
    ]
    errors: Dict[str, Exception] = {}
    for _round in range(2):
        models = _candidates(client)
        if not models:
            break
        all_missing = True
        for model in models:
            for attempt in range(RETRIES_PER_MODEL):
                started = False
                try:
                    for chunk in client.models.generate_content_stream(
                        model=model,
                        contents=contents,
                        config=_config(model, system_instruction=system, temperature=0.6,
                                       max_output_tokens=2000),
                    ):
                        if chunk.text:
                            started = True
                            yield chunk.text
                    return
                except Exception as e:
                    errors[model] = e
                    if started:
                        raise
                    if _model_missing(e):
                        _BAD_MODELS.add(model)
                        break
                    all_missing = False
                    if not _retryable(e):
                        raise
                    time.sleep(2 * (attempt + 1))
        if not all_missing:
            break
    raise _final_error(errors)


def roadmap_to_markdown(r: Dict) -> str:
    md = ["# My Career Roadmap\n", r.get("summary", ""), "\n## Career Paths"]
    for c in r.get("career_paths", []):
        md.append(f"### {c['title']} ({c['match_score']}% match)\n{c['why']}\n"
                  f"- Entry roles: {', '.join(c['entry_roles'])}\n"
                  f"- Salary: {c['salary_range_lpa']} LPA\n- Outlook: {c['growth_outlook']}")
    md.append("\n## Skill Gap")
    for s in r.get("skill_gap", []):
        md.append(f"- **{s['skill']}** [{s['priority']}] {'(have)' if s['already_have'] else ''} {s['note']}")
    md.append("\n## Learning Resources")
    for x in r.get("resources", []):
        md.append(f"- {x['title']} - {x['platform']} ({x['type']}, {x['cost']}) for {x['skill']}")
    md.append("\n## 90-Day Plan")
    for p in r.get("roadmap_90_days", []):
        md.append(f"### {p['phase']}: {p['focus']}")
        md += [f"- {a}" for a in p["actions"]]
    md.append(f"\n**Next step:** {r.get('next_step', '')}")
    return "\n".join(md)