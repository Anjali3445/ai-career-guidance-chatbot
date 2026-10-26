"""CareerBuddy - AI-Powered Career Guidance Chatbot (Streamlit UI)."""
import html
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from career_engine import (LANGUAGES, analyze_resume, chat_stream, evaluate_answer, extract_pdf_text,
                           generate_questions, generate_roadmap, get_client, interview_summary,
                           roadmap_to_markdown)

load_dotenv()
st.set_page_config(page_title="CareerBuddy | AI Career Guidance", page_icon="🎯", layout="wide")

CSS = """
<style>
.block-container {padding-top: 3.5rem; max-width: 1200px;}
.hero {background: linear-gradient(135deg,#6C5CE7 0%,#3B82F6 55%,#06B6D4 100%); border-radius: 20px;
       padding: 1.8rem 2.1rem; color: #fff; margin-bottom: 1.2rem; box-shadow: 0 10px 30px rgba(59,130,246,.25);}
.hero h1 {margin: 0; font-size: 2.1rem; color: #fff; padding: 0;}
.hero p {margin: .4rem 0 .6rem; opacity: .95; font-size: 1.05rem;}
.chip {display: inline-block; padding: .22rem .7rem; border-radius: 999px; font-size: .8rem;
       margin: .15rem .25rem .15rem 0; background: rgba(127,127,127,.18);}
.hero .chip {background: rgba(255,255,255,.22); color: #fff;}
.chip.green {background: rgba(16,185,129,.18); color: #10b981;}
.chip.red {background: rgba(239,68,68,.18); color: #ef4444;}
.chip.orange {background: rgba(245,158,11,.18); color: #f59e0b;}
.chip.blue {background: rgba(59,130,246,.18); color: #3b82f6;}
.card {border: 1px solid rgba(127,127,127,.25); border-radius: 16px; padding: 1.1rem 1.3rem;
       margin-bottom: 1rem; background: rgba(127,127,127,.06);}
.card h3 {margin: 0 0 .2rem; padding: 0; font-size: 1.2rem;}
.card p {margin: .35rem 0;}
.card ul {margin: .3rem 0 0 1rem; padding: 0;}
.badge {float: right; padding: .3rem .85rem; border-radius: 12px; font-weight: 700; color: #fff;
        background: linear-gradient(135deg,#6C5CE7,#3B82F6);}
.bar {height: 8px; border-radius: 6px; background: rgba(127,127,127,.25); margin: .5rem 0 .7rem; overflow: hidden;}
.bar > div {height: 100%; background: linear-gradient(90deg,#6C5CE7,#06B6D4);}
.muted {opacity: .75; font-size: .9rem;}
.phase {border-top: 5px solid #6C5CE7;}
.question {border-left: 6px solid #6C5CE7; font-size: 1.1rem;}
.footer {text-align: center; opacity: .55; font-size: .85rem; margin-top: 2rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape(str(x))


def num(x, hi: int = 100) -> int:
    try:
        return max(0, min(hi, int(float(x))))
    except (TypeError, ValueError):
        return 0


def as_list(x) -> list:
    return x if isinstance(x, list) else []


def chips(items, cls="blue") -> str:
    return "".join(f'<span class="chip {cls}">{esc(i)}</span>' for i in as_list(items))


def get_api_key() -> str:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "")


API_KEY = get_api_key()

SKILL_OPTIONS = ["Python", "Java", "C/C++", "JavaScript", "SQL", "HTML/CSS", "React",
                 "Machine Learning", "Data Analysis", "Excel", "Cloud (AWS/Azure)",
                 "Git/GitHub", "Communication", "Problem Solving", "Networking",
                 "Cyber Security", "UI/UX Design", "Digital Marketing"]
INTEREST_OPTIONS = ["Software Development", "Data Science / AI", "Cloud & DevOps",
                    "Cyber Security", "Product / Management", "Design", "Testing / QA",
                    "Research", "Business / Consulting", "Entrepreneurship"]

for key, default in (("roadmap", None), ("profile", None), ("messages", []),
                     ("resume_report", None), ("iv", None)):
    st.session_state.setdefault(key, default)

# ---------------- Sidebar: profile input ----------------
with st.sidebar:
    st.markdown("## 🎯 CareerBuddy")
    st.caption("Tell us about yourself")
    with st.form("profile_form"):
        language_label = st.selectbox("Language / भाषा", list(LANGUAGES.keys()))
        name = st.text_input("Name (optional)")
        education = st.selectbox("Education", ["Diploma", "B.Tech/B.E.", "BCA/B.Sc", "MCA/M.Sc",
                                               "M.Tech/M.E.", "MBA", "Other"])
        branch = st.text_input("Branch / Field", placeholder="e.g. Computer Science")
        year = st.selectbox("Year / Status", ["1st year", "2nd year", "3rd year", "Final year",
                                              "Graduated", "Working professional"])
        experience = st.selectbox("Experience", ["Fresher", "Internship only", "0-2 years", "2+ years"])
        skills = st.multiselect("Current skills", SKILL_OPTIONS)
        extra = st.text_input("Other skills (comma separated)")
        interests = st.multiselect("Interests", INTEREST_OPTIONS)
        goal = st.text_area("Career goal (optional)", placeholder="e.g. Become an ML engineer")
        work_style = st.selectbox("Work preference", ["No preference", "Remote", "Office", "Hybrid", "Startup", "MNC"])
        hours = st.slider("Learning hours per week", 2, 40, 10)
        resume_file = st.file_uploader("Resume (PDF, optional)", type=["pdf"])
        submit = st.form_submit_button("Generate Career Roadmap 🚀", use_container_width=True, type="primary")

    if not API_KEY:
        API_KEY = st.text_input("Gemini API key", type="password",
                                help="Or set GEMINI_API_KEY in .env / secrets")

# ---------------- Hero ----------------
st.markdown(
    '<div class="hero"><h1>🎯 AI Career Guidance Chatbot</h1>'
    "<p>Get personalised career paths, skill gaps, learning resources and a 90-day plan, powered by AI.</p>"
    '<span class="chip">🧭 Career Paths</span><span class="chip">🧩 Skill Gap</span>'
    '<span class="chip">📚 Resources</span><span class="chip">🗓️ 90-Day Plan</span>'
    '<span class="chip">📄 Resume Analyzer</span><span class="chip">🎤 Mock Interview</span>'
    '<span class="chip">💬 AI Mentor Chat</span><span class="chip">🌐 Hindi / English</span></div>',
    unsafe_allow_html=True,
)

if submit:
    if not API_KEY:
        st.error("Please provide your Gemini API key.")
    elif not branch.strip():
        st.warning("Please enter your branch / field.")
    else:
        resume_text = ""
        if resume_file is not None:
            try:
                resume_text = extract_pdf_text(resume_file)
            except Exception as e:
                st.warning(f"Could not read the PDF: {e}")
            if not resume_text:
                st.warning("No readable text found in the PDF (it may be a scanned image). "
                           "Continuing without the resume.")
        all_skills = skills + [s.strip() for s in extra.split(",") if s.strip()]
        st.session_state.profile = dict(
            name=name, education=education, branch=branch, year=year, experience=experience,
            skills=all_skills, interests=interests, goal=goal, work_style=work_style, hours=hours,
            language=LANGUAGES[language_label], resume_text=resume_text)
        try:
            with st.spinner("Analysing your profile and building your roadmap (15-40 seconds)..."):
                st.session_state.roadmap = generate_roadmap(get_client(API_KEY), st.session_state.profile)
            st.session_state.messages = []
            st.session_state.resume_report = None
            st.session_state.iv = None
        except Exception as e:
            st.error(f"Something went wrong: {e}")


def render_bottom(show_feedback: bool):
    """Bottom section: tech stack, privacy, disclaimer, FAQ, feedback, footer."""
    st.divider()
    a, b, c = st.columns(3)
    a.markdown(
        '<div class="card"><h3>🛠️ Tech Stack</h3><p>'
        '<span class="chip blue">Python</span><span class="chip blue">Streamlit</span>'
        '<span class="chip blue">Gemini LLM</span><span class="chip blue">Pandas</span></p>'
        '<p class="muted">A prompt-engineered LLM returns structured JSON that powers every tab.</p></div>',
        unsafe_allow_html=True)
    b.markdown(
        '<div class="card"><h3>🔒 Privacy</h3>'
        '<p class="muted">Your profile and resume stay in your browser session and are not saved by this app. '
        'They are sent to the Gemini API only to generate your results.</p></div>',
        unsafe_allow_html=True)
    c.markdown(
        '<div class="card"><h3>⚠️ Disclaimer</h3>'
        '<p class="muted">Salary figures, scores and feedback are AI estimates. Use them as guidance, '
        'and verify with job portals and mentors before making decisions.</p></div>',
        unsafe_allow_html=True)

    st.markdown("### ❓ Frequently asked questions")
    faqs = [
        ("How does CareerBuddy create my roadmap?",
         "Your profile (and resume, if uploaded) is sent to an LLM with a structured prompt. The model returns "
         "career paths, skill gaps, resources and a 90-day plan, which the app shows in the tabs."),
        ("How do the Resume Analyzer and Mock Interview work?",
         "Upload a PDF resume in the sidebar and generate your roadmap. Then the Resume tab scores your resume "
         "for your target role, and the Mock Interview tab asks questions, scores each answer and gives a final report."),
        ("Can I use Hindi?",
         "Yes. Choose Hindi or Hinglish in the Language box. Results and feedback come in that language "
         "(interview questions stay in English, like real interviews)."),
        ("Can I save my roadmap?",
         "Yes. Open the 90-Day Plan tab and click Download roadmap to get a Markdown file."),
    ]
    for q, ans in faqs:
        with st.expander(q):
            st.write(ans)

    if show_feedback:
        st.markdown("### ⭐ Rate this roadmap")
        rating = st.feedback("stars")
        if rating is not None:
            st.success("Thanks for your feedback! 🙌")

    st.markdown(
        '<div class="footer">Made with Python, Streamlit and Gemini · '
        'AI suggestions are guidance, not guarantees</div>', unsafe_allow_html=True)


roadmap = st.session_state.roadmap
profile = st.session_state.profile

# ---------------- Landing page ----------------
if not roadmap:
    c1, c2, c3 = st.columns(3)
    cards = [("1️⃣ Share your profile", "Add your education, skills and goals in the sidebar. Upload your resume (optional)."),
             ("2️⃣ Get your roadmap", "AI suggests best-fit careers, your skill gaps, resources and a 90-day plan."),
             ("3️⃣ Practise and improve", "Analyse your resume, take a mock interview and chat with your AI mentor.")]
    for col, (t, d) in zip((c1, c2, c3), cards):
        col.markdown(f'<div class="card"><h3>{t}</h3><p class="muted">{d}</p></div>', unsafe_allow_html=True)
    st.info("👈 Fill in your profile in the sidebar and click **Generate Career Roadmap** to begin.")
    render_bottom(False)
    st.stop()

# ---------------- Results ----------------
paths = as_list(roadmap.get("career_paths"))
gap = as_list(roadmap.get("skill_gap"))
have = [s for s in gap if s.get("already_have")]
learn = [s for s in gap if not s.get("already_have")]
high = [s for s in learn if str(s.get("priority", "")).lower() == "high"]
top = max(paths, key=lambda p: num(p.get("match_score"))) if paths else {}

st.success(roadmap.get("summary", ""))

k1, k2, k3, k4 = st.columns(4)
with k1.container(border=True):
    st.metric("Best match", f"{num(top.get('match_score'))}%", top.get("title", "-"), delta_color="off")
with k2.container(border=True):
    st.metric("Skills to learn", len(learn))
with k3.container(border=True):
    st.metric("High priority", len(high))
with k4.container(border=True):
    st.metric("Hours / week", profile["hours"])

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🧭 Career Paths", "🧩 Skill Gap", "📚 Resources", "🗓️ 90-Day Plan",
    "📄 Resume Analyzer", "🎤 Mock Interview", "💬 Ask CareerBuddy"])

with tab1:
    if paths:
        df = pd.DataFrame({"Match %": [num(p.get("match_score")) for p in paths]},
                          index=[p.get("title", "") for p in paths])
        st.bar_chart(df, horizontal=True, height=180)
    for i, c in enumerate(paths, 1):
        score = num(c.get("match_score"))
        st.markdown(
            f'<div class="card"><span class="badge">{score}% match</span>'
            f'<h3>#{i} {esc(c.get("title", ""))}</h3>'
            f'<div class="bar"><div style="width:{score}%"></div></div>'
            f'<p>{esc(c.get("why", ""))}</p><div>{chips(c.get("entry_roles"))}</div>'
            f'<p class="muted">💰 Est. salary: <b>{esc(c.get("salary_range_lpa", "-"))} LPA</b> '
            f'&nbsp;|&nbsp; 📈 {esc(c.get("growth_outlook", ""))}</p></div>',
            unsafe_allow_html=True,
        )

with tab2:
    if gap:
        ready = len(have) / len(gap)
        st.markdown(f"**Skill readiness: {int(ready * 100)}%** ({len(have)} of {len(gap)} key skills)")
        st.progress(ready)
    colL, colR = st.columns(2)
    colL.markdown("#### ✅ Skills you have")
    for s in have:
        colL.markdown(f'<div class="card"><span class="chip green">{esc(s.get("skill", ""))}</span>'
                      f'<p class="muted">{esc(s.get("note", ""))}</p></div>', unsafe_allow_html=True)
    if not have:
        colL.caption("No matching skills listed yet. Add your skills in the sidebar.")
    colR.markdown("#### 🎯 Skills to learn")
    tone = {"high": "red", "medium": "orange", "low": "green"}
    for s in learn:
        pr = str(s.get("priority", "Medium"))
        colR.markdown(f'<div class="card"><span class="chip {tone.get(pr.lower(), "blue")}">{esc(pr)} priority</span> '
                      f'<b>{esc(s.get("skill", ""))}</b><p class="muted">{esc(s.get("note", ""))}</p></div>',
                      unsafe_allow_html=True)

with tab3:
    cols = st.columns(2)
    for i, r in enumerate(as_list(roadmap.get("resources"))):
        cost = str(r.get("cost", ""))
        cost_cls = "green" if cost.lower().startswith("free") else "orange"
        cols[i % 2].markdown(
            f'<div class="card"><h3>{esc(r.get("title", ""))}</h3>'
            f'<p class="muted">{esc(r.get("platform", ""))}</p>'
            f'<span class="chip blue">{esc(r.get("type", ""))}</span>'
            f'<span class="chip {cost_cls}">{esc(cost)}</span>'
            f'<span class="chip">for {esc(r.get("skill", ""))}</span></div>',
            unsafe_allow_html=True,
        )
    st.caption("Search the title on the platform to find the latest version.")

with tab4:
    phases = as_list(roadmap.get("roadmap_90_days"))
    if phases:
        pcols = st.columns(min(len(phases), 4))
        for col, p in zip(pcols, phases):
            items = "".join(f"<li>{esc(a)}</li>" for a in as_list(p.get("actions")))
            col.markdown(f'<div class="card phase"><span class="chip blue">{esc(p.get("phase", ""))}</span>'
                         f'<h3>{esc(p.get("focus", ""))}</h3><ul>{items}</ul></div>', unsafe_allow_html=True)
    st.info(f"🚀 **This week's next step:** {roadmap.get('next_step', '')}")
    st.download_button("⬇️ Download roadmap (Markdown)", roadmap_to_markdown(roadmap),
                       file_name="career_roadmap.md", type="primary")

# ---------------- Resume Analyzer ----------------
with tab5:
    if not profile.get("resume_text"):
        st.info("📄 Upload your resume (PDF) in the sidebar and click **Generate Career Roadmap** again "
                "to unlock the Resume Analyzer.")
    else:
        target = st.text_input("Target role", value=top.get("title") or profile.get("goal") or "")
        if st.button("🔍 Analyze my resume", type="primary"):
            try:
                with st.spinner("Reading your resume..."):
                    st.session_state.resume_report = analyze_resume(get_client(API_KEY), profile, target)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
        rep = st.session_state.resume_report
        if rep:
            score = num(rep.get("resume_score"))
            m, s = st.columns([1, 3])
            with m.container(border=True):
                st.metric("Resume score", f"{score}/100")
            s.markdown(f'<div class="card"><p>{esc(rep.get("summary", ""))}</p></div>', unsafe_allow_html=True)
            st.progress(score / 100)
            st.markdown("#### 🛠️ Skills found in your resume")
            st.markdown(chips(rep.get("skills_found"), "green") or "-", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.markdown("#### 💪 Strengths")
            for t in as_list(rep.get("strengths")):
                c1.markdown(f'<div class="card">{esc(t)}</div>', unsafe_allow_html=True)
            c2.markdown("#### 🔧 Improvements")
            for it in as_list(rep.get("improvements")):
                issue, fix = (it.get("issue", ""), it.get("fix", "")) if isinstance(it, dict) else (str(it), "")
                c2.markdown(f'<div class="card"><b>{esc(issue)}</b><p class="muted">{esc(fix)}</p></div>',
                            unsafe_allow_html=True)
            st.markdown("#### 🔑 Missing keywords for this role")
            st.markdown(chips(rep.get("missing_keywords"), "orange") or "-", unsafe_allow_html=True)
            st.markdown("#### 🤖 ATS tips")
            for t in as_list(rep.get("ats_tips")):
                st.markdown(f"- {t}")

# ---------------- Mock Interview ----------------
with tab6:
    iv = st.session_state.iv
    roles = [p.get("title", "") for p in paths if p.get("title")] or [profile.get("goal") or "Software Developer"]
    if iv is None:
        st.markdown("Practise with AI. You get questions one by one, a score and feedback for each answer, "
                    "and a final report.")
        c1, c2, c3 = st.columns(3)
        role = c1.selectbox("Interview for role", roles)
        level = c2.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
        n_q = c3.slider("Number of questions", 3, 8, 5)
        if st.button("🎤 Start mock interview", type="primary"):
            try:
                with st.spinner("Preparing your interview..."):
                    qs = generate_questions(get_client(API_KEY), profile, role, level, n_q)
                st.session_state.iv = dict(role=role, level=level, questions=qs, results=[], idx=0, summary=None)
                st.rerun()
            except Exception as e:
                st.error(f"Something went wrong: {e}")
    else:
        qs = iv["questions"]
        total = len(qs)
        st.caption(f"Role: **{iv['role']}** · Difficulty: **{iv['level']}**")
        if iv["idx"] >= total:
            # ----- final report -----
            if iv["summary"] is None:
                try:
                    with st.spinner("Preparing your final report..."):
                        iv["summary"] = interview_summary(get_client(API_KEY), profile, iv["role"],
                                                          iv["level"], iv["results"])
                except Exception as e:
                    st.error(f"Could not create the final report: {e}")
            sm = iv["summary"] or {}
            avg = sum(num(r.get("score"), 10) for r in iv["results"]) / max(len(iv["results"]), 1)
            a, b, c = st.columns(3)
            with a.container(border=True):
                st.metric("Average score", f"{avg:.1f}/10")
            with b.container(border=True):
                st.metric("Overall", f"{num(sm.get('overall_score'))}/100")
            with c.container(border=True):
                st.metric("Readiness", str(sm.get("readiness", "-")))
            st.markdown(f'<div class="card"><p>{esc(sm.get("summary", ""))}</p></div>', unsafe_allow_html=True)
            d1, d2, d3 = st.columns(3)
            for col, title, key in ((d1, "💪 Strengths", "strengths"), (d2, "🔧 Improve", "improvements"),
                                    (d3, "🚀 Next steps", "next_steps")):
                col.markdown(f"#### {title}")
                for t in as_list(sm.get(key)):
                    col.markdown(f"- {t}")
            st.markdown("#### 📝 Question-wise review")
            for i, r in enumerate(iv["results"], 1):
                with st.expander(f"Q{i}: {r['question']}  ·  {num(r.get('score'), 10)}/10"):
                    st.markdown(f"**Your answer:** {r['answer']}")
                    st.markdown(f"**Feedback:** {r.get('feedback', '')}")
                    if r.get("model_answer"):
                        st.markdown(f"**Ideal answer outline:** {r['model_answer']}")
            if st.button("🔁 Start a new interview", type="primary"):
                st.session_state.iv = None
                st.rerun()
        else:
            # ----- ask question i -----
            i = iv["idx"]
            q = qs[i]
            st.progress(i / total, text=f"Question {i + 1} of {total}")
            st.markdown(
                f'<div class="card question"><span class="chip blue">{esc(q.get("type", "Question"))}</span>'
                f'<p><b>{esc(q["question"])}</b></p>'
                f'<p class="muted">💡 They look for: {esc(q.get("what_they_look_for", ""))}</p></div>',
                unsafe_allow_html=True)
            if len(iv["results"]) <= i:
                ans = st.text_area("Your answer", key=f"iv_ans_{i}", height=160,
                                   placeholder="Type your answer here...")
                if st.button("Submit answer", type="primary"):
                    if len(ans.strip()) < 5:
                        st.warning("Please write your answer first.")
                    else:
                        try:
                            with st.spinner("Evaluating your answer..."):
                                ev = evaluate_answer(get_client(API_KEY), profile, iv["role"], iv["level"],
                                                     q["question"], ans)
                            iv["results"].append(dict(question=q["question"], answer=ans, **{
                                k: v for k, v in ev.items() if k != "question" and k != "answer"}))
                            st.rerun()
                        except Exception as e:
                            st.error(f"Something went wrong: {e}")
            else:
                r = iv["results"][i]
                st.markdown(
                    f'<div class="card"><span class="badge">{num(r.get("score"), 10)}/10</span><h3>Feedback</h3>'
                    f'<p>{esc(r.get("feedback", ""))}</p></div>', unsafe_allow_html=True)
                f1, f2 = st.columns(2)
                f1.markdown("**👍 What went well**")
                for t in as_list(r.get("strengths")):
                    f1.markdown(f"- {t}")
                f2.markdown("**🔧 How to improve**")
                for t in as_list(r.get("improve")):
                    f2.markdown(f"- {t}")
                if r.get("model_answer"):
                    with st.expander("See an ideal answer outline"):
                        st.write(r["model_answer"])
                label = "Next question ▶" if i + 1 < total else "Finish and see report 🏁"
                if st.button(label, type="primary"):
                    iv["idx"] += 1
                    st.rerun()
            if st.button("Quit interview"):
                st.session_state.iv = None
                st.rerun()

# ---------------- Chat ----------------
with tab7:
    st.caption("Ask follow-ups about interviews, resumes, projects, or compare careers.")
    quick = ["Give me 10 interview questions for my top path", "How can I improve my resume?",
             "Compare my top 2 career paths", "Suggest 3 portfolio projects for me"]
    qcols = st.columns(len(quick))
    for c, qk in zip(qcols, quick):
        if c.button(qk, use_container_width=True):
            st.session_state.pending = qk
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    prompt = st.chat_input("Ask a career question...") or st.session_state.pop("pending", None)
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                reply = st.write_stream(chat_stream(get_client(API_KEY), profile, roadmap,
                                                    st.session_state.messages))
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"Error: {e}")

render_bottom(True)