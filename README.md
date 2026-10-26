# CareerBuddy – AI Career Guidance Chatbot

An AI-powered career guidance chatbot built using Python, Streamlit, and Google Gemini API.

## Live Demo

🚀 https://careerbuddy-ai-guidance.streamlit.app/

## Features

* Personalized Career Recommendations
* Skill Gap Analysis
* Learning Resources
* 90-Day Learning Plan
* Resume Analyzer
* Mock Interview
* AI Mentor Chat
* Hindi and English Support

## Tech Stack

* Python
* Streamlit
* Google Gemini API

## Developer

Anjali Prajapati

GitHub: https://github.com/Anjali3445
# AI-Powered Career Guidance Chatbot

Suggests career paths, skill gaps, learning resources and a 90-day plan from a user's profile,
then lets the user ask follow-up questions in a chat. Built with **Python + Streamlit + Google Gemini (LLM, free tier)**.

## Features
- Profile input: education, branch, skills, interests, goals, work preference, weekly hours
- LLM-generated structured report: 3 career paths with match score, skill gap, resources, 90-day plan
- Context-aware streaming chatbot (knows your profile and roadmap)
- Download roadmap as Markdown
- Robust JSON parsing with retry; no hallucinated URLs by design

## Architecture
```
User profile (Streamlit form)
        |
        v
career_engine.generate_roadmap()  --> LLM (JSON report)
        |
        v
Tabs: Paths | Skill Gap | Resources | 90-Day Plan
        |
        v
career_engine.chat_stream()  --> LLM (profile + roadmap as context)
```

## Run locally
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # Windows: copy .env.example .env
# put your GEMINI_API_KEY in .env  (free key at aistudio.google.com/apikey)
streamlit run app.py
```
Open http://localhost:8501

## Deploy

### Option 1: Streamlit Community Cloud (free, easiest)
1. Push this folder to a public/private GitHub repo (`.env` is git-ignored).
2. Go to share.streamlit.io, sign in with GitHub, click **Create app**.
3. Select repo, branch, main file `app.py`.
4. **Advanced settings > Secrets**, paste: `GEMINI_API_KEY = "your_key"`
5. Deploy. You get a public URL to submit on the platform.

### Option 2: Docker / Render / Railway
```bash
docker build -t career-bot .
docker run -p 8501:8501 -e GEMINI_API_KEY=your_key career-bot
```
On Render: New > Web Service > connect repo > Environment: Docker > add env var `GEMINI_API_KEY`.

### Option 3: Hugging Face Spaces
Create a Space with the Streamlit SDK, upload files, add `GEMINI_API_KEY` under Settings > Secrets.

## Security
Never commit your API key. Use `.env` locally and platform secrets in production.

## Future scope
Resume (PDF) upload and analysis, live job-market data, mock interview mode, multilingual support,
user accounts to save roadmaps.
