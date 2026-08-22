# SIGNAL — Research & Competitor Tracking Agent

An autonomous AI agent that tracks research trends, competitor moves, and industry news, and delivers a structured, decision-ready briefing — built for the **Research & Competitor Tracking** hackathon theme.

Unlike a static scraper, the agent **reasons** about what information it needs, **decides which tool to call**, observes the result, and loops until it has enough to answer — a full [ReAct](https://arxiv.org/abs/2210.03629) (Reason → Act → Observe → Final Answer) pattern, implemented with LangGraph and Google Gemini.

---

## What it does

Given a query like:

> "Track the latest competitor moves and funding news for OpenAI vs Anthropic this month."

the agent:
1. **Reasons** about what's needed to answer it
2. **Acts** — calls a tool (web search or research search), with a query it generates itself
3. **Observes** the tool's result
4. Repeats steps 1–3 as many times as needed, refining its search each round
5. Produces a **Final Answer**: Summary → Key Findings → Suggested Next Actions

You can watch every step of this live in the UI as it happens.

## Architecture

```
┌─────────────────┐      SSE stream       ┌──────────────────┐
│  frontend/       │ <──────────────────  │  server.py        │
│  index.html      │   POST /api/query     │  (FastAPI)         │
│  (live trace UI) │ ──────────────────>   │                    │
└─────────────────┘                       └─────────┬──────────┘
                                                       │
                                                       v
                                            ┌────────────────────┐
                                            │  main.py             │
                                            │  LangGraph ReAct      │
                                            │  agent (Gemini)       │
                                            └─────────┬──────────┘
                                                       │
                                    ┌──────────────────┴──────────────────┐
                                    v                                     v
                          ┌──────────────────┐                ┌──────────────────┐
                          │  web_search        │                │  research_search   │
                          │  (Tavily API)       │                │  (arXiv API)        │
                          │  news, funding,      │                │  academic papers,    │
                          │  competitor moves     │                │  research trends      │
                          └──────────────────┘                └──────────────────┘
```

## Tech stack

- **LLM**: Google Gemini, via `langchain-google-genai`
- **Agent framework**: LangGraph (`create_react_agent`)
- **Tools**: [Tavily](https://tavily.com) (live web/news search), [arXiv](https://arxiv.org) (academic research search)
- **Backend**: FastAPI, streaming responses over Server-Sent Events
- **Frontend**: vanilla HTML/CSS/JS — no build step, no framework — a "live intel terminal" UI

## Project structure

```
.
├── main.py              # Agent definition: tools, ReAct graph, CLI test loop
├── server.py             # FastAPI backend, streams agent steps over SSE
├── frontend/
│   └── index.html         # Demo UI — live trace + briefing panel
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get API keys (both free tier)
- **Gemini**: https://aistudio.google.com/apikey
- **Tavily**: https://tavily.com

Copy `.env.example` to `.env` and fill in your keys, **or** set them directly as environment variables:

**macOS / Linux:**
```bash
export GOOGLE_API_KEY="your-gemini-key"
export TAVILY_API_KEY="your-tavily-key"
```

**Windows PowerShell:**
```powershell
$env:GOOGLE_API_KEY="your-gemini-key"
$env:TAVILY_API_KEY="your-tavily-key"
```

### 3a. Run the agent standalone (terminal / CLI)
```bash
python main.py
```
Press Enter with no input to run 3 built-in sample queries, or type your own.

### 3b. Run with the full web UI (recommended for demo)
```bash
uvicorn server:app --reload --port 8000
```
Then open `frontend/index.html` directly in your browser (double-click it, or right-click → Open With → Browser).

## Example queries to try

- "Track the latest competitor moves and funding news for OpenAI vs Anthropic this month."
- "What's the current research trend in on-device LLM inference for mobile phones?"
- "Summarize recent news on autonomous delivery robots for last-mile delivery, plus any academic research on the topic."

## Notes

- `GOOGLE_API_KEY` and `TAVILY_API_KEY` are required — the app will exit with a clear message if either is missing.
- The Gemini model string in `main.py` may need updating over time as Google deprecates older model versions — check the error message if you get a `404 NOT_FOUND`, it names the current replacement model.
- This is a hackathon prototype: no auth, no rate limiting, no persistence beyond in-memory conversation state (`MemorySaver`). Not intended for production deployment as-is.
