\# SIGNAL — Research \& Competitor Tracking Agent



A \*\*multi-agent\*\* AI system that tracks research trends, competitor moves, and industry news, and delivers a structured, decision-ready briefing — built for the \*\*Research \& Competitor Tracking\*\* hackathon theme.



A \*\*Supervisor\*\* agent reads the query, decides which specialist(s) it needs, dispatches them, and synthesizes their findings into one final briefing. Each specialist independently runs a full \[ReAct](https://arxiv.org/abs/2210.03629) (Reason → Act → Observe → Final Answer) loop with its own tool before reporting back — this is genuine multi-agent orchestration, not one agent juggling two tools.



\---



\## Architecture — Supervisor + 2 specialist agents



| Agent | Responsibility | Tool | Reasoning pattern |

|---|---|---|---|

| \*\*Supervisor\*\* | Routes the query to the right specialist(s); synthesizes their findings into one final briefing | — (LLM reasoning only) | Structured routing decision + synthesis |

| \*\*CompetitorIntelAgent\*\* | Live news, competitor moves, funding rounds, product launches, market signals | `web\_search` (Tavily) | Full ReAct loop |

| \*\*ResearchTrendsAgent\*\* | Academic papers, technical research trends, state-of-the-art methods | `research\_search` (arXiv) | Full ReAct loop |



\*\*Flow for a query:\*\*

1\. Supervisor reasons about the query and decides: does this need CompetitorIntelAgent, ResearchTrendsAgent, or both?

2\. Each dispatched specialist runs its own independent Reason → Act → Observe loop, calling its tool as many times as needed, refining its search each round

3\. Each specialist returns a scoped summary of what it found — it does not see or care about the other specialist's domain

4\. The Supervisor \*\*collaborates the findings\*\*: it reads both specialists' summaries and synthesizes one final briefing — Summary → Key Findings (attributed per agent) → Suggested Next Actions



You can watch the Supervisor's routing decision and every specialist's live trace in the UI as it happens.



```

┌─────────────────┐      SSE stream       ┌──────────────────┐

│  frontend/       │ <──────────────────  │  server.py        │

│  index.html      │   POST /api/query     │  (FastAPI)         │

│  (live trace UI) │ ──────────────────>   │                    │

└─────────────────┘                       └─────────┬──────────┘

&#x20;                                                      │

&#x20;                                                      v

&#x20;                                         ┌──────────────────────┐

&#x20;                                         │  SUPERVISOR (main.py)  │

&#x20;                                         │  routes -> dispatches   │

&#x20;                                         │  -> synthesizes          │

&#x20;                                         └───────────┬────────────┘

&#x20;                                                      │

&#x20;                                   ┌──────────────────┴──────────────────┐

&#x20;                                   v                                     v

&#x20;                    ┌────────────────────────┐              ┌────────────────────────┐

&#x20;                    │  CompetitorIntelAgent     │              │  ResearchTrendsAgent      │

&#x20;                    │  (own ReAct loop)           │              │  (own ReAct loop)           │

&#x20;                    │  tool: web\_search             │              │  tool: research\_search        │

&#x20;                    │  (Tavily — news, funding,      │              │  (arXiv — papers, trends)       │

&#x20;                    │  competitor moves)               │              │                                    │

&#x20;                    └────────────────────────┘              └────────────────────────┘

```



\## Tech stack



\- \*\*LLM\*\*: Google Gemini, via `langchain-google-genai`

\- \*\*Agent framework\*\*: LangGraph (`create\_react\_agent` per specialist) + a custom Supervisor orchestration layer (routing via structured output, then synthesis)

\- \*\*Tools\*\*: \[Tavily](https://tavily.com) (live web/news search — CompetitorIntelAgent), \[arXiv](https://arxiv.org) (academic research search — ResearchTrendsAgent)

\- \*\*Backend\*\*: FastAPI, streaming responses over Server-Sent Events

\- \*\*Frontend\*\*: vanilla HTML/CSS/JS — no build step, no framework — a "live intel terminal" UI, showing the Supervisor's routing decision and each specialist's tagged trace



\## Project structure



```

.

├── main.py              # Supervisor + 2 specialist agents (CompetitorIntelAgent, ResearchTrendsAgent), CLI test loop

├── server.py             # FastAPI backend, streams supervisor routing + specialist traces + synthesis over SSE

├── frontend/

│   └── index.html         # Demo UI — routing card + tagged live trace + briefing panel

├── requirements.txt

├── .env.example

└── README.md

```



\## Setup



\### 1. Install dependencies

```bash

pip install -r requirements.txt

```



\### 2. Get API keys (both free tier)

\- \*\*Gemini\*\*: https://aistudio.google.com/apikey

\- \*\*Tavily\*\*: https://tavily.com



Copy `.env.example` to `.env` and fill in your keys, \*\*or\*\* set them directly as environment variables:



\*\*macOS / Linux:\*\*

```bash

export GOOGLE\_API\_KEY="your-gemini-key"

export TAVILY\_API\_KEY="your-tavily-key"

```



\*\*Windows PowerShell:\*\*

```powershell

$env:GOOGLE\_API\_KEY="your-gemini-key"

$env:TAVILY\_API\_KEY="your-tavily-key"

```



\### 3a. Run the agent standalone (terminal / CLI)

```bash

python main.py

```

Press Enter with no input to run 3 built-in sample queries, or type your own.



\### 3b. Run with the full web UI (recommended for demo)

```bash

uvicorn server:app --reload --port 8000

```

Then open `frontend/index.html` directly in your browser (double-click it, or right-click → Open With → Browser).



\## Example queries to try



\- "Track the latest competitor moves and funding news for OpenAI vs Anthropic this month."

\- "What's the current research trend in on-device LLM inference for mobile phones?"

\- "Summarize recent news on autonomous delivery robots for last-mile delivery, plus any academic research on the topic."



\## Notes



\- `GOOGLE\_API\_KEY` and `TAVILY\_API\_KEY` are required — the app will exit with a clear message if either is missing.

\- The Gemini model string in `main.py` may need updating over time as Google deprecates older model versions — check the error message if you get a `404 NOT\_FOUND`, it names the current replacement model.

\- This is a hackathon prototype: no auth, no rate limiting, no persistence beyond in-memory conversation state (`MemorySaver`). Not intended for production deployment as-is.

