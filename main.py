"""
Autonomous Research & Competitor Tracking Agent — Multi-Agent Architecture
---------------------------------------------------------------------------
Theme: Research & Competitor Tracking

Architecture:
    SUPERVISOR (Gemini) decides, per query, which specialist agent(s) are
    needed, then orchestrates them and synthesizes their outputs into one
    final briefing.

    Specialist 1 — CompetitorIntelAgent
        Responsibility: live news, competitor moves, funding, product
        launches, market signals. Own ReAct loop. Own tool: web_search
        (Tavily).

    Specialist 2 — ResearchTrendsAgent
        Responsibility: academic / technical research trends. Own ReAct
        loop. Own tool: research_search (arXiv).

    Each specialist independently runs a full Reason -> Act -> Observe loop
    with its own tool before returning a scoped answer to the Supervisor.
    The Supervisor then performs the final synthesis step, combining both
    specialists' findings — this is the "meaningful collaboration between
    agents" layer, not just two tools bolted onto one agent.

Run:
    pip install -r requirements.txt
    (fill in .env with GOOGLE_API_KEY and TAVILY_API_KEY)
    python main.py
"""

import os
import sys
from typing import List

from dotenv import load_dotenv
load_dotenv()  # reads .env file in the current directory, if present

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
import arxiv
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. Environment / key checks
# ---------------------------------------------------------------------------
def check_env() -> None:
    missing = [
        k for k in ("GOOGLE_API_KEY", "TAVILY_API_KEY") if not os.environ.get(k)
    ]
    if missing:
        print(f"[ERROR] Missing environment variables: {', '.join(missing)}")
        print("Set them in a .env file (see .env.example), e.g.:")
        print('  GOOGLE_API_KEY="your-gemini-key"')
        print('  TAVILY_API_KEY="your-tavily-key"')
        sys.exit(1)


def get_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=temperature)


# ---------------------------------------------------------------------------
# 2. Specialist 1 — CompetitorIntelAgent
#    Responsibility: news, competitor moves, funding, launches, market signals
# ---------------------------------------------------------------------------
COMPETITOR_AGENT_PROMPT = """You are the Competitor Intelligence specialist.

Your ONLY responsibility: find and summarize live, time-sensitive information —
competitor announcements, product launches, funding rounds, pricing changes,
market moves, industry press. You do NOT cover academic research; that is a
different specialist's job.

Use the `web_search` tool as many times as needed, refining your query each
round, until you have enough to answer. Then return a concise, factual
summary of what you found (not a full report — the Supervisor will combine
your findings with another specialist's).
"""


def make_web_search_tool() -> TavilySearch:
    return TavilySearch(
        max_results=5,
        name="web_search",
        description=(
            "Search the live web for recent news, competitor announcements, "
            "product launches, funding rounds, blog posts, and industry press. "
            "Input should be a focused search query string."
        ),
    )


def build_competitor_agent():
    return create_react_agent(
        model=get_llm(),
        tools=[make_web_search_tool()],
        prompt=COMPETITOR_AGENT_PROMPT,
    )


# ---------------------------------------------------------------------------
# 3. Specialist 2 — ResearchTrendsAgent
#    Responsibility: academic papers / research direction
# ---------------------------------------------------------------------------
RESEARCH_AGENT_PROMPT = """You are the Research Trends specialist.

Your ONLY responsibility: find and summarize academic/technical research —
papers, methods, algorithms, state-of-the-art techniques. You do NOT cover
live news, funding, or company announcements; that is a different
specialist's job.

Use the `research_search` tool as many times as needed, refining your query
each round, until you have enough to answer. Then return a concise, factual
summary of what you found (not a full report — the Supervisor will combine
your findings with another specialist's).
"""


def make_research_search_tool():
    @tool
    def research_search(query: str) -> str:
        """Search arXiv for academic papers and research trends on a technical
        topic. Input should be a focused topic/keyword string."""
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        entries = []
        for r in client.results(search):
            authors = ", ".join(a.name for a in r.authors[:4])
            summary = r.summary.replace("\n", " ").strip()
            if len(summary) > 400:
                summary = summary[:400] + "..."
            entries.append(
                f"Title: {r.title}\n"
                f"Authors: {authors}\n"
                f"Published: {r.published.date()}\n"
                f"Summary: {summary}\n"
                f"URL: {r.entry_id}"
            )
        if not entries:
            return "No arXiv results found for this query."
        return "\n\n---\n\n".join(entries)

    return research_search


def build_research_agent():
    return create_react_agent(
        model=get_llm(),
        tools=[make_research_search_tool()],
        prompt=RESEARCH_AGENT_PROMPT,
    )


# ---------------------------------------------------------------------------
# 4. Supervisor — decides routing, then synthesizes
# ---------------------------------------------------------------------------
class RoutingDecision(BaseModel):
    need_competitor_agent: bool = Field(
        description="True if the query needs live news/competitor/market intel"
    )
    need_research_agent: bool = Field(
        description="True if the query needs academic/technical research trends"
    )
    reasoning: str = Field(description="One sentence explaining the routing choice")


SUPERVISOR_ROUTING_PROMPT = """You are the Supervisor of a multi-agent research
system. Given the user's query, decide which specialist agent(s) are needed:

- competitor_agent: live news, competitor moves, funding, launches, market signals
- research_agent: academic papers, technical research trends

A query may need one or both. Route to both only when the query genuinely
spans both domains.
"""

SUPERVISOR_SYNTHESIS_PROMPT = """You are the Supervisor of a multi-agent research
system. You dispatched specialist agent(s) and received their findings below.
Synthesize them into ONE final, decision-ready briefing structured as:

1. Summary (2-3 sentences)
2. Key Findings (bullet points; note which specialist each finding came from)
3. Suggested Next Actions (2-3 bullets)

Be factual. If a specialist's findings are thin or conflict, say so explicitly.
"""


def route_query(llm: ChatGoogleGenerativeAI, query: str) -> RoutingDecision:
    router = llm.with_structured_output(RoutingDecision)
    result = router.invoke([
        SystemMessage(content=SUPERVISOR_ROUTING_PROMPT),
        HumanMessage(content=query),
    ])
    return result


def extract_text(content) -> str:
    """Gemini sometimes returns content as a list of parts (text/dict blocks,
    occasionally including non-text blocks like thought signatures) instead
    of a plain string. This flattens it down to clean text."""
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "\n".join(p for p in parts if p)
    return content


def synthesize(llm: ChatGoogleGenerativeAI, query: str, findings: dict) -> str:
    findings_block = "\n\n".join(
        f"--- {name} findings ---\n{text}" for name, text in findings.items()
    )
    result = llm.invoke([
        SystemMessage(content=SUPERVISOR_SYNTHESIS_PROMPT),
        HumanMessage(content=f"Original query: {query}\n\n{findings_block}"),
    ])
    return extract_text(result.content)


# ---------------------------------------------------------------------------
# 5. Orchestration — run the full multi-agent flow, printing each step
# ---------------------------------------------------------------------------
def run_query(query: str) -> str:
    llm = get_llm()
    print(f"\n{'='*70}\nUSER QUERY: {query}\n{'='*70}")

    # --- Supervisor: routing decision ---
    decision = route_query(llm, query)
    print(f"\n[SUPERVISOR] Routing decision: {decision.reasoning}")
    print(f"  -> competitor_agent: {decision.need_competitor_agent}")
    print(f"  -> research_agent:   {decision.need_research_agent}")

    findings = {}

    # --- Specialist 1 ---
    if decision.need_competitor_agent:
        print("\n[AGENT: CompetitorIntelAgent] starting...")
        agent = build_competitor_agent()
        for step in agent.stream({"messages": [HumanMessage(content=query)]}, stream_mode="values"):
            last_msg = step["messages"][-1]
            if getattr(last_msg, "tool_calls", None):
                for tc in last_msg.tool_calls:
                    print(f"  [ACT] {tc['name']} | args: {tc['args']}")
            elif last_msg.type == "tool":
                print(f"  [OBSERVE] -> {str(last_msg.content)[:200]}...")
            elif last_msg.type == "ai" and last_msg.content:
                findings["CompetitorIntelAgent"] = extract_text(last_msg.content)
        print("[AGENT: CompetitorIntelAgent] done.")

    # --- Specialist 2 ---
    if decision.need_research_agent:
        print("\n[AGENT: ResearchTrendsAgent] starting...")
        agent = build_research_agent()
        for step in agent.stream({"messages": [HumanMessage(content=query)]}, stream_mode="values"):
            last_msg = step["messages"][-1]
            if getattr(last_msg, "tool_calls", None):
                for tc in last_msg.tool_calls:
                    print(f"  [ACT] {tc['name']} | args: {tc['args']}")
            elif last_msg.type == "tool":
                print(f"  [OBSERVE] -> {str(last_msg.content)[:200]}...")
            elif last_msg.type == "ai" and last_msg.content:
                findings["ResearchTrendsAgent"] = extract_text(last_msg.content)
        print("[AGENT: ResearchTrendsAgent] done.")

    # --- Supervisor: synthesis ---
    print("\n[SUPERVISOR] Synthesizing final briefing...")
    final = synthesize(llm, query, findings)
    print(f"\n[FINAL BRIEFING]\n{final}")
    return final


# ---------------------------------------------------------------------------
# 6. Test loop / entry point
# ---------------------------------------------------------------------------
SAMPLE_QUERIES: List[str] = [
    "Track the latest competitor moves and funding news for OpenAI vs Anthropic this month.",
    "What's the current research trend in on-device LLM inference for mobile phones?",
    "Summarize recent news on autonomous delivery robots for last-mile delivery, plus any relevant academic research on the topic.",
]


def main():
    check_env()
    print("Research & Competitor Tracking — Multi-Agent System (Supervisor + 2 specialists)")
    print("Type a query, or press Enter to run built-in sample queries. 'exit' to quit.\n")

    user_in = input("Query (blank = run samples): ").strip()
    if user_in.lower() == "exit":
        return

    if user_in:
        run_query(user_in)
    else:
        for q in SAMPLE_QUERIES:
            run_query(q)

    while True:
        follow_up = input("\nAsk another question ('exit' to quit): ").strip()
        if follow_up.lower() in ("exit", "quit", ""):
            break
        run_query(follow_up)


if __name__ == "__main__":
    main()