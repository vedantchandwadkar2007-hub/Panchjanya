"""
Autonomous Research & Competitor Tracking Agent
------------------------------------------------
Theme: Research & Competitor Tracking
Pattern: ReAct (Reason -> Act -> Observe -> Final Answer) via LangGraph's
         prebuilt create_react_agent, powered by Google Gemini.

Tools:
  1. web_search      -> Tavily Search API (news, competitor moves, blogs, launches)
  2. research_search  -> arXiv API (academic papers / research trend tracking)

The agent dynamically decides which tool(s) to call based on the user's query,
observes tool output, loops as needed, and produces a final synthesized answer.

Run:
    pip install -r requirements.txt   (see pip install line below)
    export GOOGLE_API_KEY="..."
    export TAVILY_API_KEY="..."
    python main.py
"""

import os
import sys
from typing import List

from dotenv import load_dotenv
load_dotenv()  # reads .env file in the current directory, if present

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# 1. Environment / key checks
# ---------------------------------------------------------------------------
def check_env() -> None:
    missing = [
        k for k in ("GOOGLE_API_KEY", "TAVILY_API_KEY") if not os.environ.get(k)
    ]
    if missing:
        print(f"[ERROR] Missing environment variables: {', '.join(missing)}")
        print("Set them before running, e.g.:")
        print('  export GOOGLE_API_KEY="your-gemini-key"')
        print('  export TAVILY_API_KEY="your-tavily-key"')
        sys.exit(1)


# ---------------------------------------------------------------------------
# 2. Tool 1 -> Web / News search (competitor moves, launches, press, funding)
# ---------------------------------------------------------------------------
def make_web_search_tool() -> TavilySearch:
    return TavilySearch(
        max_results=5,
        name="web_search",
        description=(
            "Search the live web for recent news, competitor announcements, "
            "product launches, funding rounds, blog posts, and industry press. "
            "Use this for anything time-sensitive or company/market specific. "
            "Input should be a focused search query string."
        ),
    )


# ---------------------------------------------------------------------------
# 3. Tool 2 -> Academic / research search (papers, research trends)
# ---------------------------------------------------------------------------
def make_research_search_tool() -> ArxivQueryRun:
    wrapper = ArxivAPIWrapper(top_k_results=5, doc_content_chars_max=2000)
    tool = ArxivQueryRun(api_wrapper=wrapper)
    tool.name = "research_search"
    tool.description = (
        "Search arXiv for academic papers and research trends on a technical "
        "topic (e.g. new model architectures, algorithms, patents-adjacent "
        "research). Use this when the user asks about research direction, "
        "state-of-the-art techniques, or what's being published in a field. "
        "Input should be a focused topic/keyword string."
    )
    return tool


# ---------------------------------------------------------------------------
# 4. Build the ReAct agent
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an autonomous Research & Competitor Tracking analyst.

Your job: given a user request about a company, technology, or industry,
figure out what information is needed, call the right tool(s) to gather it,
and produce a concise, decision-ready briefing.

Rules:
- Use `web_search` for anything current: competitor news, product launches,
  funding, hiring signals, market moves, patents filed recently reported in press.
- Use `research_search` for academic/technical research trends (arXiv papers).
- You may call multiple tools, and call the same tool more than once with
  refined queries, before answering.
- Always reason about which tool fits before acting.
- When you have enough information, produce a Final Answer structured as:
    1. Summary (2-3 sentences)
    2. Key Findings (bullet points, with source names/links where available)
    3. Suggested Next Actions (2-3 bullets)
- Be factual. If sources conflict or info is thin, say so explicitly.
"""


def build_agent():
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0.2,
    )

    tools = [make_web_search_tool(), make_research_search_tool()]

    checkpointer = MemorySaver()  # lets the agent keep short-term context across turns

    agent = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=checkpointer,
    )
    return agent


# ---------------------------------------------------------------------------
# 5. Run helper — streams the Reason/Act/Observe loop to the terminal
# ---------------------------------------------------------------------------
def run_query(agent, query: str, thread_id: str = "session-1") -> str:
    config = {"configurable": {"thread_id": thread_id}}
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=query)]

    final_text = ""
    print(f"\n{'='*70}\nUSER QUERY: {query}\n{'='*70}")

    for step in agent.stream({"messages": messages}, config=config, stream_mode="values"):
        last_msg = step["messages"][-1]

        # Tool call requested by the LLM (the "Act" step)
        if getattr(last_msg, "tool_calls", None):
            for tc in last_msg.tool_calls:
                print(f"\n[ACT] Calling tool: {tc['name']}  |  args: {tc['args']}")

        # Tool result (the "Observe" step)
        elif last_msg.type == "tool":
            preview = str(last_msg.content)[:300].replace("\n", " ")
            print(f"[OBSERVE] ({last_msg.name}) -> {preview}...")

        # Final AI answer
        elif last_msg.type == "ai" and last_msg.content:
            final_text = last_msg.content
            print(f"\n[REASON/FINAL ANSWER]\n{final_text}")

    return final_text


# ---------------------------------------------------------------------------
# 6. Test loop / entry point
# ---------------------------------------------------------------------------
SAMPLE_QUERIES: List[str] = [
    "Track the latest competitor moves and funding news for OpenAI vs Anthropic this month.",
    "What's the current research trend in on-device LLM inference for mobile phones?",
    "Summarize recent news on autonomous delivery robots for campus/last-mile use, plus any relevant academic research on the topic.",
]


def main():
    check_env()
    agent = build_agent()

    print("Research & Competitor Tracking Agent — ReAct (Gemini + LangGraph)")
    print("Type a query, or press Enter to run the built-in sample queries. 'exit' to quit.\n")

    user_in = input("Query (blank = run samples): ").strip()

    if user_in.lower() == "exit":
        return

    if user_in:
        run_query(agent, user_in)
    else:
        for i, q in enumerate(SAMPLE_QUERIES, 1):
            run_query(agent, q, thread_id=f"sample-{i}")

    # Simple interactive continuation loop
    while True:
        follow_up = input("\nAsk another question ('exit' to quit): ").strip()
        if follow_up.lower() in ("exit", "quit", ""):
            break
        run_query(agent, follow_up)


if __name__ == "__main__":
    main()