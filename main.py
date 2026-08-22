import os
import sys
from typing import Dict, Any, TypedDict
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
import arxiv
from duckduckgo_search import DDGS

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# ---------------------------------------------------------------------------
# 1. Environment & Config
# ---------------------------------------------------------------------------
def check_env():
    if not os.environ.get("GROQ_API_KEY"):
        print("[ERROR] GROQ_API_KEY missing in .env")
        sys.exit(1)


def get_llm(model_name: str = "llama-3.1-8b-instant", temp: float = 0.1) -> ChatGroq:
    """Returns the Groq LLM. llama-3.1-8b-instant is extremely fast and token-efficient."""
    return ChatGroq(model=model_name, temperature=temp)


def extract_text(content) -> str:
    """Safely extracts string content from LLM outputs."""
    if isinstance(content, list):
        return "\n".join(str(p) for p in content if p)
    return str(content)


# ---------------------------------------------------------------------------
# 2. Resilient Tools (Task 5: Failure Recovery & Tool Fallback)
# ---------------------------------------------------------------------------
def robust_web_search(query: str) -> str:
    """Searches live web. Includes automatic Tavily -> DuckDuckGo fallback."""
    try:
        if os.environ.get("TAVILY_API_KEY"):
            from langchain_tavily import TavilySearch
            tavily = TavilySearch(max_results=2)
            res = tavily.invoke(query)
            if res:
                return str(res)[:1000] # Truncate to save Free Tier Tokens
    except Exception as e:
        print(f"  [TOOL WARNING] Tavily failed ({e}), falling back to DDGS...")
    
    # Autonomous Fallback if Tavily crashes or quota limits are hit
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=2))
            formatted = "\n".join([f"- {r['title']}: {r['body'][:200]}" for r in results])
            return formatted if formatted else "No web results found."
    except Exception as e:
        return f"All web search providers failed: {e}"


def robust_arxiv_search(query: str) -> str:
    """Searches arXiv. Includes error handling and token truncation."""
    try:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=2, sort_by=arxiv.SortCriterion.Relevance)
        entries = []
        for r in client.results(search):
            summary = r.summary.replace('\n', ' ')[:200]
            entries.append(f"Title: {r.title}\nSummary: {summary}...")
        return "\n\n".join(entries) if entries else "No academic papers found."
    except Exception as e:
        return f"arXiv search failed: {e}"


# ---------------------------------------------------------------------------
# 3. LangGraph State Schema (Task 4: Shared State)
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    query: str
    need_competitor: bool
    need_research: bool
    competitor_data: str
    research_data: str
    eval_passed: bool
    retry_count: int
    final_briefing: str


class RouteSchema(BaseModel):
    need_competitor: bool = Field(description="Set to true if query asks for market moves, news, or competitors.")
    need_research: bool = Field(description="Set to true if query asks for academic papers, science, or tech trends.")
    reasoning: str = Field(description="Brief reason for routing choice.")


# ---------------------------------------------------------------------------
# 4. LangGraph Nodes (Task 5: Adaptive Task Decomposition)
# ---------------------------------------------------------------------------
def supervisor_router_node(state: AgentState) -> Dict[str, Any]:
    print("\n[SUPERVISOR] Formulating dynamic routing plan...")
    llm = get_llm("llama-3.1-8b-instant")
    structured_llm = llm.with_structured_output(RouteSchema)
    prompt = f"Analyze this user query and decide the routing strategy: '{state['query']}'"
    
    try:
        decision = structured_llm.invoke([SystemMessage(content=prompt)])
        print(f"  -> Need Competitor Intel: {decision.need_competitor}")
        print(f"  -> Need Research Intel: {decision.need_research}")
        return {
            "need_competitor": decision.need_competitor,
            "need_research": decision.need_research,
            "retry_count": state.get("retry_count", 0)
        }
    except Exception:
        # Fallback if structure parsing fails
        return {"need_competitor": True, "need_research": True, "retry_count": 0}


def competitor_agent_node(state: AgentState) -> Dict[str, Any]:
    if not state.get("need_competitor", False):
        return {"competitor_data": "Not required by routing plan."}
    print("[AGENT: Competitor] Gathering market intelligence...")
    data = robust_web_search(state["query"])
    return {"competitor_data": data}


def research_agent_node(state: AgentState) -> Dict[str, Any]:
    if not state.get("need_research", False):
        return {"research_data": "Not required by routing plan."}
    print("[AGENT: Research] Gathering academic intelligence...")
    data = robust_arxiv_search(state["query"])
    return {"research_data": data}


def evaluator_node(state: AgentState) -> Dict[str, Any]:
    """Task 5: Self-Evaluation and Loop / Deadlock Detection."""
    print("[EVALUATOR] Checking data integrity & conflicts...")
    retries = state.get("retry_count", 0)
    
    # Loop safeguard: never retry more than once to avoid infinite deadlock
    if retries >= 1:
        print("  -> Max retries reached. Forcing synthesis to avoid deadlock.")
        return {"eval_passed": True, "retry_count": retries}
        
    comp_ok = bool(state.get("competitor_data") and "failed" not in state["competitor_data"].lower())
    res_ok = bool(state.get("research_data") and "failed" not in state["research_data"].lower())
    
    passed = comp_ok and res_ok
    if not passed:
        print("  -> Data incomplete. Triggering replan/retry.")
    else:
        print("  -> Data approved.")
        
    return {"eval_passed": passed, "retry_count": retries + 1}


def synthesis_node(state: AgentState) -> Dict[str, Any]:
    print("[SUPERVISOR] Synthesizing final briefing...")
    llm = get_llm("llama-3.1-8b-instant", temp=0.3)
    prompt = f"""
    You are the Lead Intelligence Strategist.
    Original Query: {state['query']}
    
    Competitor Intelligence:
    {state.get('competitor_data', 'N/A')}
    
    Academic & Research Intelligence:
    {state.get('research_data', 'N/A')}
    
    Write a final markdown briefing:
    1. Executive Summary
    2. Threat / Opportunity Rating (1-10)
    3. Actionable Next Steps
    """
    res = llm.invoke([SystemMessage(content=prompt)])
    return {"final_briefing": extract_text(res.content)}


# ---------------------------------------------------------------------------
# 5. Graph Assembly (Task 4 & 5: Parallel Exec & Checkpointing)
# ---------------------------------------------------------------------------
def build_intel_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("supervisor", supervisor_router_node)
    workflow.add_node("competitor_agent", competitor_agent_node)
    workflow.add_node("research_agent", research_agent_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("synthesizer", synthesis_node)
    
    workflow.set_entry_point("supervisor")
    
    # Parallel dispatch from supervisor to both agents
    workflow.add_edge("supervisor", "competitor_agent")
    workflow.add_edge("supervisor", "research_agent")
    
    # Join parallel paths into evaluation
    workflow.add_edge("competitor_agent", "evaluator")
    workflow.add_edge("research_agent", "evaluator")
    
    # Conditional Self-Correction Edge based on Evaluation
    def check_eval(state: AgentState):
        return "synthesizer" if state["eval_passed"] else "supervisor"
        
    workflow.add_conditional_edges(
        "evaluator",
        check_eval,
        {"synthesizer": "synthesizer", "supervisor": "supervisor"}
    )
    
    workflow.add_edge("synthesizer", END)
    
    # Task 4: MemorySaver for context persistence across turns
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 6. Execution Loop
# ---------------------------------------------------------------------------
graph_app = build_intel_graph()

def run_query(query: str, thread_id: str = "demo-session-1") -> str:
    print(f"\n{'='*70}\nUSER QUERY: {query}\n{'='*70}")
    
    # Thread ID required for MemorySaver to maintain state across turns
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {"query": query}
    
    # Stream the graph execution
    for _ in graph_app.stream(initial_state, config=config, stream_mode="updates"):
        pass # Nodes handle their own print statements
            
    final_state = graph_app.get_state(config).values
    final_briefing = final_state.get("final_briefing", "Error generating briefing.")
    
    print("\n[FINAL BRIEFING]\n")
    print(final_briefing)
    return final_briefing


# --- Stub Methods to Prevent server.py from Crashing ---
def route_query(*args, **kwargs):
    class MockDecision:
        need_competitor_agent = True
        need_research_agent = True
        reasoning = "LangGraph StateGraph takes over routing."
    return MockDecision()
def build_competitor_agent(*args, **kwargs): pass
def build_research_agent(*args, **kwargs): pass
def synthesize(*args, **kwargs): return "See terminal for LangGraph trace output."
def extract_text_mock(*args, **kwargs): return ""


def main():
    check_env()
    print("🚀 SIGNAL Agent Upgraded: LangGraph State Machine (Tasks 4 & 5)")
    print("Type a query, or press Enter to run a test. 'exit' to quit.\n")

    while True:
        user_in = input("\nQuery: ").strip()
        if user_in.lower() in ("exit", "quit"):
            break
        if not user_in:
            user_in = "Track the latest news on OpenAI robotics and on-device VLA models."
            
        run_query(user_in)

if __name__ == "__main__":
    main()