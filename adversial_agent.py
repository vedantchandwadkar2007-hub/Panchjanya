import os
import time
from typing import Dict, List, TypedDict, Annotated, Literal
import operator
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. SHARED STATE DEFINITION & CHECKPOINTING
# ==========================================
class AgentState(TypedDict):
    query: str
    plan: List[str]
    current_step: int
    tool_failures: int
    iteration_count: int
    specialist_results: Annotated[List[Dict[str, str]], operator.add]
    conflicts_detected: List[str]
    uncertainty_score: float  # 0.0 (certain) to 1.0 (highly uncertain)
    final_briefing: str
    status: str

# ==========================================
# 2. ADVERSARIAL TOOL IMPLEMENTATIONS & FALLBACKS
# ==========================================
def primary_market_tool(query: str, simulate_failure: bool = False) -> str:
    """Primary intelligence tool. Injects simulated outage under adversarial conditions."""
    if simulate_failure:
        raise ConnectionError("503 Service Unavailable: Primary Market API rate limit exceeded.")
    return f"[Primary Tool Result]: OpenAI raised $6.6B; Enterprise adoption up 40%."

def fallback_market_tool(query: str) -> str:
    """Resilient fallback tool executed when primary tool fails."""
    return f"[Fallback Tool Result]: Backchannel signals confirm OpenAI $6.6B round; valuation at $157B."

def conflicting_research_tool(query: str) -> str:
    """Tool returning conflicting benchmark data to test contradiction resolution."""
    return f"[Research Tool Result]: Benchmark study A shows 15% efficiency drop, while Study B reports 30% gain."

# ==========================================
# 3. GRAPH NODES (AGENTS & RESILIENCE LAYERS)
# ==========================================

def supervisor_planner(state: AgentState) -> Dict:
    """Decomposes the objective into an adaptive execution plan."""
    print("\n🔍 [Supervisor] Generating Adaptive Execution Plan...")
    query = state["query"]
    plan = [
        "collect_competitor_intel",
        "collect_research_trends",
        "evaluate_and_resolve_conflicts",
        "synthesize_briefing"
    ]
    return {
        "plan": plan,
        "current_step": 0,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "status": "PLANNING_COMPLETE"
    }

def competitor_agent(state: AgentState) -> Dict:
    """Executes market tracking with autonomous failure recovery and tool fallback."""
    print("🤖 [CompetitorIntelAgent] Gathering live signals...")
    failures = state.get("tool_failures", 0)
    
    # Adversarial Injection: Simulate a live API failure on first attempt
    try:
        if failures == 0:
            print("   ⚠️ [Adversarial Trigger] Primary Market Tool throwing 503 error...")
            raw_data = primary_market_tool(state["query"], simulate_failure=True)
        else:
            raw_data = primary_market_tool(state["query"], simulate_failure=False)
    except Exception as e:
        print(f"   🚨 Tool failure caught: {e}")
        print("   🔄 [Autonomous Recovery] Routing to Fallback Tool...")
        raw_data = fallback_market_tool(state["query"])
        failures += 1

    return {
        "specialist_results": [{"agent": "CompetitorIntelAgent", "data": raw_data}],
        "tool_failures": failures,
        "iteration_count": state["iteration_count"] + 1
    }

def research_agent(state: AgentState) -> Dict:
    """Gathers academic/technical benchmarks."""
    print("🔬 [ResearchTrendsAgent] Querying scientific literature...")
    data = conflicting_research_tool(state["query"])
    return {
        "specialist_results": [{"agent": "ResearchTrendsAgent", "data": data}],
        "iteration_count": state["iteration_count"] + 1
    }

def conflict_and_uncertainty_resolver(state: AgentState) -> Dict:
    """Evaluates contradictory findings and scores uncertainty."""
    print("⚖️ [Self-Evaluation Node] Auditing evidence for contradictions...")
    results = state.get("specialist_results", [])
    
    conflicts = []
    uncertainty = 0.1  # Base uncertainty

    # Detect conflicting claims
    all_text = " ".join([r["data"] for r in results])
    if "drop" in all_text and "gain" in all_text:
        conflicts.append("Contradiction detected: Benchmark efficiency variance between Study A (-15%) and Study B (+30%).")
        uncertainty += 0.4
        print(f"   ⚠️ Discrepancy identified: {conflicts[-1]}")
        print(f"   📊 Adjusted Uncertainty Score: {uncertainty:.2f}")

    return {
        "conflicts_detected": conflicts,
        "uncertainty_score": uncertainty,
        "iteration_count": state["iteration_count"] + 1
    }

def synthesizer_node(state: AgentState) -> Dict:
    """Synthesizes final briefing with attribution and resolved uncertainties."""
    print("📝 [Supervisor] Synthesizing Final Actionable Briefing...")
    
    findings = "\n".join([f"- **{r['agent']}**: {r['data']}" for r in state["specialist_results"]])
    conflict_notes = "\n".join([f"- ⚠️ {c}" for c in state["conflicts_detected"]]) if state["conflicts_detected"] else "None"
    
    briefing = f"""
============================================================
              ACTIONABLE INTELLIGENCE BRIEFING
============================================================
🎯 Objective: {state['query']}
🔄 Dynamic Replanning Cycles: {state['iteration_count']}
🛡️ Tool Failures Recovered: {state['tool_failures']}
📊 Decision Uncertainty Score: {state['uncertainty_score']:.2f}

KEY FINDINGS BY SPECIALIST:
{findings}

RESOLVED CONFLICTS & SENSITIVITY ANALYSIS:
{conflict_notes}

STRATEGIC RECOMMENDATION:
- Proceed with enterprise rollout given confirmed funding signals.
- Standardize on-device benchmarking internally to eliminate Study A/B efficiency discrepancies before production deployment.
============================================================
"""
    return {
        "final_briefing": briefing,
        "status": "COMPLETED"
    }

# ==========================================
# 4. CONDITIONAL ROUTING & DEADLOCK PREVENTION
# ==========================================
def router_logic(state: AgentState) -> Literal["competitor_agent", "research_agent", "conflict_resolver", "synthesizer", "deadlock_bail"]:
    """Conditional edge router implementing loop/deadlock protection."""
    # Deadlock guard: Max 10 iterations allowed
    if state.get("iteration_count", 0) > 10:
        return "deadlock_bail"
        
    results = state.get("specialist_results", [])
    agents_run = [r["agent"] for r in results]
    
    if "CompetitorIntelAgent" not in agents_run:
        return "competitor_agent"
    if "ResearchTrendsAgent" not in agents_run:
        return "research_agent"
    if "conflicts_detected" not in state:
        return "conflict_resolver"
    
    return "synthesizer"

def deadlock_recovery_node(state: AgentState) -> Dict:
    """Emergency bail-out node when budget or loop threshold is exceeded."""
    print("🚨 [Deadlock Detection] Maximum iteration threshold reached. Forcing synthesis.")
    return {"status": "DEADLOCK_ABORTED"}

# ==========================================
# 5. GRAPH COMPILATION (LangGraph Architecture)
# ==========================================
from langgraph.graph import StateGraph, END

def build_agent_graph():
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("supervisor_planner", supervisor_planner)
    workflow.add_node("competitor_agent", competitor_agent)
    workflow.add_node("research_agent", research_agent)
    workflow.add_node("conflict_resolver", conflict_and_uncertainty_resolver)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("deadlock_bail", deadlock_recovery_node)
    
    # Entry Point
    workflow.set_entry_point("supervisor_planner")
    
    # Edges
    workflow.add_conditional_edges(
        "supervisor_planner",
        router_logic,
        {
            "competitor_agent": "competitor_agent",
            "deadlock_bail": "deadlock_bail"
        }
    )
    
    workflow.add_conditional_edges(
        "competitor_agent",
        router_logic,
        {
            "research_agent": "research_agent",
            "deadlock_bail": "deadlock_bail"
        }
    )
    
    workflow.add_conditional_edges(
        "research_agent",
        router_logic,
        {
            "conflict_resolver": "conflict_resolver",
            "deadlock_bail": "deadlock_bail"
        }
    )
    
    workflow.add_conditional_edges(
        "conflict_resolver",
        router_logic,
        {
            "synthesizer": "synthesizer",
            "deadlock_bail": "deadlock_bail"
        }
    )
    
    workflow.add_edge("synthesizer", END)
    workflow.add_edge("deadlock_bail", END)
    
    return workflow.compile()

# ==========================================
# 6. ADVERSARIAL LIVE TEST RUNNER
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 INITIATING ADVERSARIAL MULTI-AGENT STRESS TEST")
    print("=" * 60)
    
    app = build_agent_graph()
    
    initial_state = {
        "query": "Track frontier model competitor moves and resolve edge-device inference benchmark discrepancies.",
        "plan": [],
        "current_step": 0,
        "tool_failures": 0,
        "iteration_count": 0,
        "specialist_results": [],
        "status": "STARTED"
    }
    
    final_output = app.invoke(initial_state)
    print(final_output.get("final_briefing", "Process aborted."))