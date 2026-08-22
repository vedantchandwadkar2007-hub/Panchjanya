"""
FastAPI backend for the multi-agent Research & Competitor Tracking system.

Streams, over Server-Sent Events:
  - the Supervisor's routing decision
  - each specialist agent's own ACT/OBSERVE trace (tagged with agent name)
  - the Supervisor's final synthesized briefing

Run:
    pip install -r requirements.txt
    (fill .env with GOOGLE_API_KEY and TAVILY_API_KEY)
    uvicorn server:app --reload --port 8000
"""

import json

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from main import (
    check_env,
    get_llm,
    route_query,
    synthesize,
    build_competitor_agent,
    build_research_agent,
    extract_text,
)

check_env()

app = FastAPI(title="Research & Competitor Tracking — Multi-Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


def sse(event_type: str, payload: dict) -> str:
    return f"data: {json.dumps({'type': event_type, **payload})}\n\n"


def run_specialist_streaming(agent, query: str, agent_name: str):
    """Yields SSE events for one specialist's ReAct loop; returns its final text."""
    final_text = ""
    for step in agent.stream({"messages": [HumanMessage(content=query)]}, stream_mode="values"):
        last_msg = step["messages"][-1]

        if getattr(last_msg, "tool_calls", None):
            for tc in last_msg.tool_calls:
                yield sse("act", {"agent": agent_name, "tool": tc["name"], "args": tc["args"]})

        elif last_msg.type == "tool":
            yield sse("observe", {
                "agent": agent_name,
                "tool": last_msg.name,
                "content": str(last_msg.content)[:800],
            })

        elif last_msg.type == "ai" and last_msg.content:
            final_text = extract_text(last_msg.content)

    return final_text


def stream_multi_agent_response(query: str):
    llm = get_llm()

    yield sse("status", {"message": "Supervisor deciding routing..."})

    try:
        decision = route_query(llm, query)
        yield sse("supervisor_route", {
            "reasoning": decision.reasoning,
            "need_competitor_agent": decision.need_competitor_agent,
            "need_research_agent": decision.need_research_agent,
        })

        findings = {}

        if decision.need_competitor_agent:
            yield sse("agent_start", {"agent": "CompetitorIntelAgent"})
            agent = build_competitor_agent()
            gen = run_specialist_streaming(agent, query, "CompetitorIntelAgent")
            final_text = ""
            try:
                while True:
                    event = next(gen)
                    yield event
            except StopIteration as stop:
                final_text = stop.value or ""
            findings["CompetitorIntelAgent"] = final_text
            yield sse("agent_done", {"agent": "CompetitorIntelAgent"})

        if decision.need_research_agent:
            yield sse("agent_start", {"agent": "ResearchTrendsAgent"})
            agent = build_research_agent()
            gen = run_specialist_streaming(agent, query, "ResearchTrendsAgent")
            final_text = ""
            try:
                while True:
                    event = next(gen)
                    yield event
            except StopIteration as stop:
                final_text = stop.value or ""
            findings["ResearchTrendsAgent"] = final_text
            yield sse("agent_done", {"agent": "ResearchTrendsAgent"})

        yield sse("status", {"message": "Supervisor synthesizing final briefing..."})
        final_briefing = synthesize(llm, query, findings)
        yield sse("final", {"content": final_briefing})

    except Exception as e:
        yield sse("error", {"message": str(e)})

    yield sse("done", {})


@app.post("/api/query")
def query(req: QueryRequest):
    return StreamingResponse(
        stream_multi_agent_response(req.query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/health")
def health():
    return {"status": "ok"}