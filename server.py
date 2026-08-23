"""
FastAPI backend for the LangGraph StateGraph multi-agent system.
Streams each node's trace events (act/observe/self_eval/final/etc.) to the
frontend over Server-Sent Events as they complete.

Run:
    pip install -r requirements.txt
    (fill .env with GROQ_API_KEY and TAVILY_API_KEY)
    uvicorn server:app --reload --port 8000
"""

import json
import uuid
import os

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from main import check_env, build_agent_graph

check_env()

app = FastAPI(title="Research & Competitor Tracking — LangGraph API")

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


def stream_graph_response(query: str):
    graph = build_agent_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
    initial_state = {
        "query": query, "iteration_count": 0, "specialist_results": [],
        "trace": [], "tool_failures": [], "conflicts_detected": [],
        "uncertainty_score": 0.0, "status": "STARTED",
    }

    yield sse("status", {"message": "Graph execution started"})

    try:
        seen_trace_count = 0
        seen_findings_count = 0
        for update in graph.stream(initial_state, config=config, stream_mode="values"):
            trace = update.get("trace", [])
            new_entries = trace[seen_trace_count:]
            seen_trace_count = len(trace)

            for entry in new_entries:
                etype = entry.get("type")
                if etype == "supervisor_route":
                    yield sse("supervisor_route", {
                        "reasoning": entry["reasoning"],
                        "need_competitor_agent": entry["need_competitor"],
                        "need_research_agent": entry["need_research"],
                    })
                elif etype == "agent_start":
                    yield sse("agent_start", {"agent": entry["agent"]})
                elif etype == "act":
                    yield sse("act", {"agent": entry["agent"], "tool": entry["tool"], "args": entry["args"]})
                elif etype == "observe":
                    yield sse("observe", {"agent": entry["agent"], "tool": entry["tool"], "content": entry["content"]})
                elif etype == "agent_done":
                    yield sse("agent_done", {"agent": entry["agent"]})
                elif etype == "self_eval":
                    yield sse("self_eval", {
                        "conflicts": entry["conflicts"],
                        "uncertainty_score": entry["uncertainty_score"],
                        "reasoning": entry["reasoning"],
                    })
                elif etype == "final":
                    yield sse("final", {"content": entry["content"]})
                elif etype == "error":
                    yield sse("error", {"message": entry["message"]})

            # emit "finding" events as soon as each specialist's result lands
            # (before the final briefing), so the document cards populate first
            results = update.get("specialist_results", [])
            new_findings = results[seen_findings_count:]
            seen_findings_count = len(results)
            for r in new_findings:
                yield sse("finding", {"agent": r["agent"], "content": r["data"]})

    except Exception as e:
        yield sse("error", {"message": str(e)})

    yield sse("done", {})


@app.post("/api/query")
def query(req: QueryRequest):
    return StreamingResponse(
        stream_graph_response(req.query),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/health")
def health():
    return {"status": "ok"}


# Serve the frontend as static files from this same app, so the whole thing
# deploys as one service (one URL for both UI and API — simplest for hosting).
if os.path.isdir("frontend"):
    @app.get("/")
    def serve_frontend():
        return FileResponse("frontend/index.html")

    app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")