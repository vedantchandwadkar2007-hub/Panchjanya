"""
FastAPI backend for the Research & Competitor Tracking agent.

Exposes:
  POST /api/query   -> Server-Sent Events stream of the agent's
                        Reason -> Act -> Observe -> Final Answer loop.
  GET  /api/health   -> simple health check

Run:
    pip install -r requirements.txt
    export GOOGLE_API_KEY="..."
    export TAVILY_API_KEY="..."
    uvicorn server:app --reload --port 8000

Then open frontend/index.html (it points at http://localhost:8000).
"""

import json
import uuid

from dotenv import load_dotenv
load_dotenv()  # reads .env file in the current directory, if present

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage

from main import build_agent, SYSTEM_PROMPT, check_env

check_env()
agent = build_agent()

app = FastAPI(title="Research & Competitor Tracking Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this for real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    thread_id: str | None = None


def sse(event_type: str, payload: dict) -> str:
    return f"data: {json.dumps({'type': event_type, **payload})}\n\n"


def stream_agent_response(query: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=query)]

    yield sse("status", {"message": "Agent started"})

    try:
        for step in agent.stream({"messages": messages}, config=config, stream_mode="values"):
            last_msg = step["messages"][-1]

            if getattr(last_msg, "tool_calls", None):
                for tc in last_msg.tool_calls:
                    yield sse("act", {"tool": tc["name"], "args": tc["args"]})

            elif last_msg.type == "tool":
                content = str(last_msg.content)
                yield sse("observe", {
                    "tool": last_msg.name,
                    "content": content[:800],
                })

            elif last_msg.type == "ai" and last_msg.content:
                content = last_msg.content
                if isinstance(content, list):
                    # Gemini can return content as a list of parts (text/dict blocks)
                    text_parts = []
                    for part in content:
                        if isinstance(part, str):
                            text_parts.append(part)
                        elif isinstance(part, dict):
                            text_parts.append(part.get("text", ""))
                    content = "\n".join(p for p in text_parts if p)
                yield sse("final", {"content": content})

    except Exception as e:
        yield sse("error", {"message": str(e)})

    yield sse("done", {})


@app.post("/api/query")
def query(req: QueryRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    return StreamingResponse(
        stream_agent_response(req.query, thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/health")
def health():
    return {"status": "ok"}