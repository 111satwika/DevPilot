"""DevPilot's HTTP surface -- what the frontend talks to.

Two deliberately minimal endpoints: /health (fast, proves the server is up)
and /ask (slow, inherits the agent's real per-turn latency -- proves the
whole pipeline works identically over HTTP as it does in-process). Plain
synchronous blocking handler, no job queue -- see
DevPilot_AI_Implementation_Log.html Entry 20 for why that's the right call
here, same reasoning as Terminal/Docker's timeouts.

Entry 21: /ask now also returns the tool-call trace (not just the final
answer), and CORS is enabled -- scoped to the dev frontend's exact origin,
not a wildcard -- since this is the first time the API is reachable from a
browser.

Run from the project root: uvicorn backend.main:app --port 8001
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llm.agent import ask as agent_ask

app = FastAPI(title="DevPilot AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # the Vite dev server, nothing broader
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class AskRequest(BaseModel):
    message: str


class AskResponse(BaseModel):
    answer: str
    tool_calls: list[dict]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    result = await agent_ask(request.message)
    return AskResponse(answer=result.answer, tool_calls=result.tool_calls)
