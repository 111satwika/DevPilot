"""DevPilot's first HTTP surface -- what a future frontend would talk to.

Two deliberately minimal endpoints: /health (fast, proves the server is up)
and /ask (slow, inherits the agent's real per-turn latency -- proves the
whole pipeline works identically over HTTP as it does in-process). Plain
synchronous blocking handler, no job queue -- see
DevPilot_AI_Implementation_Log.html Entry 20 for why that's the right call
here, same reasoning as Terminal/Docker's timeouts.

Run from the project root: uvicorn backend.main:app --port 8000
"""

from fastapi import FastAPI
from pydantic import BaseModel

from llm.agent import ask as agent_ask

app = FastAPI(title="DevPilot AI")


class AskRequest(BaseModel):
    message: str


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    answer = await agent_ask(request.message)
    return AskResponse(answer=answer)
