# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chatbot_ipynb_to_pyhton import agent  # adjust if module name is different


# ---- Your existing function ----
def ask_fifa_bot(message: str, thread_id: str = "default") -> str:
    
    config = {"configurable": {"thread_id": thread_id}}
    
    result = agent.invoke({"messages": message}, config)
    
    return result["messages"][-1].content

# ---- FastAPI app ----
app = FastAPI()

# CORS so your frontend (different port/domain) can call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # in prod: put your frontend URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Request & Response models ----
class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = "default"


class ChatResponse(BaseModel):
    reply: str


# ---- Routes ----
@app.post("/chat", response_model=ChatResponse)
async def chat_with_fifa_bot(body: ChatRequest):
    reply = ask_fifa_bot(body.message, body.thread_id or "default")
    return ChatResponse(reply=reply)


@app.get("/")
async def root():
    return {"status": "ok", "message": "FIFA bot backend is running 🚀"}
