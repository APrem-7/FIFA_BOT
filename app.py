# app.py - FastAPI Backend for FIFA Bot

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from main_func import ask_fifa_bot

app = FastAPI(title="FIFA World Cup Chatbot API")

# Allow your frontend to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # in dev: allow everything
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"  # optional: for conversation history

class ChatResponse(BaseModel):
    reply: str
    thread_id: str

@app.get("/")
def read_root():
    return {"message": "FIFA World Cup Chatbot API", "endpoint": "/chat"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Chat endpoint - receive message from frontend and return bot response
    
    Usage:
    POST /chat
    {
        "message": "Tell me about Portugal vs Morocco",
        "thread_id": "user123"
    }
    """
    try:
        # Get response from bot
        reply = ask_fifa_bot(req.message, thread_id=req.thread_id)
        return ChatResponse(reply=reply, thread_id=req.thread_id)
    except Exception as e:
        return ChatResponse(
            reply=f"Error: {str(e)}", 
            thread_id=req.thread_id
        )

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

