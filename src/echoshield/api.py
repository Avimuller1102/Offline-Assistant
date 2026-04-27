from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from echoshield.engine import EchoShield, EchoConfig

app = FastAPI(
    title="EchoShield API",
    description="Offline-first, proof-carrying conversational assistant API.",
    version="0.1.0"
)

# Global shield instance
shield: Optional[EchoShield] = None

class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.7
    consensus: bool = True
    session_id: str = "default"

class ChatResponse(BaseModel):
    reply: str
    meta: Dict[str, Any]

@app.on_event("startup")
def load_model():
    """Load the model on startup if possible. In production you might lazy-load."""
    global shield
    print("Initializing EchoShield Model...")
    try:
        cfg = EchoConfig()
        shield = EchoShield(cfg)
        print("EchoShield Initialized Successfully.")
    except Exception as e:
        print(f"Warning: Could not initialize model on startup: {e}")

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": shield is not None}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    global shield
    if shield is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or unavailable offline.")
        
    if not shield.limiter.allow():
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")

    # Temporary configuration override for this request
    shield.cfg.temperature = request.temperature
    shield.cfg.consensus = request.consensus
    
    shield.memory.add("User", request.message)
    response_text, meta = shield._consensus(request.message)
    shield.memory.add("Bot", response_text)
    
    return ChatResponse(reply=response_text, meta=meta)
