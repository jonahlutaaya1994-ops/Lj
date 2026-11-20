
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

# -------------------------------
# CONFIGURATION
# -------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LJ-AI")

PACKET_API_KEY = os.getenv("PACKET_API_KEY")
PACKET_BASE_URL = os.getenv(
    "PACKET_BASE_URL",
    "https://api.packet.ai/v1"
)
DEFAULT_MODEL = os.getenv(
    "PACKET_MODEL",
    "Qwen/Qwen2.5-1.5B-Instruct"
)

app = FastAPI(
    title="LJ AI Backend",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# DATA MODELS
# -------------------------------

class Message(BaseModel):
    role: str = Field(
        pattern="^(system|user|assistant)$"
    )
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str | None = None
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=512, ge=1, le=4096)

# -------------------------------
# HEALTH CHECK
# -------------------------------

@app.get("/")
def home():
    return {
        "name": "LJ AI Backend",
        "status": "online",
        "version": "2.0.0"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "api_key_configured": bool(PACKET_API_KEY),
        "model": DEFAULT_MODEL,
        "provider": "Packet AI"
    }

# -------------------------------
# MODEL INFORMATION
# -------------------------------

@app.get("/models")
def models():
    return {
        "default_model": DEFAULT_MODEL,
        "available_models": [
            "Qwen/Qwen2.5-1.5B-Instruct",
            "nvidia/DeepSeek-V4.1-Flash-NVFP4",
            "MiniMaxAI/MiniMax-M3.1-Flash"
        ]
    }

# -------------------------------
# CHAT ENGINE
# -------------------------------

@app.post("/chat")
def chat(request: ChatRequest):

    if not PACKET_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="PACKET_API_KEY is not configured on Render."
        )

    if not request.messages:
        raise HTTPException(
            status_code=400,
            detail="Please provide at least one message."
        )

    try:
        client = OpenAI(
            api_key=PACKET_API_KEY,
            base_url=PACKET_BASE_URL,
            timeout=60.0,
            max_retries=1
        )

        response = client.chat.completions.create(
            model=request.model or DEFAULT_MODEL,
            messages=[
                message.model_dump()
                for message in request.messages
            ],
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        reply = response.choices[0].message.content or ""

        logger.info(
            "Chat successful. Model: %s",
            response.model
        )

        return {
            "success": True,
            "reply": reply,
            "model": response.model
        }

    except Exception as error:
        # Do not log the API key or request headers.
        logger.exception("Packet AI request failed")

        raise HTTPException(
            status_code=502,
            detail=f"Packet AI error: {str(error)[:1000]}"
        )

# -------------------------------
# SERVER STATUS
# -------------------------------

@app.get("/status")
def status():
    return {
        "server": "LJ-2",
        "backend": "online",
        "packet_ai_key": "configured" if PACKET_API_KEY else "missing",
        "model": DEFAULT_MODEL
    }
