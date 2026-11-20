
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

app = FastAPI(title="LJ AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PACKET_API_KEY = os.getenv("PACKET_API_KEY")
PACKET_BASE_URL = "https://api.packet.ai/v1"
DEFAULT_MODEL = os.getenv(
    "PACKET_MODEL",
    "Qwen/Qwen2.5-1.5B-Instruct"
)


class Message(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int = 512


@app.get("/")
def home():
    return {
        "name": "LJ AI Backend",
        "status": "online"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "api_key_configured": bool(PACKET_API_KEY)
    }


@app.post("/chat")
def chat(request: ChatRequest):
    if not PACKET_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Packet AI API key is missing"
        )

    try:
        client = OpenAI(
            api_key=PACKET_API_KEY,
            base_url=PACKET_BASE_URL
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

        return {
            "reply": response.choices[0].message.content or "",
            "model": response.model
        }

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Packet AI request failed"
        )
