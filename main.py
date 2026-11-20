
import os
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

# =====================================
# LJ AI CONFIGURATION
# =====================================

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
    description="Backend engine for LJ AI",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# =====================================
# PACKET AI CLIENT
# =====================================

def get_packet_client():

    if not PACKET_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="PACKET_API_KEY is missing from Render."
        )

    return OpenAI(
        api_key=PACKET_API_KEY,
        base_url=PACKET_BASE_URL,
        timeout=60.0,
        max_retries=1
    )


# =====================================
# REQUEST MODELS
# =====================================

class Message(BaseModel):

    role: str = Field(
        pattern="^(system|user|assistant)$"
    )

    content: str


class ChatRequest(BaseModel):

    messages: list[Message]

    model: str | None = None

    temperature: float = Field(
        default=0.7,
        ge=0,
        le=2
    )

    max_tokens: int = Field(
        default=512,
        ge=1,
        le=4096
    )


# =====================================
# HOME
# =====================================

@app.get("/")
def home():

    return {
        "name": "LJ AI Backend",
        "status": "online",
        "version": "3.0.0"
    }


# =====================================
# HEALTH CHECK
# =====================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "api_key_configured": bool(PACKET_API_KEY),
        "provider": "Packet AI",
        "default_model": DEFAULT_MODEL
    }


# =====================================
# SERVER STATUS
# =====================================

@app.get("/status")
def server_status():

    return {
        "server": "LJ-2",
        "backend": "online",
        "packet_api": "configured" if PACKET_API_KEY else "missing",
        "default_model": DEFAULT_MODEL
    }


# =====================================
# DISCOVER AVAILABLE MODELS
# =====================================

@app.get("/packet-models")
def packet_models():

    try:

        client = get_packet_client()

        result = client.models.list()

        available_models = []

        for model in result.data:

            available_models.append({
                "id": model.id,
                "owned_by": getattr(
                    model,
                    "owned_by",
                    None
                )
            })

        return {
            "success": True,
            "count": len(available_models),
            "models": available_models
        }

    except HTTPException:
        raise

    except Exception as error:

        logger.exception(
            "Packet AI model discovery failed"
        )

        raise HTTPException(
            status_code=502,
            detail={
                "message": "Could not retrieve Packet AI models",
                "error": str(error)[:1000]
            }
        )


# =====================================
# CHAT ENGINE
# =====================================

@app.post("/chat")
def chat(request: ChatRequest):

    if not request.messages:

        raise HTTPException(
            status_code=400,
            detail="No messages provided."
        )

    try:

        client = get_packet_client()

        selected_model = (
            request.model or DEFAULT_MODEL
        )

        response = client.chat.completions.create(

            model=selected_model,

            messages=[
                message.model_dump()
                for message in request.messages
            ],

            temperature=request.temperature,

            max_tokens=request.max_tokens
        )

        reply = (
            response.choices[0].message.content or ""
        )

        logger.info(
            "Chat successful using model: %s",
            response.model
        )

        return {
            "success": True,
            "reply": reply,
            "model": response.model
        }

    except HTTPException:
        raise

    except Exception as error:

        logger.exception(
            "Packet AI chat request failed"
        )

        raise HTTPException(
            status_code=502,
            detail={
                "message": "Packet AI request failed",
                "error": str(error)[:1000]
            }
        )


# =====================================
# SERVER INFORMATION
# =====================================

@app.get("/info")
def info():

    return {
        "application": "LJ AI",
        "backend_version": "3.0.0",
        "provider": "Packet AI",
        "default_model": DEFAULT_MODEL,
        "api_key_configured": bool(PACKET_API_KEY),
        "endpoints": [
            "/",
            "/health",
            "/status",
            "/info",
            "/packet-models",
            "/chat",
            "/docs"
        ]
    }
