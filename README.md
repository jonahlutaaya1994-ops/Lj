
# LJ AI Backend

Backend server for LJ AI.

## Technology
- Python
- FastAPI
- Packet AI
- Render

## Start server

pip install -r requirements.txt

uvicorn main:app --reload

## Environment variables

PACKET_API_KEY
PACKET_MODEL
PACKET_BASE_URL

Never upload your private API key to GitHub.

## Chat endpoint

POST /chat

## Health endpoint

GET /health
