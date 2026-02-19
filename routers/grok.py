import os
from typing import AsyncGenerator
from fastapi.responses import StreamingResponse
from xai_sdk import Client
from xai_sdk.chat import user
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from schemas import ChatRequest

load_dotenv()


router = APIRouter(prefix="/xai", tags=["xai"])


@router.post("/generate")
async def generate_non_thinking(request: ChatRequest, model_name: str="grok-4-1-fast-non-reasoning"):
    client = Client(api_key=os.getenv("XAI_API_KEY"))

    try:
        chat = client.chat.create(model=model_name)
        chat.append(user(request.prompt))
        response = chat.sample()

        return {
            "status": "success",
            "response": response.content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate_thinking")
async def generate_thinking(request: ChatRequest, model_name: str="grok-4-1-fast-reasoning"):
    client = Client(api_key=os.getenv("XAI_API_KEY"))

    try:
        chat = client.chat.create(model=model_name)
        chat.append(user(request.prompt))
        response = chat.sample()

        return {
            "status": "success",
            "response": response.content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def stream_sse(request: ChatRequest, model_name: str="grok-4-1-fast-non-reasoning"):
    client = Client(api_key=os.getenv("XAI_API_KEY"))

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            chat = client.chat.create(model=model_name)
            chat.append(user(request.promp))

            for _, chunk in chat.stream():
                if chunk.content:
                    yield f"data: {chunk.content}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
