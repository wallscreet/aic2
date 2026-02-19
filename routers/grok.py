import os
from typing import AsyncGenerator
from fastapi.responses import StreamingResponse
from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.tools import web_search
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from schemas import ChatRequest, ChatResponse

load_dotenv()


router = APIRouter(prefix="/xai", tags=["xai"])


@router.post("/generate", response_model=ChatResponse)
async def generate_non_thinking(request: ChatRequest, model_name: str="grok-4-1-fast-non-reasoning"):
    client = Client(api_key=os.getenv("XAI_API_KEY"))

    try:
        chat = client.chat.create(model=model_name)
        chat.append(user(request.prompt))
        response = chat.sample()

        return {
            "status": "success",
            "model": model_name,
            "reasoning": None,
            "response": response.content,
            "citations": None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate_with_search", response_model=ChatResponse)
async def generate_with_search(request: ChatRequest, model_name: str="grok-4-1-fast-non-reasoning"):
    client = Client(api_key=os.getenv("XAI_API_KEY"))
    
    try:
        chat = client.chat.create(model=model_name, tools=[web_search()])
        
        chat.append(user(request.prompt))
        response = chat.sample()
        citations_list = response.citations if response.citations else None

        return {
            "status": "success",
            "model": model_name,
            "reasoning": None,
            "response": response.content,
            "citations": citations_list
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate_thinking", response_model=ChatResponse)
async def generate_thinking(request: ChatRequest, model_name: str="grok-4-1-fast-reasoning"):
    client = Client(api_key=os.getenv("XAI_API_KEY"))

    try:
        chat = client.chat.create(model=model_name)
        chat.append(user(request.prompt))
        response = chat.sample()

        return {
            "status": "success",
            "model": model_name,
            "reasoning": None,
            "response": response.content,
            "citations": None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def stream_sse(request: ChatRequest, model_name: str="grok-4-1-fast-non-reasoning"):
    client = Client(api_key=os.getenv("XAI_API_KEY"))

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            chat = client.chat.create(model=model_name)
            chat.append(user(request.prompt))

            for _, chunk in chat.stream():
                if chunk.content:
                    yield f"data: {chunk.content}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


