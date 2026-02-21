from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException 
from fastapi.responses import StreamingResponse 
from schemas import ChatRequest, ChatResponse
from ollama import chat


router = APIRouter(prefix="/ollama", tags=["ollama"])


@router.post("/generate", response_model=ChatResponse)
async def generate_non_thinking(request: ChatRequest, model_name: str= "granite4:3b-h"):
    messages = [
        {
            "role": "user",
            "content": request.prompt,
        },
    ]

    try:
        response = chat(model=model_name, messages=messages)

        return {
            "status": "success",
            "model": model_name,
            "reasoning": None,
            "response": response['message']['content'],
            "citations": None 
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate_thinking", response_model=ChatResponse)
async def generate_with_thinking(request: ChatRequest, model_name: str="qwen3:0.6b"):
    messages = [
            {
                "role": "user",
                "content": request.prompt,
            },
    ]

    try:
        response = chat(model=model_name, messages=messages, think=True)

        return {
            "status": "success",
            "model": model_name,
            "reasoning": response.message.thinking,
            "response": response.message.content,
            "citations": None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def stream_sse(request: ChatRequest, model_name: str="granite4:3b-h"):
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            messages = [
                    {
                        "role": "user",
                        "content": request.prompt
                    },
            ]

            for part in chat(model=model_name, messages=messages, stream=True):
                content = part.get('message', {}).get('content', '')

                if content:
                    yield f"data: {content}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: [ERROR]: {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/stream_thinking")
async def stream_with_thinking(request: ChatRequest, model_name: str="qwen3:0.6b"):
    async def event_generator() -> AsyncGenerator[str, None]:
        messages = [
            {
                "role": "user",
                "content": request.prompt
            }
        ]

        try:
            stream = chat(model=model_name, messages=messages, stream=True, options={'think': True})

            for chunk in stream:
                thinking = chunk.get('message', {}).get('thinking')
                content = chunk.get('message', {}).get('content')

                if thinking:
                    yield f"event: thought\ndata: {thinking}\n\ndata"

                if content:
                    yield f"event: message\ndata: {content}\n\n"

            yield "event: control\ndata: [DONE]\n\n"

        except Exception as e:
            yield f"event: [ERROR]\ndata: {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
