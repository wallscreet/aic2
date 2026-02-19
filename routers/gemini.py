import os
from typing import AsyncGenerator
from google.genai import types
from google import genai
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException 
from fastapi.responses import StreamingResponse 
from schemas import ChatRequest, ChatResponse

load_dotenv()


router = APIRouter(prefix="/gemini", tags=["gemini"])


def extract_reasoning(response):
    """Parses the response parts to find the model's internal thoughts."""
    thoughts = []
    # Gemini responses are a list of 'candidates', each with 'parts'
    for part in response.candidates[0].content.parts:
        # In the 2026 SDK, parts have a 'thought' attribute
        if hasattr(part, 'thought') and part.thought:
            thoughts.append(part.text)
    
    return "\n".join(thoughts)


def get_thinking_config(model_name: str, level: str = "low"):
    """Maps user-friendly levels to model-specific parameters."""
    
    config_map = {
        "gemini-2.5-flash": {
            "low": {"thinking_budget": 1024},
            "medium": {"thinking_budget": 4096},
            "high": {"thinking_budget": 16384},
        },
        "gemini-3-flash": {
            "low": {"thinking_level": "low"},
            "medium": {"thinking_level": "medium"},
            "high": {"thinking_level": "high"},
        }
    }

    selected_params = config_map.get(model_name, config_map["gemini-3-flash"])[level]
    
    return types.ThinkingConfig(include_thoughts=True, **selected_params)


@router.post("/generate")
async def generate_non_thinking(request: ChatRequest):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=request.prompt
        )
        
        return {
            "status": "success",
            "response": response.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate_thinking", response_model=ChatResponse)
async def generate_with_thinking(request: ChatRequest, model_name: str="gemini-2.5-flash"):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    try:
        thinking_config = get_thinking_config(model_name=model_name, level="low")
        
        response = client.models.generate_content(
                model=model_name,
                contents=request.prompt,
                config=types.GenerateContentConfig(thinking_config=thinking_config))

        return {
                "status": "success",
                "model": model_name,
                "reasoning": extract_reasoning(response),
                "response": response.text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline Error: {str(e)}")


@router.post("/stream")
async def stream_sse(request: ChatRequest, model_name: str="gemini-2.5-flash"):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            response = client.models.generate_content_stream(
                    model=model_name,
                    contents=request.prompt
            )

            for chunk in response:
                if chunk.text:
                    # SSE Protocol: start with 'data: ' and end with '\n\n'
                    yield f"data: {chunk.text}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/stream_thinking")
async def stream_with_thinking(request: ChatRequest, model_name: str="gemini-2.5-flash"):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            thinking_config = get_thinking_config(model_name=model_name, level="low")
            response = client.models.generate_content_stream(
                    model=model_name,
                    contents=request.prompt,
                    config=types.GenerateContentConfig(thinking_config=thinking_config)
            )

            for chunk in response:
                candidate = chunk.candidates[0] if chunk.candidates else None
                
                # Check for safety block
                if candidate and candidate.finish_reason == "SAFETY":
                    yield "event: error\ndata: Safety filter triggered. Content blocked.\n\n"
                    break

                if not candidate or not candidate.content or not candidate.content.parts:
                    continue

                for part in candidate.content.parts:
                    if hasattr(part, 'thought') and part.thought:
                        # send to the 'thought' event channel
                        yield f"event: thought\ndata: {part.text}\n\n"
                    elif part.text:
                        # send to the 'message' event channel
                        yield f"event: message\ndata: {part.text}\n\n"

            yield "event: control\ndata: [DONE]\n\n"

        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
