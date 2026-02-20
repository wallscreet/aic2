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
