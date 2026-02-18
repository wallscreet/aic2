from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from google import genai
import os
from dotenv import load_dotenv


load_dotenv()

router = APIRouter(prefix="/gemini", tags=["gemini"])

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class ChatRequest(BaseModel):
    prompt: str


@router.post("/generate")
async def generate_non_thinking(request: ChatRequest):
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
