from pydantic import BaseModel, Field
from typing import Optional, Annotated, List


class ChatRequest(BaseModel):
    prompt: Annotated[str, Field(description="User input")]


class ChatResponse(BaseModel):
    status: Annotated[str, Field(description="Response status")]
    model: Annotated[str, Field(description="Model name")]
    reasoning: Annotated[Optional[str], Field(description="The model's reasoning process")]
    response: Annotated[str, Field(description="The model's response")]
    citations: Annotated[Optional[List[str]], Field(default=None, description="List of sources cited in the response using web search")]
