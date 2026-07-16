from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from backend.ai.nutrition_chat import NutritionChatService

router = APIRouter(tags=["chat"])

_chat_service = NutritionChatService()


class ChatRequest(BaseModel):
    message: str
    product_report: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    success: bool
    error: Optional[str] = None


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    result = _chat_service.ask(
        question=req.message,
        product_report=req.product_report,
        session_id=req.session_id,
    )
    return ChatResponse(
        session_id=result.session_id,
        answer=result.answer,
        success=result.success,
        error=result.error,
    )
