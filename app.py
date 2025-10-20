# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from utils import chatbot  # singleton instance

app = FastAPI(title="Llama 3-8B (modular + metrics)")

class GenerateIn(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9


@app.on_event("startup")
def load_model():
    chatbot.load()


@app.get("/health")
def health():
    return chatbot.health()


@app.post("/generate")
def generate(x: GenerateIn):
    if not chatbot.is_loaded():
        raise HTTPException(status_code=503, detail=f"Model not loaded: {chatbot.health().get('load_error')}")
    try:
        result = chatbot.generate(
            prompt=x.prompt,
            max_new_tokens=x.max_new_tokens,
            temperature=x.temperature,
            top_p=x.top_p,
        )
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

