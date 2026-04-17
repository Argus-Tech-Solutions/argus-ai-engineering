from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Argus LLM API")
generator = pipeline("text-generation", model="gpt2")

class Request(BaseModel):
    prompt: str
    max_tokens: int = 128

@app.post("/v1/chat")
def chat(req: Request):
    out = generator(req.prompt, max_new_tokens=req.max_tokens)
    return {"response": out[0]["generated_text"], "model": "argus-llm-v1"}
