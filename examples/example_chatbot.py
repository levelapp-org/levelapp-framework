from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Tiny Chatbot")

client = OpenAI(api_key="<YOUR-API-KEY-HERE>")

SYSTEM_PROMPT = "Play role as a medical assistant."


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


def generate_reply(user_message: str) -> str:
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # pick any chat-capable model you have access to
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"LLM error: {e}")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message:
        raise HTTPException(status_code=400, detail="`message` is required.")
    try:
        reply = generate_reply(req.message)
        return ChatResponse(reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/healthz")
def health():
    return {"status": "ok"}
