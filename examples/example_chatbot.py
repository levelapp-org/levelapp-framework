import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="Tiny Chatbot")

client = OpenAI(api_key=openai_api_key)

SYSTEM_PROMPT = """
You are a medical assistant for a dental clinic, 
helping patients book appointments and answer inquiries about medical services.

## Behavior:
- Always reply in a convivial, professional tone.
- Be concise and clear.

## Instructions:
1. Identify the type of appointment the user requires based on their request.
2. If the user asks to book an appointment, return the booking information in a structured JSON format.
   - The JSON must include:
     - `reply_text`: A friendly confirmation message.
     - `metadata`: A dict containing the following info:
        1. `appointment_type`: One of "ROUTINE", "SURGICAL", or "RESTORATIVE".
        2. `appointment_date`: The date of the appointment (format: YYYY-MM-DD).
        3. `doctor_name`: One of "Dr. Tony Tony Chopper", "Dr. Trafalgar D. Water Law", or "Dr. Crocus".
   - Example JSON output:
     ```json
     {
       "reply_text": "Your ROUTINE appointment with Dr. Tony Tony Chopper is booked for 2025-12-01.",
       "appointment_type": "ROUTINE",
       "appointment_date": "2025-12-01",
       "doctor_name": "Dr. Tony Tony Chopper"
     }
     ```
3. If the user does not request a booking, return only the "reply_text".

## Additional Information:
- Dr. Tony Tony Chopper handles ROUTINE appointments.
- Dr. Trafalgar D. Water Law handles SURGICAL appointments.
- Dr. Crocus handles RESTORATIVE appointments.
"""



class ChatRequest(BaseModel):
    message: str

class Metadata(BaseModel):
    appointment_type: str = ""
    appointment_date: str = ""
    doctor_name: str = ""

class ChatResponse(BaseModel):
    reply_text: str
    metadata: Metadata = {}

def generate_reply(user_message: str) -> str:
    try:
      resp = client.chat.completions.parse(
          model="gpt-4o-mini",  # pick any chat-capable model you have access to
          messages=[
              {"role": "system", "content": SYSTEM_PROMPT},
              {"role": "user", "content": user_message},
          ],
          temperature=0.3,
          response_format=ChatResponse
      )
      return resp.choices[0].message.parsed
    except Exception as e:
      raise RuntimeError(f"LLM error: {e}")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message:
      raise HTTPException(status_code=400, detail="`message` is required.")
    try:
      reply = generate_reply(req.message)
      return reply
    except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))


@app.get("/healthz")
def health():
  return {"status": "ok"}
