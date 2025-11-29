# main.py (replace entire file with this exact content)
import sys
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# ensure env vars are loaded before importing heavy modules
from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

# import worker AFTER load_dotenv to avoid Playwright/subprocess issues during import
from worker import handle_quiz_request

# application
APP_EMAIL = os.getenv("YOUR_EMAIL")
APP_SECRET = os.getenv("SECRET")

app = FastAPI(title="LLM Quiz Solver Endpoint")

class QuizPayload(BaseModel):
    email: str
    secret: str
    url: str

@app.post("/quiz")
async def receive_quiz(payload: QuizPayload, background_tasks: BackgroundTasks):
    if payload.secret != APP_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    background_tasks.add_task(handle_quiz_request, payload.email, payload.secret, payload.url)
    return {"status": "accepted", "message": "Quiz processing started"}

@app.get("/health")
def health():
    # load_dotenv() already ran at import time; return email from env
    return {"status": "ok", "email": APP_EMAIL}
