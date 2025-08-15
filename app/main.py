import os
import uvicorn
import asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from router.translate import router as translate_router
from config import QUEUE_MAX_SIZE

app = FastAPI(title="Translatotron 2 + OpenVoice API", version="1.0")

# CORS setup for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global async queue for processing
task_queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)

@app.middleware("http")
async def queue_middleware(request: Request, call_next):
    if task_queue.full():
        return JSONResponse(
            status_code=503,
            content={"detail": "Server busy. Try again later."}
        )
    await task_queue.put(1)
    try:
        response = await call_next(request)
        return response
    finally:
        task_queue.get_nowait()
        task_queue.task_done()

# Attach router
app.include_router(translate_router, prefix="/api/translate", tags=["Translate"])

@app.get("/")
async def root():
    return {"status": "Translatotron2 + OpenVoice API is running."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
