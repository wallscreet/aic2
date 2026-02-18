from fastapi import FastAPI
from routers import gemini

app = FastAPI(title="AI C2")

app.include_router(gemini.router)

@app.get("/")
async def root():
    return {"message": "Wallscreet AI C2 Online"}

