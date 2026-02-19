from fastapi import FastAPI
from routers import gemini, grok

app = FastAPI(title="AI C2")

app.include_router(gemini.router)
app.include_router(grok.router)


@app.get("/")
async def root():
    return {"message": "Wallscreet AI C2 Online"}

