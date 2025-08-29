
from fastapi import FastAPI

app = FastAPI(title="Q-Sentinel Backend", version="1.0")

@app.get("/")
def read_root():
    return {"status": "online", "message": "Backend Q-Sentinel rulează corect!"}

@app.get("/health")
def health_check():
    return {"health": "ok"}

