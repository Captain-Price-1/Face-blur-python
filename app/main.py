from fastapi import FastAPI

app = FastAPI(title="Face Blur App", version="0.1.0")


@app.get("/api/health")
def health():
    return {"ok": True}
