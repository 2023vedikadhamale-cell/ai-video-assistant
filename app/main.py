"""FastAPI backend: ingest a video, then ask questions about it."""
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import db, llm, rag
from .schemas import AskIn, AskOut, Source, VideoIn, VideoOut

app = FastAPI(title="AI Video Assistant")

# Middleware: runs on every request/response. Here it adds CORS headers.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATIC = PROJECT_ROOT / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.on_event("startup")
def startup() -> None:
    db.init_db()


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.post("/api/videos", response_model=VideoOut)
def add_video(body: VideoIn):
    try:
        video_id = rag.extract_video_id(body.url)
        segments = rag.fetch_transcript(video_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=404, detail="No transcript available for this video")
    chunks = rag.chunk_transcript(segments)
    embeddings = rag.embed([c["text"] for c in chunks])
    db.save_video(video_id, chunks, embeddings)
    return VideoOut(video_id=video_id, num_chunks=len(chunks))


@app.get("/api/videos")
def videos():
    return db.list_videos()


@app.post("/api/ask", response_model=AskOut)
def ask(body: AskIn):
    chunks, matrix = db.load_chunks(body.video_id)
    if not chunks:
        raise HTTPException(status_code=404, detail="Video not added yet")
    hits = rag.retrieve(body.question, chunks, matrix, k=4)
    q = body.question.lower()
    if len(q.split()) <= 5 or any(w in q for w in ("summar", "taught", "about", "overview")):
        step = max(1, len(chunks) // 6)
        seen = {h["start"] for h in hits}
        hits = sorted(
            hits + [c for c in chunks[::step][:6] if c["start"] not in seen],
            key=lambda c: c["start"],
        )
    try:
        text = llm.answer(body.question, hits, rag.fmt_time)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"AI error: {e}")
    sources = [
        Source(
            timestamp=rag.fmt_time(h["start"]),
            link=f"https://youtu.be/{body.video_id}?t={h['start']}",
            text=h["text"],
        )
        for h in hits
    ]
    return AskOut(answer=text, sources=sources)
