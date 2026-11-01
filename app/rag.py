"""RAG core: transcript -> chunks -> embeddings -> retrieval."""
import os
import re
import numpy as np

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_model = None


def extract_video_id(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url.strip()):
        return url.strip()
    raise ValueError("Could not find a YouTube video id in that URL")


def fetch_transcript(video_id: str) -> list[dict]:
    """Returns [{'start': seconds, 'text': '...'}]. Tries English then Hindi."""
    from youtube_transcript_api import YouTubeTranscriptApi

    fetched = YouTubeTranscriptApi().fetch(video_id, languages=["en", "hi"])
    return [{"start": int(s.start), "text": s.text.replace("\n", " ")} for s in fetched]


def chunk_transcript(segments: list[dict], window: int = 8, stride: int = 5) -> list[dict]:
    """Sliding window over caption segments.

    window = segments per chunk, stride = step. window > stride means chunks OVERLAP,
    so a sentence split across a boundary still appears whole in one chunk.
    """
    chunks = []
    for i in range(0, len(segments), stride):
        part = segments[i : i + window]
        if not part:
            break
        chunks.append({"start": part[0]["start"], "text": " ".join(p["text"] for p in part)})
        if i + window >= len(segments):
            break
    return chunks


def embed(texts: list[str]) -> np.ndarray:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBED_MODEL)
    return np.asarray(_model.encode(texts, normalize_embeddings=True), dtype=np.float32)


def retrieve(question: str, chunks: list[dict], matrix: np.ndarray, k: int = 4) -> list[dict]:
    """Cosine similarity (embeddings are normalised, so dot product = cosine)."""
    if not chunks:
        return []
    q = embed([question])[0]
    scores = matrix @ q
    top = np.argsort(-scores)[:k]
    return [{**chunks[i], "score": float(scores[i])} for i in sorted(top, key=lambda j: -scores[j])]


def fmt_time(sec: int) -> str:
    return f"{sec // 60:02d}:{sec % 60:02d}"
