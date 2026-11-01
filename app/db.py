"""SQLite storage: one table for videos, one for transcript chunks + embeddings."""
import os
import sqlite3
import numpy as np

DB_PATH = os.getenv("DB_PATH", "assistant.db")


def conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL REFERENCES videos(video_id),
                start_sec INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_video ON chunks(video_id);
            """
        )


def save_video(video_id: str, chunks: list[dict], embeddings: np.ndarray) -> None:
    with conn() as c:
        c.execute("DELETE FROM chunks WHERE video_id = ?", (video_id,))
        c.execute("INSERT OR REPLACE INTO videos(video_id) VALUES (?)", (video_id,))
        c.executemany(
            "INSERT INTO chunks(video_id, start_sec, text, embedding) VALUES (?,?,?,?)",
            [
                (video_id, ch["start"], ch["text"], emb.astype(np.float32).tobytes())
                for ch, emb in zip(chunks, embeddings)
            ],
        )


def load_chunks(video_id: str) -> tuple[list[dict], np.ndarray]:
    with conn() as c:
        rows = c.execute(
            "SELECT start_sec, text, embedding FROM chunks WHERE video_id = ? ORDER BY start_sec",
            (video_id,),
        ).fetchall()
    chunks = [{"start": r["start_sec"], "text": r["text"]} for r in rows]
    if not rows:
        return chunks, np.zeros((0, 1), dtype=np.float32)
    matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    return chunks, matrix


def list_videos() -> list[str]:
    with conn() as c:
        return [r["video_id"] for r in c.execute("SELECT video_id FROM videos ORDER BY created_at DESC")]
