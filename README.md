# AI Video Assistant (RAG + Pydantic + Sarvam AI)

Ask questions about any YouTube video that has captions. Answers come with clickable timestamps.

## Run it (10 minutes)
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # then paste your SARVAM_API_KEY (dashboard.sarvam.ai)
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000, paste a YouTube link, click **Add video**, then ask a question.
First run downloads the embedding model (~90 MB). Pick a video with captions.

## Architecture
```
Browser (HTML/JS)
   |  POST /api/videos {url}            POST /api/ask {video_id, question}
   v
FastAPI  --- CORS middleware --- Pydantic validation (schemas.py)
   |
   +-- ingest:  YouTube transcript -> sliding-window chunks -> embeddings -> SQLite
   +-- ask:     embed question -> cosine top-4 chunks -> grounded prompt -> Sarvam LLM
                -> answer + timestamp sources
```

| File | Job |
|---|---|
| app/schemas.py | Pydantic request/response models and validation |
| app/rag.py | transcript fetch, chunking (overlap), embeddings, retrieval |
| app/db.py | SQLite tables: videos, chunks (+ embedding BLOB) |
| app/llm.py | Sarvam chat call with a "answer only from context" prompt |
| app/main.py | FastAPI routes, middleware, error handling |
| static/index.html | Minimal frontend |

## Optional upgrades (only if you have time)
- Hindi videos: set EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- Videos without captions: transcribe audio with Sarvam speech-to-text, then feed the same pipeline
- Hybrid search (BM25 + vectors) and a reranker
- JWT login so each user sees only their own videos
- Move vectors to FAISS or Chroma for large collections
