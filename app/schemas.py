"""Pydantic models: validate every request and response at the API boundary."""
from pydantic import BaseModel, Field


class VideoIn(BaseModel):
    url: str = Field(min_length=11, max_length=300, description="YouTube video URL")


class VideoOut(BaseModel):
    video_id: str
    num_chunks: int


class AskIn(BaseModel):
    video_id: str = Field(min_length=11, max_length=11)
    question: str = Field(min_length=3, max_length=500)


class Source(BaseModel):
    timestamp: str      # e.g. "03:25"
    link: str           # jumps to that moment in the video
    text: str


class AskOut(BaseModel):
    answer: str
    sources: list[Source]
