"""Sarvam AI (Indian LLM) call with a grounded prompt."""
import os
from sarvamai import SarvamAI

MODEL = os.getenv("SARVAM_MODEL", "sarvam-105b")

SYSTEM = (
    "You are a helpful video assistant. Use the video title and transcript excerpts to answer. "
    "Give a clear, direct answer in 2 to 5 sentences. Cite timestamps like [03:25]. "
    "If the excerpts only partly answer, share what they do say and state what is missing. "
    "If nothing relevant is there, say the video's captions don't cover it. Never invent names or facts. "
    "Reply in the language of the question."
)


def answer(question: str, hits: list[dict], fmt_time, title: str = "") -> str:
    key = os.getenv("SARVAM_API_KEY")
    if not key:
        raise RuntimeError("SARVAM_API_KEY is not set")
    context = "\n".join(f"[{fmt_time(h['start'])}] {h['text']}" for h in hits)
    client = SarvamAI(api_subscription_key=key)
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"Video title: {title}\n\nTranscript excerpts:\n{context}\n\nQuestion: {question}",
        },
    ]
    # Reasoning tokens share the max_tokens budget, so give plenty of room and retry once.
    for budget in (2500, 4000):
        resp = client.chat.completions(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=budget,
            reasoning_effort="low",
        )
        text = resp.choices[0].message.content
        if text and text.strip():
            return text.strip()
    return "The model could not produce an answer. Please rephrase and try again."