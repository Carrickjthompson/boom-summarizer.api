from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os

app = FastAPI(title="Boom Summarizer API", version="1.0")

class SummaryOut(BaseModel):
    purpose: str
    insights: List[str]
    risks: List[str]
    next_steps: List[str]
    notes: Optional[List[str]] = None
    model_used: str

@app.get("/health")
def health():
    return {"ok": True}

def heuristic_summarize(text: str) -> SummaryOut:
    # Fallback summary without LLM: crude but deterministic
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    head = lines[:3]
    tail = lines[-3:] if len(lines) >= 3 else lines
    insights = (head + tail)[:5] or ["No salient lines detected."]
    return SummaryOut(
        purpose="Summarize an uploaded document or pasted text into Boom’s standard sections.",
        insights=insights,
        risks=["Not verified. Heuristic mode."],
        next_steps=["Provide clearer objectives.", "Flag key metrics to extract.", "Request LLM mode if available."],
        notes=["LLM disabled or key missing."],
        model_used="heuristic"
    )

async def read_file_bytes(upload: UploadFile) -> str:
    # Minimal text extraction for .txt only; PDFs/others should be preprocessed client-side
    if upload.filename.lower().endswith(".txt"):
        return (await upload.read()).decode("utf-8", errors="ignore")
    raise HTTPException(status_code=400, detail="Only .txt accepted in this minimal build. Use 'text' field for pasted content.")

@app.post("/summarize", response_model=SummaryOut)
async def summarize(
    text: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
    summary_type: str = Form(default="executive")  # "short" | "detailed" | "executive"
):
    # Input guard
    if not text and not file:
        raise HTTPException(status_code=400, detail="Provide 'text' or a .txt 'file'.")

    # Assemble input text
    if file:
        text = await read_file_bytes(file)
    assert text is not None

    # Optional OpenAI LLM path (enable later by setting OPENAI_API_KEY)
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        try:
            import httpx
            prompt = f"""Summarize the document into this JSON with short, precise bullets:
{{
  "purpose": "...",
  "insights": ["...", "..."],
  "risks": ["...", "..."],
  "next_steps": ["...", "..."]
}}
Constraints: U.S. English, numbered bullets where useful, no fluff. Summary type: {summary_type}.
Document:
{text[:12000]}"""
            # Using Responses API-style call for portability; replace with your provider as needed.
            # This keeps the skeleton; you can swap to your preferred endpoint.
            headers = {"Authorization": f"Bearer {openai_key}"}
            payload = {
                "model": "gpt-4o-mini",
                "input": [{"role":"user","content": prompt}]
            }
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post("https://api.openai.com/v1/responses", json=payload, headers=headers)
                r.raise_for_status()
                data = r.json()
            # Extract text; be defensive
            txt = str(data)
            # Simple bracket find; you may replace with robust JSON parsing if using JSON mode
            start = txt.find("{")
            end = txt.rfind("}")
            body = txt[start:end+1] if start != -1 and end != -1 else ""
            import json
            parsed = json.loads(body)
            return SummaryOut(
                purpose=parsed.get("purpose",""),
                insights=parsed.get("insights",[]) or [],
                risks=parsed.get("risks",[]) or [],
                next_steps=parsed.get("next_steps",[]) or [],
                model_used="gpt-4o-mini"
            )
        except Exception:
            # Fall back silently to heuristic
            pass

    return heuristic_summarize(text)
