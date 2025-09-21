from typing import List, Optional, Union
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging

# transformers imports
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Config ---
MODEL_NAME = "google/pegasus-xsum"  # change if you prefer another PEGASUS checkpoint
DEVICE = 0 if torch.cuda.is_available() else -1  # use GPU (0) if available, else CPU (-1)
DEFAULT_MAX_LENGTH = 60
DEFAULT_MIN_LENGTH = 10
DEFAULT_NUM_BEAMS = 4

app = FastAPI(title="PEGASUS Summarization API", version="1.0")

# --- Pydantic models ---
class SummarizeRequest(BaseModel):
    text: Optional[str] = Field(None, description="Single document text to summarize")
    texts: Optional[List[str]] = Field(None, description="List of documents to summarize")
    max_length: Optional[int] = Field(DEFAULT_MAX_LENGTH, description="Maximum summary length")
    min_length: Optional[int] = Field(DEFAULT_MIN_LENGTH, description="Minimum summary length")
    num_beams: Optional[int] = Field(DEFAULT_NUM_BEAMS, description="Number of beams for beam search")

class SummaryResult(BaseModel):
    original: str
    summary: str

class SummarizeResponse(BaseModel):
    results: List[SummaryResult]


# --- Startup: load model/pipeline ---
@app.on_event("startup")
def load_model():
    """Load the tokenizer and model into memory (blocking at startup)."""
    global summarizer
    try:
        logger.info(f"Loading model {MODEL_NAME} on device {DEVICE}...")
        # Using pipeline provides a convenient wrapper; you can also load model/tokenizer separately
        summarizer = pipeline(
            "summarization",
            model=MODEL_NAME,
            device=DEVICE,
        )
        logger.info("Model loaded.")
    except Exception as e:
        logger.exception("Failed to load model at startup: %s", e)
        raise


# --- Helper that runs the blocking pipeline in a threadpool ---
def _summarize_blocking(inputs: List[str], max_length: int, min_length: int, num_beams: int) -> List[str]:
    """Blocking call to the summarization pipeline. Returns list of summaries.
    This runs in a threadpool when invoked from the async endpoint.
    """
    # The pipeline accepts a list; set parameters
    params = {
        "max_length": max_length,
        "min_length": min_length,
        "num_beams": num_beams,
        # "early_stopping": True,  # optional
    }
    outputs = summarizer(inputs, **params)
    # pipeline returns list of dicts with 'summary_text'
    summaries = [o["summary_text"] if isinstance(o, dict) and "summary_text" in o else (o[0]["summary_text"] if isinstance(o, list) and len(o) and isinstance(o[0], dict) and "summary_text" in o[0] else str(o)) for o in outputs]
    return summaries


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(req: SummarizeRequest):
    """Summarize a single document (text) or multiple documents (texts).

    Examples:
    - POST {"text": "Long article..."}
    - POST {"texts": ["doc1...", "doc2..."]}
    """
    # Validate input
    inputs: List[str] = []
    if req.text and req.text.strip():
        inputs = [req.text.strip()]
    elif req.texts and isinstance(req.texts, list) and len(req.texts) > 0:
        # filter out empty strings
        inputs = [t.strip() for t in req.texts if isinstance(t, str) and t.strip()]
    else:
        raise HTTPException(status_code=400, detail="Provide either 'text' or a non-empty 'texts' list.")

    # clamp/validate params
    max_length = max(1, int(req.max_length or DEFAULT_MAX_LENGTH))
    min_length = max(0, int(req.min_length or DEFAULT_MIN_LENGTH))
    num_beams = max(1, int(req.num_beams or DEFAULT_NUM_BEAMS))

    # Run blocking summarization in threadpool to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    try:
        summaries = await loop.run_in_executor(None, _summarize_blocking, inputs, max_length, min_length, num_beams)
    except Exception as e:
        logger.exception("Error during summarization: %s", e)
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")

    results = [SummaryResult(original=inp, summary=out) for inp, out in zip(inputs, summaries)]
    return SummarizeResponse(results=results)


# --- Health check ---
@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}


# --- If run directly, start uvicorn (convenience) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_pegasus_summarizer:app", host="0.0.0.0", port=8000, log_level="info")
