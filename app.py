"""
app.py — FastAPI entry point for ClauseWise.

Accepts contract file uploads, runs the full inference pipeline, and returns
the model's response. Handles S3 storage and CloudWatch-compatible logging.

Environment variables required:
    GROQ_API_KEY        — Groq API key for DeepSeek-R1 reasoning
    AWS_S3_BUCKET       — S3 bucket name for document storage
    AWS_REGION          — AWS region (e.g. us-east-1)

Environment variables optional:
    PORT                — Port to run on (default: 8000)
"""
# Imports
import logging
import os
import uuid
from contextlib import asynccontextmanager

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.classifier import _ensure_loaded as load_classifier
from src.extractor import _ensure_loaded as load_extractor
from src.segmenter import ContractSegmenter, load_contract_text
from src.pipeline import route_user_query

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Config
S3_BUCKET = os.environ.get("AWS_S3_BUCKET", "")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
PORT = int(os.environ.get("PORT", 8000))

# S3 client (only used if S3_BUCKET is set)
_s3_client = None

def _get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=AWS_REGION)
    return _s3_client

# Lifespan — pre-load models at container startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load LegalBERT and FLAN-T5 into memory when the container starts.
    Without this, the first request waits 30-60s for model loading.
    ECS health checks would also fail during that window.
    """
    logger.info("Loading classifier (LegalBERT)...")
    load_classifier()
    logger.info("Classifier loaded.")

    logger.info("Loading extractor (FLAN-T5 + LoRA)...")
    load_extractor()
    logger.info("Extractor loaded.")

    logger.info("All models loaded. ClauseWise is ready.")
    yield
    logger.info("Shutting down.")

# App
app = FastAPI(
    title="ClauseWise",
    description="Legal contract analysis via NLP — clause classification, extraction, and reasoning.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class HealthResponse(BaseModel):
    status: str

class AnalyzeResponse(BaseModel):
    answer: str
    document_key: str | None  # S3 key, or None if S3 not configured
    num_clauses: int

# Endpoints
@app.get("/health", response_model=HealthResponse)
def health():
    """
    ECS pings this every 30 seconds to verify the container is alive.
    Must respond quickly — model loading happens at startup, not here.
    """
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(..., description="Contract file (.txt or .pdf)"),
    question: str = Form(..., description="Question to ask about the contract"),
):
    """
    Main endpoint. Accepts a contract file and a question, runs the full
    ClauseWise pipeline, and returns the model's answer.

    Steps:
        1. Validate file type
        2. Save to a temp path locally
        3. Upload to S3 (if configured)
        4. Load and segment contract text
        5. Run route_user_query through the full pipeline
        6. Return answer, S3 key, and clause count
    """
    # Validate file type
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()

    if ext not in (".txt", ".pdf"):
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext}'. Upload a .txt or .pdf file.",
        )

    # Save to temp file
    # ECS tasks have a writable /tmp directory
    temp_path = f"/tmp/{uuid.uuid4().hex}{ext}"

    try:
        contents = await file.read()
        with open(temp_path, "wb") as f_out:
            f_out.write(contents)
        logger.info("Saved upload to %s (%d bytes)", temp_path, len(contents))
    except Exception as e:
        logger.error("Failed to save uploaded file: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

    # Upload to S3
    document_key = None

    if S3_BUCKET:
        document_key = f"uploads/{uuid.uuid4().hex}{ext}"
        try:
            s3 = _get_s3_client()
            s3.upload_file(temp_path, S3_BUCKET, document_key)
            logger.info("Uploaded to S3: s3://%s/%s", S3_BUCKET, document_key)
        except (BotoCoreError, ClientError) as e:
            logger.error("S3 upload failed: %s", e)
            # Non-fatal — continue processing even if S3 upload fails
            document_key = None
    else:
        logger.warning("AWS_S3_BUCKET not set — skipping S3 upload.")

    # Load and segment contract
    try:
        contract_text = load_contract_text(temp_path)
    except Exception as e:
        logger.error("Failed to read contract: %s", e)
        raise HTTPException(status_code=422, detail=f"Could not read contract: {e}")

    segmenter = ContractSegmenter()
    clauses = segmenter.segment_contract(contract_text)

    if not clauses:
        raise HTTPException(
            status_code=422,
            detail="No clauses could be extracted from the uploaded document.",
        )

    logger.info("Segmented contract into %d clauses.", len(clauses))

    # Run pipeline
    try:
        answer = route_user_query(question, clauses)
        logger.info("Pipeline complete. Answer length: %d chars", len(answer))
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        raise HTTPException(status_code=500, detail="Inference pipeline failed.")

    # Return response
    return AnalyzeResponse(
        answer=answer,
        document_key=document_key,
        num_clauses=len(clauses),
    )

# Local dev entrypoint
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=True)