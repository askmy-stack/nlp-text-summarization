import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import RedirectResponse, StreamingResponse

from textSummarizer.components.prediction import PredictionPipeline
from textSummarizer.grading.geval import geval_score
from textSummarizer.grading.llm_judge import LLMJudge
from textSummarizer.grading.rubric import GradingRubric
from textSummarizer.models import ModelFactory
from textSummarizer.multimodal.base import InputType, MultimodalInput
from textSummarizer.multimodal.router import MultimodalRouter
from textSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from textSummarizer.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from textSummarizer.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from textSummarizer.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from textSummarizer.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline
from textSummarizer.pipelines import STRATEGY_PATTERN
from textSummarizer.pipelines.citations import summarize_with_citations
from textSummarizer.serving.auth import (
    APIKeyMiddleware,
    SimpleRateLimitMiddleware,
    verify_api_key,
    verify_train_key,
)
from textSummarizer.serving.gpu_pool import get_gpu_pool
from textSummarizer.serving.sandbox import resolve_sandboxed_path

API_VERSION = "1.2.0"
MAX_VIDEO_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
VIDEO_MIME_TYPES = {
    "video/mp4",
    "video/webm",
    "video/quicktime",
    "video/x-matroska",
}

SUMMARIZE_EXAMPLE = {
    "text": (
        "Artificial intelligence is reshaping healthcare, finance, and transportation. "
        "Machine learning models can detect diseases from medical images. "
        "Natural language processing powers chatbots and document summarization."
    ),
    "model": "extractive",
    "strategy": "stuff",
    "max_length": 128,
}

SUMMARIZE_RESPONSE_EXAMPLE = {
    "summary": (
        "Artificial intelligence is reshaping healthcare, finance, and transportation. "
        "Machine learning models can detect diseases from medical images."
    ),
    "model": "extractive",
    "strategy": "stuff",
}


class SummarizeRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [SUMMARIZE_EXAMPLE]},
    )

    text: str = Field(..., min_length=1, max_length=50000, description="Source text to summarize")
    model: str = Field(default="extractive", description="Model name from the registry")
    strategy: str = Field(
        default="stuff",
        pattern=STRATEGY_PATTERN,
        description="Long-document strategy (stuff, map_reduce, refine, hierarchical, rag)",
    )
    max_length: int = Field(default=128, ge=16, le=512, description="Maximum summary length")


class SummarizeResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [SUMMARIZE_RESPONSE_EXAMPLE]},
    )

    summary: str
    model: str
    strategy: str


class SummarizeWithCitationsResponse(SummarizeResponse):
    citations: list[dict]


class HealthResponse(BaseModel):
    status: str
    version: str
    models_available: int


class MultimodalJsonRequest(BaseModel):
    input_type: Literal["text", "image", "audio", "video"] = Field(
        ..., description="Modality of the input"
    )
    text: str | None = Field(default=None, description="Plain text (for text modality)")
    base64_data: str | None = Field(
        default=None, description="Base64-encoded file content (image/audio/video)"
    )
    path: str | None = Field(default=None, description="Server-side file path")
    model: str = Field(default="extractive", description="Text summarization model")
    strategy: str = Field(
        default="stuff",
        pattern=STRATEGY_PATTERN,
        description="Long-document strategy (stuff, map_reduce, refine, hierarchical, rag)",
    )
    max_length: int = Field(default=128, ge=16, le=512, description="Maximum summary length")


class MultimodalResponse(BaseModel):
    input_type: str
    summary: str
    model: str
    strategy: str
    caption: str | None = None
    transcript: str | None = None
    document: str | None = None
    visual_captions: str | None = None
    chapters: list[dict] | None = None


class GEvalResult(BaseModel):
    geval_score: float
    geval_reason: str
    method: str
    model: str | None = None


class GradeRequest(BaseModel):
    source: str = Field(..., min_length=1, max_length=50000, description="Source text")
    summary: str = Field(..., min_length=1, max_length=10000, description="Summary to grade")
    threshold: float = Field(default=3.5, ge=1.0, le=5.0, description="Pass threshold (1-5)")
    rubric: str | None = Field(default=None, description="YAML rubric name from config/rubrics/")
    use_geval: bool = Field(
        default=False,
        description="Run deepeval G-Eval LLM judge when API key is configured",
    )


class GradeScoreResponse(BaseModel):
    coherence: int
    faithfulness: int
    fluency: int
    relevance: int
    average: float
    feedback: str


class GradeResponse(BaseModel):
    score: GradeScoreResponse
    passes: bool
    threshold: float
    geval: GEvalResult | None = None


def _summarize_impl(request: SummarizeRequest) -> str:
    if request.model == "pegasus":
        pipeline = PredictionPipeline(model_name="pegasus")
        return pipeline.predict(
            request.text, strategy=request.strategy, max_length=request.max_length
        )

    pool = get_gpu_pool()
    summarizer = pool.get_model(request.model, lambda: ModelFactory.create(request.model))
    return summarizer.summarize(
        request.text, max_length=request.max_length, strategy=request.strategy
    )


def _run_summarization(request: SummarizeRequest) -> str:
    try:
        if request.model == "extractive":
            return _summarize_impl(request)
        return get_gpu_pool().run(_summarize_impl, request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


async def _stream_summary(request: SummarizeRequest) -> AsyncIterator[str]:
    summary = _run_summarization(request)
    words = summary.split()
    if not words:
        yield 'data: {"done": true, "summary": ""}\n\n'
        return

    partial: list[str] = []
    for word in words:
        partial.append(word)
        payload = {"token": word, "partial": " ".join(partial)}
        yield f"data: {json.dumps(payload)}\n\n"

    final = {
        "done": True,
        "summary": summary,
        "model": request.model,
        "strategy": request.strategy,
    }
    yield f"data: {json.dumps(final)}\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = ModelFactory.list_models()
    # Warm extractive model in cache for faster first request
    ModelFactory.create("extractive")
    yield


def _rate_limit() -> int:
    return int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))


app = FastAPI(
    title="SummarizeHub API",
    description=(
        "Multimodal summarization platform — text, image, audio, and video inputs with "
        "subjective grading and MCP agent integration"
    ),
    version=API_VERSION,
    lifespan=lifespan,
)
app.add_middleware(SimpleRateLimitMiddleware, requests_per_minute=_rate_limit())
app.add_middleware(APIKeyMiddleware)


@app.get("/", tags=["docs"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health():
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        models_available=len(ModelFactory.list_models()),
    )


@app.get("/models", tags=["models"])
async def list_models():
    return ModelFactory.list_models()


@app.post(
    "/summarize",
    response_model=SummarizeResponse,
    tags=["inference"],
    dependencies=[Depends(verify_api_key)],
)
async def summarize(request: SummarizeRequest):
    summary = _run_summarization(request)
    return SummarizeResponse(summary=summary, model=request.model, strategy=request.strategy)


@app.post(
    "/summarize/stream",
    tags=["inference"],
    dependencies=[Depends(verify_api_key)],
)
async def summarize_stream(request: SummarizeRequest):
    return StreamingResponse(
        _stream_summary(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post(
    "/summarize/citations",
    response_model=SummarizeWithCitationsResponse,
    tags=["inference"],
    dependencies=[Depends(verify_api_key)],
)
async def summarize_with_citation_spans(request: SummarizeRequest):
    summary = _run_summarization(request)
    result = summarize_with_citations(request.text, summary)
    return SummarizeWithCitationsResponse(
        summary=result["summary"],
        model=request.model,
        strategy=request.strategy,
        citations=result["citations"],
    )


def _multimodal_result_to_response(result: dict, strategy: str) -> MultimodalResponse:
    return MultimodalResponse(
        input_type=result["input_type"],
        summary=result["summary"],
        model=result["model"],
        strategy=strategy,
        caption=result.get("caption"),
        transcript=result.get("transcript"),
        document=result.get("document"),
        visual_captions=result.get("visual_captions"),
        chapters=result.get("chapters"),
    )


def _validate_video_upload(file: UploadFile, content: bytes) -> None:
    if file.content_type and file.content_type not in VIDEO_MIME_TYPES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported video MIME type: {file.content_type}. "
                f"Supported: {', '.join(sorted(VIDEO_MIME_TYPES))}"
            ),
        )
    if len(content) > MAX_VIDEO_UPLOAD_BYTES:
        max_mb = MAX_VIDEO_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=422,
            detail=f"Video file exceeds maximum size of {max_mb} MB",
        )


@app.post(
    "/summarize/multimodal",
    response_model=MultimodalResponse,
    tags=["inference"],
    dependencies=[Depends(verify_api_key)],
)
async def summarize_multimodal_json(request: MultimodalJsonRequest):
    try:
        input_type = InputType(request.input_type)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid input_type: {request.input_type}",
        ) from exc

    sandboxed_path = str(resolve_sandboxed_path(request.path)) if request.path else None

    router = MultimodalRouter(text_model=request.model)
    try:
        result = router.summarize(
            MultimodalInput(
                input_type=input_type,
                text=request.text,
                base64_data=request.base64_data,
                path=sandboxed_path,
            ),
            max_length=request.max_length,
            strategy=request.strategy,
        )
    except (ValueError, NotImplementedError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return _multimodal_result_to_response(result, request.strategy)


@app.post(
    "/summarize/multimodal/upload",
    response_model=MultimodalResponse,
    tags=["inference"],
    dependencies=[Depends(verify_api_key)],
)
async def summarize_multimodal_upload(
    file: Annotated[UploadFile, File()],
    input_type: Literal["image", "audio", "video"] = Form(...),
    model: str = Form(default="extractive"),
    strategy: str = Form(default="stuff"),
    max_length: int = Form(default=128, ge=16, le=512),
):
    content = await file.read()
    if input_type == "video":
        _validate_video_upload(file, content)
        if strategy == "stuff":
            strategy = "map_reduce"
    router = MultimodalRouter(text_model=model)
    try:
        result = router.summarize(
            MultimodalInput(
                input_type=InputType(input_type),
                data=content,
                metadata={"filename": file.filename or ""},
            ),
            max_length=max_length,
            strategy=strategy,
        )
    except (ValueError, NotImplementedError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return _multimodal_result_to_response(result, strategy)


@app.post(
    "/grade",
    response_model=GradeResponse,
    tags=["grading"],
    dependencies=[Depends(verify_api_key)],
)
async def grade_summary(request: GradeRequest):
    if request.rubric:
        rubric = GradingRubric.from_yaml(request.rubric)
    else:
        rubric = GradingRubric(threshold=request.threshold)
    judge = LLMJudge(use_llm=False)
    score = judge.grade(request.source, request.summary)

    geval_payload: GEvalResult | None = None
    if request.use_geval:
        raw = geval_score(request.source, request.summary, use_geval=True)
        geval_payload = GEvalResult(
            geval_score=float(raw["geval_score"]),
            geval_reason=str(raw["geval_reason"]),
            method=str(raw["method"]),
            model=str(raw["model"]) if raw.get("model") else None,
        )

    return GradeResponse(
        score=GradeScoreResponse(**score.to_dict()),
        passes=rubric.passes(score),
        threshold=rubric.threshold,
        geval=geval_payload,
    )


@app.post("/train", tags=["training"], dependencies=[Depends(verify_train_key)])
async def train_pipeline():
    stages = [
        DataIngestionTrainingPipeline(),
        DataValidationTrainingPipeline(),
        DataTransformationTrainingPipeline(),
        ModelTrainerTrainingPipeline(),
        ModelEvaluationTrainingPipeline(),
    ]
    for stage in stages:
        stage.main()
    return {"status": "training completed"}
