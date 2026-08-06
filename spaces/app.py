"""SummarizeHub HuggingFace Space — GPU-backed abstractive summarization."""

from __future__ import annotations

import os
import time
from collections import defaultdict

import gradio as gr

try:
    import spaces
except ImportError:
    spaces = None  # type: ignore[assignment]

from textSummarizer.models import ModelFactory
from textSummarizer.pipelines import STRATEGIES

MODELS = ["bart", "flan-t5", "t5", "pegasus", "extractive"]
DEFAULT_MODEL = "bart"
SPACE_RATE_LIMIT_PER_MINUTE = int(os.getenv("SPACE_RATE_LIMIT_PER_MINUTE", "10"))

# Per-client sliding window (timestamps of accepted requests in the last minute).
_hits: dict[str, list[float]] = defaultdict(list)


def _client_key(request: gr.Request | None) -> str:
    if request is None:
        return "anon"
    client = getattr(request, "client", None)
    host = getattr(client, "host", None) if client is not None else None
    session = getattr(request, "session_hash", None)
    return str(host or session or "anon")


def _allow_request(request: gr.Request | None) -> bool:
    key = _client_key(request)
    now = time.monotonic()
    window = [t for t in _hits[key] if now - t < 60.0]
    if len(window) >= SPACE_RATE_LIMIT_PER_MINUTE:
        _hits[key] = window
        return False
    window.append(now)
    _hits[key] = window
    return True


def _summarize_impl(text: str, model: str, strategy: str, max_length: int) -> str:
    if not text.strip():
        return "Please provide text to summarize."
    summarizer = ModelFactory.create(model)
    return summarizer.summarize(text, max_length=max_length, strategy=strategy)


def _guarded_summarize(
    text: str,
    model: str,
    strategy: str,
    max_length: int,
    request: gr.Request,
) -> str:
    if not _allow_request(request):
        return (
            f"Rate limit exceeded (max {SPACE_RATE_LIMIT_PER_MINUTE} requests/minute). "
            "Please wait a moment and try again."
        )
    return _summarize_impl(text, model, strategy, max_length)


if spaces is not None:

    @spaces.GPU(duration=60)
    def summarize(
        text: str,
        model: str,
        strategy: str,
        max_length: int,
        request: gr.Request,
    ) -> str:
        return _guarded_summarize(text, model, strategy, max_length, request)

else:

    def summarize(
        text: str,
        model: str,
        strategy: str,
        max_length: int,
        request: gr.Request,
    ) -> str:
        return _guarded_summarize(text, model, strategy, max_length, request)


demo = gr.Interface(
    fn=summarize,
    inputs=[
        gr.Textbox(label="Text", lines=10, placeholder="Paste article or dialogue here..."),
        gr.Dropdown(choices=MODELS, value=DEFAULT_MODEL, label="Model"),
        gr.Dropdown(choices=list(STRATEGIES), value="stuff", label="Strategy"),
        gr.Slider(minimum=32, maximum=256, value=128, step=16, label="Max length"),
    ],
    outputs=gr.Textbox(label="Summary", lines=6),
    title="SummarizeHub",
    description=(
        "GPU-backed abstractive summarization with BART, FLAN-T5, and long-document "
        f"strategies including hierarchical and RAG. "
        f"Public demo rate limit: {SPACE_RATE_LIMIT_PER_MINUTE} requests/minute per client."
    ),
    examples=[
        [
            (
                "Artificial intelligence is reshaping healthcare, finance, and transportation. "
                "Machine learning models can detect diseases from medical images. "
                "Natural language processing powers chatbots and document summarization."
            ),
            "bart",
            "stuff",
            128,
        ],
    ],
)

if __name__ == "__main__":
    demo.launch()
