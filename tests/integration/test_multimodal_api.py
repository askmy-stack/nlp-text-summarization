from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from textSummarizer.serving.app import app
from textSummarizer.serving.sandbox import get_sandbox_dir


@pytest.mark.asyncio
async def test_grade_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/grade",
            json={
                "source": "AI is reshaping healthcare. Machine learning detects disease.",
                "summary": "AI reshapes healthcare with machine learning.",
                "threshold": 3.0,
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert data["score"]["average"] >= 1.0
    assert "passes" in data


@pytest.mark.asyncio
async def test_multimodal_text_json():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/summarize/multimodal",
            json={
                "input_type": "text",
                "text": "AI is changing the world. Machine learning powers assistants.",
                "model": "extractive",
                "strategy": "stuff",
                "max_length": 64,
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert data["input_type"] == "text"
    assert data["summary"]


@pytest.mark.asyncio
async def test_multimodal_invalid_type():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/summarize/multimodal",
            json={
                "input_type": "pdf",
                "path": str(get_sandbox_dir() / "document.pdf"),
                "model": "extractive",
            },
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_multimodal_path_absolute_escape_returns_403():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/summarize/multimodal",
            json={
                "input_type": "image",
                "path": "/etc/passwd",
                "model": "extractive",
            },
        )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_multimodal_path_traversal_escape_returns_403():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/summarize/multimodal",
            json={
                "input_type": "audio",
                "path": "../../../../../../etc/passwd",
                "model": "extractive",
            },
        )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_multimodal_path_within_sandbox_reaches_router():
    """A path inside the sandbox should pass the sandbox guard and be handed
    to the router unchanged (mocked here to avoid loading real ML models).
    """
    transport = ASGITransport(app=app)
    sandboxed_file = get_sandbox_dir() / "does-not-exist.jpg"
    with patch(
        "textSummarizer.serving.app.MultimodalRouter.summarize",
        return_value={"input_type": "image", "summary": "ok", "model": "extractive"},
    ) as mock_summarize:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/summarize/multimodal",
                json={
                    "input_type": "image",
                    "path": str(sandboxed_file),
                    "model": "extractive",
                },
            )
    assert response.status_code == 200
    called_input = mock_summarize.call_args.args[0]
    assert called_input.path == str(sandboxed_file.resolve())
