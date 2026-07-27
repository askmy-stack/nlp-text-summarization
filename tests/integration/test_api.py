from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from textSummarizer.serving.app import app
from textSummarizer.serving.sandbox import get_sandbox_dir


@pytest.mark.asyncio
async def test_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["models_available"] >= 1


@pytest.mark.asyncio
async def test_list_models():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/models")
    assert response.status_code == 200
    assert "extractive" in response.json()


@pytest.mark.asyncio
async def test_summarize_extractive():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/summarize",
            json={
                "text": "AI is changing the world. Machine learning powers assistants.",
                "model": "extractive",
                "strategy": "stuff",
                "max_length": 64,
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert data["summary"]
    assert data["model"] == "extractive"


@pytest.mark.asyncio
async def test_summarize_extractive_map_reduce():
    transport = ASGITransport(app=app)
    text = ". ".join([f"Sentence number {i} with useful content here" for i in range(25)])
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/summarize",
            json={
                "text": text,
                "model": "extractive",
                "strategy": "map_reduce",
                "max_length": 64,
            },
        )
    assert response.status_code == 200
    assert response.json()["summary"]


@pytest.mark.asyncio
async def test_summarize_multimodal_video_json():
    transport = ASGITransport(app=app)
    with patch("textSummarizer.multimodal.router.VideoSummarizer.summarize") as mock_summarize:
        mock_summarize.return_value = {
            "document": "[0000.0s] Speech: Demo transcript.",
            "transcript": "Demo transcript.",
            "visual_captions": "A presenter.",
            "summary": "A short demo video.",
            "model": "extractive",
            "segments": [],
            "frame_captions": [],
        }
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/summarize/multimodal",
                json={
                    "input_type": "video",
                    "path": str(get_sandbox_dir() / "demo.mp4"),
                    "model": "extractive",
                    "strategy": "map_reduce",
                    "max_length": 64,
                },
            )
    assert response.status_code == 200
    data = response.json()
    assert data["input_type"] == "video"
    assert data["summary"] == "A short demo video."
    assert data["document"]
    assert data["transcript"] == "Demo transcript."


@pytest.mark.asyncio
async def test_summarize_hierarchical_strategy():
    transport = ASGITransport(app=app)
    text = ". ".join([f"Sentence number {i} with useful content here" for i in range(25)])
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/summarize",
            json={
                "text": text,
                "model": "extractive",
                "strategy": "hierarchical",
                "max_length": 64,
            },
        )
    assert response.status_code == 200
    assert response.json()["strategy"] == "hierarchical"


@pytest.mark.asyncio
async def test_summarize_unknown_model_returns_422():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/summarize",
            json={
                "text": "Hello world.",
                "model": "not-a-real-model",
                "strategy": "stuff",
                "max_length": 64,
            },
        )
    assert response.status_code == 422
    assert "Unknown model" in response.json()["detail"]
