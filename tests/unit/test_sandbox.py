import os

import pytest
from fastapi import HTTPException

from textSummarizer.serving.sandbox import get_sandbox_dir, resolve_sandboxed_path


@pytest.fixture
def sandbox(tmp_path):
    base = tmp_path / "sandbox"
    base.mkdir()
    (base / "nested").mkdir()
    (base / "file.txt").write_text("hello")
    (base / "nested" / "inner.txt").write_text("world")
    return base


def test_resolves_file_inside_sandbox(sandbox):
    resolved = resolve_sandboxed_path("file.txt", base_dir=sandbox)
    assert resolved == (sandbox / "file.txt").resolve()


def test_resolves_nested_file_inside_sandbox(sandbox):
    resolved = resolve_sandboxed_path("nested/inner.txt", base_dir=sandbox)
    assert resolved == (sandbox / "nested" / "inner.txt").resolve()


def test_rejects_absolute_path_outside_sandbox(sandbox):
    with pytest.raises(HTTPException) as excinfo:
        resolve_sandboxed_path("/etc/passwd", base_dir=sandbox)
    assert excinfo.value.status_code == 403


def test_rejects_dot_dot_traversal(sandbox):
    with pytest.raises(HTTPException) as excinfo:
        resolve_sandboxed_path("../outside.txt", base_dir=sandbox)
    assert excinfo.value.status_code == 403


def test_rejects_deep_dot_dot_traversal(sandbox):
    with pytest.raises(HTTPException) as excinfo:
        resolve_sandboxed_path("nested/../../../../etc/passwd", base_dir=sandbox)
    assert excinfo.value.status_code == 403


def test_rejects_absolute_path_that_happens_to_share_prefix(tmp_path):
    base = tmp_path / "sandbox"
    base.mkdir()
    sibling = tmp_path / "sandbox-evil"
    sibling.mkdir()
    (sibling / "secret.txt").write_text("nope")

    with pytest.raises(HTTPException) as excinfo:
        resolve_sandboxed_path(str(sibling / "secret.txt"), base_dir=base)
    assert excinfo.value.status_code == 403


def test_rejects_symlink_escape(sandbox, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    link = sandbox / "escape"
    link.symlink_to(outside)

    with pytest.raises(HTTPException) as excinfo:
        resolve_sandboxed_path("escape", base_dir=sandbox)
    assert excinfo.value.status_code == 403


def test_get_sandbox_dir_creates_directory_and_respects_env(tmp_path, monkeypatch):
    target = tmp_path / "custom_sandbox"
    monkeypatch.setenv("MULTIMODAL_SANDBOX_DIR", str(target))

    resolved = get_sandbox_dir()

    assert resolved == target.resolve()
    assert os.path.isdir(resolved)
