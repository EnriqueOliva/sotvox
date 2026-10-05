"""Runs the headless batch mode as a real subprocess.

Set SOTVOX_EXE to a frozen build (e.g. installer/dist_app/Sotvox.app/Contents/MacOS/Sotvox)
to run the same checks against the packaged app instead of the source tree.
"""
import os
import shutil
import subprocess
import sys

import pytest

from conftest import PROJECT_ROOT


def _command():
    frozen = os.environ.get("SOTVOX_EXE")
    if frozen:
        return [frozen]
    return [sys.executable, os.path.join(PROJECT_ROOT, "src", "main.py")]


@pytest.fixture
def fake_home(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ)
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    env["LOCALAPPDATA"] = str(home)
    env["HF_HOME"] = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    return home, env


def _log_dir(home):
    if sys.platform == "darwin":
        return home / "Library" / "Logs" / "Sotvox"
    return home / "Sotvox" / "logs"


def _run(env, *arguments):
    return subprocess.run(_command() + list(arguments), env=env, capture_output=True, text=True, timeout=600)


def test_batch_transcribes_new_files_and_skips_done_ones(media_dir, tmp_path, fake_home):
    home, env = fake_home
    inbox = tmp_path / "inbox"
    output = tmp_path / "out"
    inbox.mkdir()
    for name in ("english.mp3", "spanish.m4a", "video.mp4", "silence.wav"):
        shutil.copy(media_dir / name, inbox / name)
    (inbox / "notes.pdf").write_text("not media")

    result = _run(env, "--transcribe", str(inbox), "--output", str(output),
                  "--model", "tiny", "--device", "CPU")
    assert result.returncode == 0, result.stderr

    assert sorted(path.name for path in output.iterdir()) == ["english.txt", "spanish.txt", "video.txt"]
    assert "fox" in (output / "english.txt").read_text(encoding="utf-8").lower()

    log = (_log_dir(home) / "batch.txt").read_text(encoding="utf-8")
    assert "3 transcribed, 0 failed" in log
    assert "silence.wav: no speech detected" in log
    assert "Loading model tiny on cpu" in log

    english_written = (output / "english.txt").stat().st_mtime_ns
    (output / "spanish.txt").unlink()
    result = _run(env, "--transcribe", str(inbox), "--output", str(output), "--model", "tiny")
    assert result.returncode == 0, result.stderr
    assert (output / "spanish.txt").exists()
    assert (output / "english.txt").stat().st_mtime_ns == english_written
    log = (_log_dir(home) / "batch.txt").read_text(encoding="utf-8")
    assert "2 new file(s) to transcribe" in log  # spanish.m4a + silence.wav, which never gets a transcript


def test_batch_with_nothing_new(tmp_path, fake_home):
    home, env = fake_home
    empty = tmp_path / "empty"
    empty.mkdir()
    result = _run(env, "--transcribe", str(empty), "--output", str(tmp_path / "out"))
    assert result.returncode == 0
    assert "Nothing new to transcribe" in (_log_dir(home) / "batch.txt").read_text(encoding="utf-8")


def test_batch_missing_folder_exits_2(tmp_path, fake_home):
    home, env = fake_home
    result = _run(env, "--transcribe", str(tmp_path / "missing"))
    assert result.returncode == 2
    assert "input folder not found" in (_log_dir(home) / "batch.txt").read_text(encoding="utf-8")


def test_batch_reports_failures_with_exit_1(media_dir, tmp_path, fake_home):
    home, env = fake_home
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    shutil.copy(media_dir / "english.wav", inbox / "good.wav")
    (inbox / "broken.mp3").write_bytes(b"this is not an mp3")
    result = _run(env, "--transcribe", str(inbox), "--output", str(tmp_path / "out"), "--model", "tiny")
    assert result.returncode == 1
    log = (_log_dir(home) / "batch.txt").read_text(encoding="utf-8")
    assert "broken.mp3: FAILED" in log
    assert "1 transcribed, 1 failed" in log


def test_selftest_passes(media_dir, fake_home):
    home, env = fake_home
    result = _run(env, "--selftest", str(media_dir / "video.mp4"))
    assert result.returncode == 0
    report = (_log_dir(home) / "selftest.txt").read_text(encoding="utf-8")
    assert "CPU RESULT: PASS" in report
    assert "RESULT: PASS" in report.splitlines()[-1]
