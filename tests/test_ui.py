import os
import shutil
import sys
import subprocess
import time
import tkinter as tk
from types import SimpleNamespace

import pytest

import ui

on_mac = pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "DEFAULT_OUTPUT_DIR", str(tmp_path / "transcripts"))
    monkeypatch.setattr(ui, "LOG_DIR", str(tmp_path / "logs"))
    launched = []
    monkeypatch.setattr(ui, "subprocess", SimpleNamespace(
        Popen=lambda args, **kwargs: launched.append(args), run=subprocess.run, DEVNULL=subprocess.DEVNULL))
    try:
        instance = ui.SotvoxApp()
    except tk.TclError as error:
        pytest.skip(f"no display: {error}")
    instance.launched = launched
    _pump(instance, 0.5)
    yield instance
    try:
        instance._on_close()
    except tk.TclError:
        pass


def _pump(app, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        app.root.update()
        time.sleep(0.01)


def _wait_until(app, condition, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.root.update()
        if condition():
            return True
        time.sleep(0.05)
    return False


def _all_widgets(widget):
    yield widget
    for child in widget.winfo_children():
        yield from _all_widgets(child)


def test_window_opens_with_every_section(app):
    assert app.root.winfo_viewable()
    assert app.root.title() == "Sotvox"
    assert app.root.TkdndVersion
    texts = {w.cget("text") for w in _all_widgets(app.root) if isinstance(w, tk.Label)}
    for expected in (" Files ", " Options ", " Output ", " Progress ", " Log ",
                     "Add Files...", "Remove", "Clear All", "Transcribe", "Open Output Folder", "Browse..."):
        assert expected in texts


def test_no_label_text_is_clipped(app):
    clipped = []
    for widget in _all_widgets(app.root):
        if isinstance(widget, tk.Label) and widget.winfo_ismapped() and widget.cget("text"):
            if widget.winfo_width() + 1 < widget.winfo_reqwidth():
                clipped.append((widget.cget("text"), widget.winfo_width(), widget.winfo_reqwidth()))
    assert not clipped


@on_mac
def test_mac_window_uses_native_frame_and_cpu_only(app):
    assert not hasattr(app, "titlebar")
    assert app.root.overrideredirect() in ("", 0, False, None)
    assert "not available on macOS" in app.gpu_status_label.cget("text")
    assert not app.gpu_button.winfo_ismapped()
    assert app._resolve_device() == ("cpu", "int8")
    assert app.root.tk.call("info", "commands", "::tk::mac::Quit")
    assert app.root.tk.call("info", "commands", "::tkAboutDialog")


def test_drop_parses_paths_with_spaces(app, media_dir, tmp_path):
    spaced = tmp_path / "my recordings"
    spaced.mkdir()
    shutil.copy(media_dir / "english.mp3", spaced / "team meeting.mp3")
    data = f"{{{spaced / 'team meeting.mp3'}}} {media_dir / 'video.mp4'} {tmp_path / 'notes.pdf'}"
    app._on_drop(type("Event", (), {"data": data})())
    assert app.files == [str(spaced / "team meeting.mp3"), str(media_dir / "video.mp4")]
    assert app.file_listbox.get(0, "end") == (" [AUD]  team meeting.mp3", " [VID]  video.mp4")
    assert app.status_files.cget("text") == "2 files"

    app._on_drop(type("Event", (), {"data": str(media_dir / "video.mp4")})())
    assert len(app.files) == 2

    app.file_listbox.selection_set(0)
    app._remove_selected()
    assert app.files == [str(media_dir / "video.mp4")]
    app._clear_files()
    assert app.files == [] and app.status_files.cget("text") == "0 files"


def test_full_transcription_through_the_ui(app, media_dir, tmp_path):
    output = tmp_path / "out"
    app.output_var.set(str(output))
    app.model_var.set("tiny")
    app.language_var.set("Auto-detect")
    app._add_files([str(media_dir / name) for name in
                    ("english.mp3", "spanish.m4a", "video.mp4", "no_audio.mp4", "silence.wav")])

    app._toggle_transcription()
    assert app.is_transcribing
    assert _wait_until(app, lambda: not app.is_transcribing, timeout=300)
    _pump(app, 0.3)

    assert sorted(path.name for path in output.iterdir()) == ["english.txt", "spanish.txt", "video.txt"]
    log = app.log_text.get("1.0", "end")
    assert "Model 'tiny' loaded on CPU" in log
    assert "Video has no audio track" in log
    assert "No speech detected" in log
    assert app.progress_label.cget("text") == "Done — 3 transcribed, 1 failed"
    assert app.progress_pct.cget("text") == "100%"
    assert app.transcribe_btn.winfo_children()
    assert ["afplay", ui.SOUND_SUCCESS] in app.launched


def test_cancel_stops_the_queue(app, media_dir, tmp_path):
    output = tmp_path / "out"
    app.output_var.set(str(output))
    app.model_var.set("tiny")
    app._add_files([str(media_dir / "mixed.wav"), str(media_dir / "english.wav")])
    app._toggle_transcription()
    assert _wait_until(app, lambda: "Processing: mixed.wav" in app.log_text.get("1.0", "end"), timeout=120)
    app._toggle_transcription()
    assert app.cancel_requested
    assert _wait_until(app, lambda: not app.is_transcribing, timeout=120)
    _pump(app, 0.3)
    assert not (output / "english.txt").exists()


def test_open_output_folder(app, tmp_path):
    app.output_var.set(str(tmp_path / "new folder"))
    app._open_output()
    assert os.path.isdir(tmp_path / "new folder")
    if sys.platform == "darwin":
        assert app.launched[-1] == ["open", str(tmp_path / "new folder")]


def test_dialogs_open_and_close(app):
    before = set(app.root.winfo_children())
    app._about()
    app._message_dialog("GPU Acceleration", "Test message")
    _pump(app, 0.2)
    dialogs = [w for w in app.root.winfo_children() if isinstance(w, tk.Toplevel) and w not in before]
    assert len(dialogs) == 2
    if sys.platform == "darwin":
        assert {d.title() for d in dialogs} == {"About Sotvox", "GPU Acceleration"}
    for dialog in dialogs:
        dialog.destroy()


def test_session_log_is_written(app, tmp_path):
    app._on_close()
    logs = list((tmp_path / "logs").glob("*.txt"))
    assert len(logs) == 1
    content = logs[0].read_text(encoding="utf-8")
    assert "SESSION STARTED" in content
    assert "RAM:" in content
    assert "Application closed" in content
