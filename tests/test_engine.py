import pytest

from conftest import AUDIO_FORMATS
from engine import (
    has_audio_stream, probe_file, save_transcript, transcribe_audio, transcribe_audio_multilingual,
)

ENGLISH_WORDS = ("fox", "dog", "speech")
SPANISH_WORDS = ("hola", "días", "madrid", "comida")


def _matches(text, words):
    lowered = text.lower()
    return sum(word in lowered for word in words)


@pytest.mark.parametrize("name", [entry[0] for entry in AUDIO_FORMATS] + ["video.mp4"])
def test_transcribes_every_format(media_dir, tiny_model, name):
    text_parts, info, status = transcribe_audio(tiny_model, str(media_dir / name), "en")
    text = " ".join(text_parts)
    assert status == "ok"
    assert info.duration == pytest.approx(5, abs=1)
    assert _matches(text, ENGLISH_WORDS) >= 2, text


@pytest.mark.parametrize("name", ["spanish.m4a", "video.mkv"])
def test_auto_detects_spanish(media_dir, tiny_model, name):
    text_parts, info, status = transcribe_audio(tiny_model, str(media_dir / name), None)
    assert status == "ok"
    assert info.language == "es"
    assert _matches(" ".join(text_parts), SPANISH_WORDS) >= 2


def test_progress_and_cancel(media_dir, tiny_model):
    progress = []
    transcribe_audio(tiny_model, str(media_dir / "english.wav"), "en",
                     on_progress=lambda end, total: progress.append((end, total)))
    assert progress and all(0 < end <= total + 0.5 for end, total in progress)

    text_parts, _, _ = transcribe_audio(tiny_model, str(media_dir / "english.wav"), "en",
                                        is_cancelled=lambda: True)
    assert text_parts == []


def test_silence_produces_no_text(media_dir, tiny_model):
    # Auto-detect, as the app defaults to. With a forced language the no-VAD retry can hallucinate a word.
    text_parts, _, status = transcribe_audio(tiny_model, str(media_dir / "silence.wav"), None)
    assert status == "ok"
    assert not "".join(text_parts).strip()


def test_multilingual_detects_both_languages(media_dir, tiny_model):
    chunks = []
    text_parts, info, status = transcribe_audio_multilingual(
        tiny_model, str(media_dir / "mixed.wav"),
        on_chunk=lambda number, total, start, end, language, probability: chunks.append(language))
    assert status == "ok"
    assert len(chunks) == 3
    assert chunks[0] == "en" and chunks[-1] == "es"
    assert set(info.languages) >= {"en", "es"}
    assert info.language == "+".join(sorted(info.languages))


def test_probe_and_audio_stream_detection(media_dir):
    assert has_audio_stream(str(media_dir / "video.mp4"))
    assert not has_audio_stream(str(media_dir / "no_audio.mp4"))
    assert not has_audio_stream(str(media_dir / "does_not_exist.mp4"))

    probe = probe_file(str(media_dir / "video.mp4"))
    kinds = {stream["codec_type"] for stream in probe["streams"]}
    assert kinds == {"video", "audio"}
    assert probe_file(str(media_dir / "does_not_exist.mp4")) is None


def test_save_transcript_is_utf8(tmp_path):
    saved = save_transcript("/some/folder/entrevista año.m4a", "Mañana, ¿qué tal? 日本語", str(tmp_path))
    assert saved == "entrevista año.txt"
    assert (tmp_path / saved).read_text(encoding="utf-8") == "Mañana, ¿qué tal? 日本語"
