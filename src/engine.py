import json
import math
import os
import subprocess
import time

from faster_whisper.audio import decode_audio

from constants import VIDEO_EXTENSIONS

SAMPLING_RATE = 16000
CHUNK_SECONDS = 30


def probe_file(filepath):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams",
         "-print_format", "json", filepath],
        capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW
    )
    if result.returncode != 0 or not result.stdout:
        return None
    try:
        return json.loads(result.stdout)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def _has_audio_stream(filepath):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", filepath],
        capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW
    )
    return "audio" in (result.stdout or "")


def extract_audio(filepath, output_dir, log=None):
    if not _has_audio_stream(filepath):
        raise RuntimeError("Video has no audio track — nothing to transcribe")

    name = os.path.basename(filepath)
    temp_audio = os.path.join(output_dir, f"_temp_{os.path.splitext(name)[0]}.wav")
    cmd = ["ffmpeg", "-y", "-i", filepath, "-vn", "-acodec", "pcm_s16le",
           "-ar", "16000", "-ac", "1", temp_audio]

    if log:
        log(f"  FFmpeg command: {' '.join(cmd)}")

    result = subprocess.run(
        cmd, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW
    )

    stderr = result.stderr or ""

    if log:
        log(f"  FFmpeg exit code: {result.returncode}")
        if stderr.strip():
            for line in stderr.strip().splitlines():
                log(f"  FFmpeg | {line}")

    if result.returncode != 0:
        stderr_lines = [l for l in stderr.splitlines() if not l.startswith(("ffmpeg version", "  built with", "  configuration:", "  lib"))]
        raise RuntimeError(f"FFmpeg error: {' '.join(stderr_lines[-5:])}")

    if log and os.path.exists(temp_audio):
        size_mb = os.path.getsize(temp_audio) / (1024 * 1024)
        log(f"  Temp audio created: {size_mb:.1f} MB")

    return temp_audio


def transcribe_audio(model, audio_path, lang_code, on_progress=None, is_cancelled=None):
    start_time = time.time()

    segments, info = model.transcribe(
        audio_path, language=lang_code, beam_size=5,
        vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500)
    )

    duration = info.duration
    max_time = max(30, duration * 5)
    text_parts = []
    timed_out = False

    for segment in segments:
        if is_cancelled and is_cancelled():
            break
        if time.time() - start_time > max_time:
            timed_out = True
            break
        text_parts.append(segment.text.strip())
        if on_progress and duration > 0:
            on_progress(segment.end, duration)

    if timed_out:
        return text_parts, info, "timed_out"

    if not text_parts and not (is_cancelled and is_cancelled()):
        retry_start = time.time()
        retry_max = max(15, duration * 3)
        segments, info = model.transcribe(
            audio_path, language=lang_code, beam_size=5,
            vad_filter=False, condition_on_previous_text=False
        )
        for segment in segments:
            if is_cancelled and is_cancelled():
                break
            if time.time() - retry_start > retry_max:
                return text_parts, info, "retry_timed_out"
            text_parts.append(segment.text.strip())
            if on_progress and duration > 0:
                on_progress(segment.end, duration)

    return text_parts, info, "ok"


class MultilingualInfo:
    def __init__(self, duration, duration_after_vad, languages, average_probability):
        self.duration = duration
        self.duration_after_vad = duration_after_vad
        self.languages = languages
        if languages:
            self.language = "+".join(sorted(languages.keys()))
        else:
            self.language = "unknown"
        self.language_probability = average_probability


def transcribe_audio_multilingual(model, audio_path, on_progress=None, is_cancelled=None, on_chunk=None):
    start_time = time.time()

    audio = decode_audio(audio_path, sampling_rate=SAMPLING_RATE)
    total_samples = len(audio)
    duration = total_samples / SAMPLING_RATE
    chunk_samples = CHUNK_SECONDS * SAMPLING_RATE

    max_time = max(30, duration * 8)
    text_parts = []
    languages = {}
    duration_after_vad_total = 0.0
    probability_sum = 0.0
    detection_count = 0
    timed_out = False

    num_chunks = max(1, math.ceil(total_samples / chunk_samples))

    for chunk_index in range(num_chunks):
        if is_cancelled and is_cancelled():
            break
        if time.time() - start_time > max_time:
            timed_out = True
            break

        chunk_start_sample = chunk_index * chunk_samples
        chunk_end_sample = min(chunk_start_sample + chunk_samples, total_samples)
        chunk_audio = audio[chunk_start_sample:chunk_end_sample]
        chunk_start_seconds = chunk_start_sample / SAMPLING_RATE
        chunk_end_seconds = chunk_end_sample / SAMPLING_RATE

        segments, info = model.transcribe(
            chunk_audio, language=None, beam_size=5,
            vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500)
        )

        for segment in segments:
            if is_cancelled and is_cancelled():
                break
            text_parts.append(segment.text.strip())

        detected_language = info.language
        languages[detected_language] = languages.get(detected_language, 0) + 1
        probability_sum += info.language_probability
        detection_count += 1
        if hasattr(info, "duration_after_vad") and info.duration_after_vad:
            duration_after_vad_total += info.duration_after_vad

        if on_chunk:
            on_chunk(chunk_index + 1, num_chunks, chunk_start_seconds, chunk_end_seconds,
                     detected_language, info.language_probability)

        if on_progress and duration > 0:
            on_progress(chunk_end_seconds, duration)

    if detection_count > 0:
        average_probability = probability_sum / detection_count
    else:
        average_probability = 0.0

    info_combined = MultilingualInfo(duration, duration_after_vad_total, languages, average_probability)

    if timed_out:
        return text_parts, info_combined, "timed_out"

    return text_parts, info_combined, "ok"


def save_transcript(filepath, full_text, output_dir):
    source_base = os.path.splitext(os.path.basename(filepath))[0]
    out_path = os.path.join(output_dir, source_base + ".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    return os.path.basename(out_path)
