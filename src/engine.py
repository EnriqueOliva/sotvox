import math
import os
import time

import av
from faster_whisper.audio import decode_audio

SAMPLING_RATE = 16000
CHUNK_SECONDS = 30
MICROSECONDS_PER_SECOND = 1_000_000


def probe_file(filepath):
    try:
        with av.open(filepath) as container:
            if container.duration:
                duration_seconds = f"{container.duration / MICROSECONDS_PER_SECOND:.3f}"
            else:
                duration_seconds = "unknown"

            streams = []
            for stream in container.streams:
                codec_context = stream.codec_context
                stream_info = {
                    "codec_type": stream.type,
                    "codec_name": getattr(codec_context, "name", "unknown"),
                }
                if stream.type == "video":
                    stream_info["width"] = codec_context.width
                    stream_info["height"] = codec_context.height
                    if stream.average_rate:
                        stream_info["r_frame_rate"] = f"{float(stream.average_rate):.2f}"
                elif stream.type == "audio":
                    stream_info["sample_rate"] = codec_context.sample_rate
                    stream_info["channels"] = codec_context.channels
                streams.append(stream_info)

            return {
                "format": {
                    "format_long_name": container.format.long_name,
                    "duration": duration_seconds,
                    "bit_rate": container.bit_rate,
                },
                "streams": streams,
            }
    except Exception:
        return None


def has_audio_stream(filepath):
    try:
        with av.open(filepath) as container:
            return any(stream.type == "audio" for stream in container.streams)
    except Exception:
        return False


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
