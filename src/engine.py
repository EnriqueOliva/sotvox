import collections
import math
import os
import re
import time
from bisect import bisect_left

import av
import numpy as np
from faster_whisper.audio import decode_audio

SAMPLING_RATE = 16000
CHUNK_SECONDS = 30
MICROSECONDS_PER_SECOND = 1_000_000
TRANSCRIPT_STAMP_FORMAT = "%Y-%m-%d %H-%M-%S"
FIRST_DUPLICATE_NUMBER = 2
RETRY_TARGET_PEAK = 0.5
RETRY_MAXIMUM_GAIN = 1000.0
PCM_16_BIT_SCALE = 32768.0
FIRST_TRACK_NUMBER = 1
LONGEST_SEGMENT_SECONDS = 30.0
ECHO_MINIMUM_OVERLAP = 0.5
ECHO_MINIMUM_SIMILARITY = 0.6
PARAGRAPH_SECONDS = 60.0
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
NO_CONFIDENCE = 0.0

TimedText = collections.namedtuple("TimedText", "start end text confidence")
Utterance = collections.namedtuple("Utterance", "start end label text confidence")


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


def audio_samples(audio):
    if isinstance(audio, np.ndarray):
        return audio
    else:
        return decode_audio(audio, sampling_rate=SAMPLING_RATE)


def normalize_volume(samples):
    loudest = float(np.max(np.abs(samples))) if len(samples) else 0.0
    if loudest == 0.0:
        return samples
    else:
        gain = min(RETRY_TARGET_PEAK / loudest, RETRY_MAXIMUM_GAIN)
        return (samples * gain).astype(np.float32)


def _timed_text(segment):
    return TimedText(segment.start, segment.end, segment.text.strip(), getattr(segment, "avg_logprob", NO_CONFIDENCE))


def transcribe_audio_segments(model, audio, lang_code, on_progress=None, is_cancelled=None):
    start_time = time.time()

    segments, info = model.transcribe(
        audio, language=lang_code, beam_size=5,
        vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500)
    )

    duration = info.duration
    max_time = max(30, duration * 5)
    timed_parts = []
    timed_out = False

    for segment in segments:
        if is_cancelled and is_cancelled():
            break
        if time.time() - start_time > max_time:
            timed_out = True
            break
        timed_parts.append(_timed_text(segment))
        if on_progress and duration > 0:
            on_progress(segment.end, duration)

    if timed_out:
        return timed_parts, info, "timed_out"

    if not timed_parts and not (is_cancelled and is_cancelled()):
        retry_audio = normalize_volume(audio_samples(audio))
        retry_start = time.time()
        retry_max = max(15, duration * 3)
        segments, info = model.transcribe(
            retry_audio, language=lang_code, beam_size=5,
            vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=False
        )
        for segment in segments:
            if is_cancelled and is_cancelled():
                break
            if time.time() - retry_start > retry_max:
                return timed_parts, info, "retry_timed_out"
            timed_parts.append(_timed_text(segment))
            if on_progress and duration > 0:
                on_progress(segment.end, duration)

    return timed_parts, info, "ok"


def transcribe_audio(model, audio_path, lang_code, on_progress=None, is_cancelled=None):
    timed_parts, info, status = transcribe_audio_segments(
        model, audio_path, lang_code, on_progress=on_progress, is_cancelled=is_cancelled)
    return [part.text for part in timed_parts], info, status


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


def _unused_transcript_path(output_dir, source_base, filepath):
    try:
        stamp_source = os.path.getmtime(filepath)
    except OSError:
        stamp_source = time.time()
    stamped_base = f"{source_base} {time.strftime(TRANSCRIPT_STAMP_FORMAT, time.localtime(stamp_source))}"
    candidate = os.path.join(output_dir, stamped_base + ".txt")
    copy_number = FIRST_DUPLICATE_NUMBER
    while os.path.exists(candidate):
        candidate = os.path.join(output_dir, f"{stamped_base} ({copy_number}).txt")
        copy_number += 1
    return candidate


def save_transcript(filepath, full_text, output_dir, keep_existing=False):
    source_base = os.path.splitext(os.path.basename(filepath))[0]
    out_path = os.path.join(output_dir, source_base + ".txt")
    if keep_existing and os.path.exists(out_path):
        out_path = _unused_transcript_path(output_dir, source_base, filepath)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    return os.path.basename(out_path)


def audio_stream_count(filepath):
    try:
        with av.open(filepath) as container:
            return len(container.streams.audio)
    except Exception:
        return 0


def decode_audio_stream(filepath, stream_position):
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=SAMPLING_RATE)
    chunks = []
    with av.open(filepath, metadata_errors="ignore") as container:
        stream = container.streams.audio[stream_position]
        for packet in container.demux(stream):
            try:
                frames = packet.decode()
            except av.error.InvalidDataError:
                continue
            for frame in frames:
                chunks.extend(resampled.to_ndarray().reshape(-1) for resampled in resampler.resample(frame))
        chunks.extend(resampled.to_ndarray().reshape(-1) for resampled in resampler.resample(None))
    if chunks:
        return np.concatenate(chunks).astype(np.float32) / PCM_16_BIT_SCALE
    else:
        return np.zeros(0, dtype=np.float32)


def parse_speaker_tracks(raw_tracks):
    speaker_tracks = []
    if isinstance(raw_tracks, dict):
        for raw_number, raw_label in raw_tracks.items():
            try:
                track_number = int(raw_number)
            except (TypeError, ValueError):
                continue
            if track_number >= FIRST_TRACK_NUMBER and isinstance(raw_label, str) and raw_label.strip():
                speaker_tracks.append((track_number, raw_label.strip()))
    return sorted(speaker_tracks)


def speaker_tracks_for(filepath, speaker_tracks):
    if not speaker_tracks:
        return []
    else:
        available_tracks = audio_stream_count(filepath)
        if all(track_number <= available_tracks for track_number, _label in speaker_tracks):
            return speaker_tracks
        else:
            return []


def _overlap_ratio(first, second):
    shared_seconds = min(first.end, second.end) - max(first.start, second.start)
    shortest_seconds = min(first.end - first.start, second.end - second.start)
    if shared_seconds <= 0 or shortest_seconds <= 0:
        return 0.0
    else:
        return shared_seconds / shortest_seconds


def _word_similarity(first_text, second_text):
    first_words = collections.Counter(re.findall(r"\w+", first_text.lower()))
    second_words = collections.Counter(re.findall(r"\w+", second_text.lower()))
    total_words = sum(first_words.values()) + sum(second_words.values())
    if total_words == 0:
        return 0.0
    else:
        return 2.0 * sum((first_words & second_words).values()) / total_words


def without_echoes(utterances):
    ordered = sorted(utterances, key=lambda utterance: utterance.start)
    starts = [utterance.start for utterance in ordered]
    kept = []
    for utterance in ordered:
        window_start = bisect_left(starts, utterance.start - LONGEST_SEGMENT_SECONDS)
        window_end = bisect_left(starts, utterance.end)
        has_clearer_copy = any(
            other.label != utterance.label
            and other.confidence > utterance.confidence
            and _overlap_ratio(utterance, other) >= ECHO_MINIMUM_OVERLAP
            and _word_similarity(utterance.text, other.text) >= ECHO_MINIMUM_SIMILARITY
            for other in ordered[window_start:window_end]
        )
        if not has_clearer_copy:
            kept.append(utterance)
    return kept


def format_timestamp(seconds):
    hours, remainder = divmod(int(seconds), SECONDS_PER_HOUR)
    minutes, remaining_seconds = divmod(remainder, SECONDS_PER_MINUTE)
    if hours:
        return f"{hours}:{minutes:02d}:{remaining_seconds:02d}"
    else:
        return f"{minutes:02d}:{remaining_seconds:02d}"


def _paragraph_line(paragraph):
    spoken = " ".join(utterance.text for utterance in paragraph)
    return f"[{format_timestamp(paragraph[0].start)}] {paragraph[0].label}: {spoken}"


def conversation_lines(utterances):
    lines = []
    paragraph = []
    for utterance in sorted(utterances, key=lambda utterance: (utterance.start, utterance.label)):
        speaker_changed = bool(paragraph) and utterance.label != paragraph[0].label
        paragraph_is_long = bool(paragraph) and utterance.start - paragraph[0].start >= PARAGRAPH_SECONDS
        if speaker_changed or paragraph_is_long:
            lines.append(_paragraph_line(paragraph))
            paragraph = []
        paragraph.append(utterance)
    if paragraph:
        lines.append(_paragraph_line(paragraph))
    return lines


def transcribe_speaker_tracks(model, filepath, lang_code, speaker_tracks, on_progress=None, is_cancelled=None):
    utterances = []
    languages = {}
    probability_sum = 0.0
    duration = 0.0
    duration_after_vad = 0.0
    status = "ok"
    track_count = len(speaker_tracks)
    for position, (track_number, label) in enumerate(speaker_tracks):
        if is_cancelled and is_cancelled():
            break
        track_progress = None
        if on_progress:
            def track_progress(segment_end, track_duration, position=position):
                on_progress((position * track_duration + segment_end) / track_count, track_duration)
        samples = decode_audio_stream(filepath, track_number - FIRST_TRACK_NUMBER)
        timed_parts, info, track_status = transcribe_audio_segments(
            model, samples, lang_code, on_progress=track_progress, is_cancelled=is_cancelled)
        utterances.extend(
            Utterance(part.start, part.end, label, part.text, part.confidence) for part in timed_parts if part.text)
        languages[info.language] = languages.get(info.language, 0) + 1
        probability_sum += info.language_probability
        duration = max(duration, info.duration)
        duration_after_vad += getattr(info, "duration_after_vad", 0.0) or 0.0
        if track_status != "ok":
            status = track_status
    tracks_transcribed = sum(languages.values())
    average_probability = probability_sum / tracks_transcribed if tracks_transcribed else 0.0
    info_combined = MultilingualInfo(duration, duration_after_vad, languages, average_probability)
    return conversation_lines(without_echoes(utterances)), info_combined, status


def parse_replacements(raw_replacements):
    replacements = {}
    if isinstance(raw_replacements, dict):
        for variant, replacement in raw_replacements.items():
            if isinstance(variant, str) and isinstance(replacement, str) and variant.strip():
                replacements[" ".join(variant.lower().split())] = replacement
    return replacements


def apply_replacements(text, replacements):
    if not replacements:
        return text
    else:
        variants = sorted(replacements, key=len, reverse=True)
        alternatives = "|".join(r"\s+".join(re.escape(word) for word in variant.split()) for variant in variants)
        pattern = re.compile(rf"\b(?:{alternatives})\b", re.IGNORECASE)
        return pattern.sub(lambda match: replacements[" ".join(match.group(0).lower().split())], text)
