import ctypes
import os
import sys
import time

DPI_AWARENESS_SYSTEM = 1

if not getattr(sys, "frozen", False):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from constants import CUDA_DIR, IS_FROZEN, RESOURCE_DIR


def _enable_dpi_awareness():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(DPI_AWARENESS_SYSTEM)
    except Exception:
        pass


def _register_cuda_directories():
    candidate_dirs = [CUDA_DIR]
    if not IS_FROZEN:
        site_packages_nvidia = os.path.join(RESOURCE_DIR, ".venv", "Lib", "site-packages", "nvidia")
        candidate_dirs.append(os.path.join(site_packages_nvidia, "cublas", "bin"))
        candidate_dirs.append(os.path.join(site_packages_nvidia, "cudnn", "bin"))

    for directory in candidate_dirs:
        if os.path.isdir(directory):
            try:
                os.add_dll_directory(directory)
            except (OSError, AttributeError):
                pass
            current_path = os.environ.get("PATH", "")
            if directory not in current_path:
                os.environ["PATH"] = directory + os.pathsep + current_path


_enable_dpi_awareness()
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
_register_cuda_directories()

SELFTEST_FLAG = "--selftest"
TRANSCRIBE_FLAG = "--transcribe"
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BAD_USAGE = 2


def _argument_value(flag, default=None):
    if flag in sys.argv:
        index = sys.argv.index(flag)
        if len(sys.argv) > index + 1:
            return sys.argv[index + 1]
    return default


def _run_batch_transcribe():
    """Headless batch mode used by scheduled automation.

    Transcribes every supported media file in a folder that does not already
    have a transcript, and writes a report to the log directory.
    """
    import traceback
    from datetime import datetime

    from constants import LOG_DIR, SUPPORTED_EXTENSIONS, DEFAULT_OUTPUT_DIR, LANG_MAP
    from engine import save_transcript, transcribe_audio

    input_dir = _argument_value(TRANSCRIBE_FLAG)
    output_dir = _argument_value("--output", DEFAULT_OUTPUT_DIR)
    model_name = _argument_value("--model", "large-v3-turbo")
    language_name = _argument_value("--language", "Auto-detect")
    device_choice = _argument_value("--device", "Auto")

    os.makedirs(LOG_DIR, exist_ok=True)
    report_path = os.path.join(LOG_DIR, "batch.txt")
    report = open(report_path, "a", encoding="utf-8")

    def log(message):
        stamped = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        report.write(stamped + "\n")
        report.flush()

    log(f"Batch started: input={input_dir} output={output_dir} model={model_name}")

    if not input_dir or not os.path.isdir(input_dir):
        log(f"ERROR: input folder not found: {input_dir}")
        report.close()
        return EXIT_BAD_USAGE

    os.makedirs(output_dir, exist_ok=True)

    pending = []
    for name in sorted(os.listdir(input_dir)):
        source_path = os.path.join(input_dir, name)
        extension = os.path.splitext(name)[1].lower()
        if not os.path.isfile(source_path) or extension not in SUPPORTED_EXTENSIONS:
            continue
        transcript_path = os.path.join(output_dir, os.path.splitext(name)[0] + ".txt")
        if os.path.exists(transcript_path):
            continue
        pending.append(source_path)

    if not pending:
        log("Nothing new to transcribe")
        report.close()
        return EXIT_OK

    log(f"{len(pending)} new file(s) to transcribe")

    try:
        from faster_whisper import WhisperModel

        if device_choice == "CPU":
            device, compute_type = "cpu", "int8"
        else:
            try:
                import ctranslate2
                has_gpu = ctranslate2.get_cuda_device_count() > 0
            except Exception:
                has_gpu = False
            if has_gpu and device_choice != "CPU":
                device, compute_type = "cuda", "float16"
            else:
                device, compute_type = "cpu", "int8"

        log(f"Loading model {model_name} on {device}")
        try:
            model = WhisperModel(model_name, device=device, compute_type=compute_type)
        except Exception:
            if device != "cuda":
                raise
            log("CUDA failed, falling back to CPU")
            device, compute_type = "cpu", "int8"
            model = WhisperModel(model_name, device=device, compute_type=compute_type)

        language_code = None if language_name == "Auto-detect" else LANG_MAP.get(language_name)
        succeeded, failed = 0, 0

        for source_path in pending:
            name = os.path.basename(source_path)
            started = time.time()
            try:
                text_parts, info, status = transcribe_audio(model, source_path, language_code)
                full_text = "\n".join(text_parts)
                if not full_text.strip():
                    log(f"  {name}: no speech detected, skipped")
                    continue
                saved_name = save_transcript(source_path, full_text, output_dir)
                elapsed = time.time() - started
                log(f"  {name}: {info.language}, {info.duration:.0f}s audio in {elapsed:.0f}s -> {saved_name}")
                succeeded += 1
            except Exception as error:
                log(f"  {name}: FAILED {error}")
                log(traceback.format_exc())
                failed += 1

        log(f"Batch finished: {succeeded} transcribed, {failed} failed")
        report.close()
        return EXIT_OK if failed == 0 else EXIT_FAILED
    except Exception as error:
        log(f"FATAL: {error}")
        log(traceback.format_exc())
        report.close()
        return EXIT_FAILED


def _run_selftest(media_path):
    import traceback
    from constants import LOG_DIR
    from engine import has_audio_stream, probe_file

    report_lines = [f"Sotvox self-test", f"Media: {media_path}", ""]
    try:
        import av
        import ctranslate2
        import faster_whisper
        report_lines.append(f"PyAV: {av.__version__}")
        report_lines.append(f"faster-whisper: {faster_whisper.__version__}")
        report_lines.append(f"CTranslate2: {ctranslate2.__version__}")
        report_lines.append(f"CUDA devices: {ctranslate2.get_cuda_device_count()}")

        report_lines.append(f"has_audio_stream: {has_audio_stream(media_path)}")
        probe = probe_file(media_path)
        report_lines.append(f"probe: {probe['format'] if probe else None}")

        model = faster_whisper.WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, info = model.transcribe(media_path, beam_size=5, vad_filter=True)
        text = " ".join(segment.text.strip() for segment in segments)
        report_lines.append(f"detected language: {info.language}")
        report_lines.append(f"duration: {info.duration:.2f}s")
        report_lines.append(f"transcript: {text}")
        report_lines.append("CPU RESULT: PASS")

        import gpu_pack
        if gpu_pack.libraries_available() and ctranslate2.get_cuda_device_count() > 0:
            gpu_model = faster_whisper.WhisperModel("tiny", device="cuda", compute_type="float16")
            gpu_segments, _ = gpu_model.transcribe(media_path, beam_size=5, vad_filter=True)
            gpu_text = " ".join(segment.text.strip() for segment in gpu_segments)
            report_lines.append(f"gpu transcript: {gpu_text}")
            report_lines.append("GPU RESULT: PASS")
        else:
            report_lines.append("GPU RESULT: SKIPPED (libraries not installed)")

        report_lines.append("RESULT: PASS")
    except Exception:
        report_lines.append(traceback.format_exc())
        report_lines.append("RESULT: FAIL")

    os.makedirs(LOG_DIR, exist_ok=True)
    report_path = os.path.join(LOG_DIR, "selftest.txt")
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(report_lines) + "\n")
    return report_path


if TRANSCRIBE_FLAG in sys.argv:
    sys.exit(_run_batch_transcribe())

if SELFTEST_FLAG in sys.argv:
    _selftest_index = sys.argv.index(SELFTEST_FLAG)
    _media_argument = sys.argv[_selftest_index + 1] if len(sys.argv) > _selftest_index + 1 else ""
    _run_selftest(_media_argument)
    sys.exit(0)

import ui

if __name__ == "__main__":
    app = ui.SotvoxApp()
    app.run()
