import ctypes
import os
import sys

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


if SELFTEST_FLAG in sys.argv:
    _selftest_index = sys.argv.index(SELFTEST_FLAG)
    _media_argument = sys.argv[_selftest_index + 1] if len(sys.argv) > _selftest_index + 1 else ""
    _run_selftest(_media_argument)
    sys.exit(0)

import ui

if __name__ == "__main__":
    app = ui.SotvoxApp()
    app.run()
