# Sotvox

A local audio/video transcription desktop app for Windows. Drag and drop files, get text transcripts. Runs 100% locally — no internet, no cloud, no subscriptions. Supports GPU acceleration via NVIDIA CUDA, and works on CPU-only machines too.

Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (a high-performance reimplementation of OpenAI's Whisper using CTranslate2).

## Features

- **Drag & drop** audio and video files — or click to browse
- **GPU-accelerated** via NVIDIA CUDA when available, with automatic CPU fallback
- **Batch processing** — queue hundreds of files at once
- **Video support** — auto-extracts audio from mp4, mkv, avi, mov, webm, etc.
- **Audio support** — mp3, wav, m4a, ogg, flac, wma, aac
- **99+ languages** — Spanish, English, Portuguese, French, German, and more
- **Multiple models** — from `tiny` (fastest) to `large-v3` (most accurate)
- **Output format options** — save as `.txt` transcript, or copy source files renamed by their transcript content (useful for cataloging voice clips)
- **Timeout protection** — gracefully skips files that hang due to noise or corruption
- **VAD fallback** — retries with Voice Activity Detection disabled if the first pass returns empty
- **Session logging** — automatically saves a detailed session log to `log/` on every run

## Requirements

- Windows 10 or 11 (64-bit)
- ~700 MB disk space, plus ~1.5 GB for the AI model on first use
- NVIDIA GPU with current drivers — optional, for much faster transcription

**Nothing else.** Python, FFmpeg and every library are bundled inside the app. There is no runtime to install and no setup step.

## Install

Run **`Sotvox-Setup.exe`** (~66 MB) — [download the latest release](https://github.com/EnriqueOliva/sotvox/releases/latest).

It installs per-user (no admin prompt), adds Start Menu and optional Desktop shortcuts, and the app is ready the moment setup finishes. Uninstall anytime from *Add or remove programs*.

The first transcription downloads the selected Whisper model (~1.5 GB for `large-v3`) and caches it for later runs.

### GPU acceleration (optional)

Sotvox runs on the CPU out of the box. If it detects an NVIDIA GPU, the **Options** panel offers **Enable GPU Acceleration...**, which downloads NVIDIA's CUDA libraries (~1.2 GB, once) and switches to the GPU automatically. They are kept out of the installer so users without an NVIDIA GPU never download them.

## Daily use

1. Launch **Sotvox** from the Start Menu or Desktop
2. Drag audio or video files onto the list
3. Click **Transcribe**
4. Click **Open Output Folder** to see the transcripts

Transcripts are written to `Documents\sotvox-transcripts`. Session logs go to `%LOCALAPPDATA%\Sotvox\logs`.

## Building from source

```powershell
setup\setup.vbs          # one-time developer environment (uv, Python 3.11, dependencies)
installer\build.ps1      # freezes the app and produces Sotvox-Setup.exe
```

Building the installer also needs [Inno Setup](https://jrsoftware.org/isinfo.php) (`winget install --id JRSoftware.InnoSetup -e`). `build.ps1` runs PyInstaller, verifies no CUDA libraries leaked into the bundle, then compiles the installer into the project root.

To run from source without freezing: `.venv\Scripts\pythonw.exe src\main.py`

To verify a build end-to-end (decoding, VAD, CPU and GPU inference):

```powershell
Sotvox.exe --selftest "C:\path\to\some.mp4"    # writes %LOCALAPPDATA%\Sotvox\logs\selftest.txt
```

## Project Structure

```
sotvox/
├── Sotvox-Setup.exe        # Built installer (produced by installer/build.ps1, not in git)
├── launch.vbs              # Convenience launcher for running from source
├── installer/              # Installer build sources
│   ├── sotvox.spec         # PyInstaller recipe (bundles Python, excludes CUDA)
│   ├── version_info.txt    # Windows file/product version resource
│   ├── sotvox.iss          # Inno Setup script
│   └── build.ps1           # Freeze + compile, produces Sotvox-Setup.exe
├── setup/
│   ├── setup.vbs           # Developer environment launcher
│   └── setup.ps1           # Developer environment (uv, Python 3.11, dependencies)
├── assets/                 # App icon (sotvox.ico + PNGs)
├── src/
│   ├── main.py             # Entry point (DPI, CUDA paths, self-test, app launch)
│   ├── ui.py               # GUI (tkinter + tkinterdnd2, Windows 95 styling)
│   ├── engine.py           # Transcription and media decoding (faster-whisper, PyAV)
│   ├── gpu_pack.py         # Optional CUDA library download and install
│   └── constants.py        # Paths, supported formats, languages
├── LICENSE                 # MIT License
└── .venv/                  # Developer virtual environment (created by setup)
```

At runtime the app writes only to user locations: `Documents\sotvox-transcripts` (transcripts),
`%LOCALAPPDATA%\Sotvox\logs` (session logs) and `%LOCALAPPDATA%\Sotvox\cuda` (optional GPU pack).

## Configuration

All settings are available in the app UI:

| Setting       | Options                                        | Default           |
| ------------- | ---------------------------------------------- | ----------------- |
| Language      | Auto-detect, Spanish, English, +7 more         | Spanish           |
| Model         | large-v3, medium, small, base, tiny            | large-v3          |
| Device        | Auto, CPU, GPU (CUDA)                          | Auto              |
| Output folder | Any local path                                 | `Documents\sotvox-transcripts` |

## Model Information

Models are downloaded automatically on first use and cached at `~/.cache/huggingface/`.

Available models, from lightest to heaviest: `tiny`, `base`, `small`, `medium`, `large-v3`. Smaller models are faster and use less VRAM; larger models produce more accurate transcriptions. On CPU-only machines, `small` or `medium` are recommended for a good balance of speed and accuracy.

## License

MIT License. See [LICENSE](LICENSE).

Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT), based on [OpenAI Whisper](https://github.com/openai/whisper) (MIT).
