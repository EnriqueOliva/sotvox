$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

function Write-Step($step, $msg) {
    Write-Host ""
    Write-Host "  [$step] $msg" -ForegroundColor Cyan
    Write-Host "  $('-' * 50)" -ForegroundColor DarkGray
}

function Write-Ok($msg) {
    Write-Host "       $msg" -ForegroundColor Green
}

function Write-Skip($msg) {
    Write-Host "       $msg" -ForegroundColor Yellow
}

function Write-Err($msg) {
    Write-Host "       $msg" -ForegroundColor Red
}

Write-Host ""
Write-Host "  =============================================" -ForegroundColor DarkCyan
Write-Host "       Sotvox - Setup" -ForegroundColor White
Write-Host "  =============================================" -ForegroundColor DarkCyan
Write-Host "  Project: $projectRoot" -ForegroundColor DarkGray
Write-Host ""

# Step 1: uv
Write-Step "1/5" "Checking uv (Python manager)..."
$uvPath = Get-Command uv -ErrorAction SilentlyContinue
if ($uvPath) {
    $uvVer = & uv --version 2>&1
    Write-Skip "Already installed: $uvVer"
} else {
    Write-Host "       Installing uv..." -ForegroundColor White
    try {
        irm https://astral.sh/uv/install.ps1 | iex
        $env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
        $uvVer = & uv --version 2>&1
        Write-Ok "Installed: $uvVer"
    } catch {
        Write-Err "Failed to install uv: $_"
        Write-Err "Please install manually: https://docs.astral.sh/uv/getting-started/installation/"
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Step 2: FFmpeg
Write-Step "2/5" "Checking FFmpeg (audio/video processing)..."
$ffmpegPath = Get-Command ffmpeg -ErrorAction SilentlyContinue
if ($ffmpegPath) {
    Write-Skip "Already installed: ffmpeg found at $($ffmpegPath.Source)"
} else {
    Write-Host "       Installing FFmpeg via winget..." -ForegroundColor White
    $wingetPath = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetPath) {
        try {
            & winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
            Write-Ok "FFmpeg installed."
        } catch {
            Write-Err "winget install failed. Trying choco..."
            $chocoPath = Get-Command choco -ErrorAction SilentlyContinue
            if ($chocoPath) {
                & choco install ffmpeg -y
                Write-Ok "FFmpeg installed via choco."
            } else {
                Write-Err "Could not install FFmpeg automatically."
                Write-Err "Please install manually: https://www.gyan.dev/ffmpeg/builds/"
                Read-Host "Press Enter to exit"
                exit 1
            }
        }
    } else {
        Write-Err "winget not found."
        $chocoPath = Get-Command choco -ErrorAction SilentlyContinue
        if ($chocoPath) {
            & choco install ffmpeg -y
            Write-Ok "FFmpeg installed via choco."
        } else {
            Write-Err "Neither winget nor choco found. Please install FFmpeg manually."
            Read-Host "Press Enter to exit"
            exit 1
        }
    }
}

# Step 3: Python 3.11
Write-Step "3/5" "Installing Python 3.11 via uv..."
try {
    & uv python install 3.11 2>&1 | ForEach-Object { Write-Host "       $_" -ForegroundColor Gray }
    Write-Ok "Python 3.11 ready."
} catch {
    Write-Err "Failed: $_"
    Read-Host "Press Enter to exit"
    exit 1
}

# Step 4: Virtual environment
Write-Step "4/5" "Creating virtual environment..."
Set-Location $projectRoot
if (Test-Path ".venv") {
    Write-Skip "Virtual environment already exists. Recreating..."
    Remove-Item -Recurse -Force ".venv"
}
try {
    & uv venv --python 3.11 2>&1 | ForEach-Object { Write-Host "       $_" -ForegroundColor Gray }
    Write-Ok "Virtual environment created."
} catch {
    Write-Err "Failed: $_"
    Read-Host "Press Enter to exit"
    exit 1
}

# Step 5: Dependencies
Write-Step "5/5" "Installing dependencies (this may take a few minutes)..."
$nvidiaGpu = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match "NVIDIA" } | Select-Object -First 1
$pipPkgs = @("faster-whisper", "tkinterdnd2")
if ($nvidiaGpu) {
    Write-Host "       NVIDIA GPU detected: $($nvidiaGpu.Name)" -ForegroundColor Green
    Write-Host "       faster-whisper, CUDA libraries, UI toolkit..." -ForegroundColor Gray
    $pipPkgs += "nvidia-cublas-cu12"
    $pipPkgs += "nvidia-cudnn-cu12"
} else {
    Write-Host "       No NVIDIA GPU detected - installing CPU-only mode" -ForegroundColor Yellow
    Write-Host "       faster-whisper, UI toolkit..." -ForegroundColor Gray
}
try {
    & uv pip install @pipPkgs 2>&1 | ForEach-Object {
        $line = $_.ToString()
        if ($line -match "Downloading|Installed|Resolved") {
            Write-Host "       $line" -ForegroundColor Gray
        }
    }
    Write-Ok "All dependencies installed."
} catch {
    Write-Err "Failed: $_"
    Read-Host "Press Enter to exit"
    exit 1
}

# Done
Write-Host ""
Write-Host "  =============================================" -ForegroundColor DarkCyan
Write-Host "       Setup complete!" -ForegroundColor Green
Write-Host "  =============================================" -ForegroundColor DarkCyan
Write-Host ""
Write-Host "  To launch the app, double-click:" -ForegroundColor White
Write-Host "       $projectRoot\launch.vbs" -ForegroundColor Yellow
Write-Host ""
Write-Host "  The first transcription will download the AI" -ForegroundColor Gray
Write-Host "  model (~1.5 GB). After that it's cached." -ForegroundColor Gray
Write-Host ""
Read-Host "  Press Enter to close"
