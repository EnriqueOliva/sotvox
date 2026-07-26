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
Write-Host "       Sotvox - developer environment" -ForegroundColor White
Write-Host "  =============================================" -ForegroundColor DarkCyan
Write-Host "  Project: $projectRoot" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  End users do not need this script - they run" -ForegroundColor DarkGray
Write-Host "  Sotvox-Setup.exe, which needs nothing installed." -ForegroundColor DarkGray
Write-Host ""

Write-Step "1/4" "Checking uv (Python manager)..."
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
        Write-Err "Install manually: https://docs.astral.sh/uv/getting-started/installation/"
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Step "2/4" "Installing Python 3.11 via uv..."
$pythonOutput = & uv python install 3.11 2>&1 | ForEach-Object {
    Write-Host "       $_" -ForegroundColor Gray
    $_.ToString()
}
$pythonInstalled = ($LASTEXITCODE -eq 0) -or ($pythonOutput -match "already installed")
if ($pythonInstalled) {
    Write-Ok "Python 3.11 ready."
} else {
    Write-Err "Failed to install Python 3.11."
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Step "3/4" "Creating virtual environment..."
Set-Location $projectRoot
if (Test-Path ".venv") {
    Write-Skip "Virtual environment already exists. Recreating..."
    Remove-Item -Recurse -Force ".venv"
}
& uv venv --python 3.11 2>&1 | ForEach-Object { Write-Host "       $_" -ForegroundColor Gray }
if (($LASTEXITCODE -ne 0) -or -not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Err "Failed to create the virtual environment."
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Ok "Virtual environment created."

Write-Step "4/4" "Installing dependencies (this may take a few minutes)..."
$packages = @("faster-whisper", "tkinterdnd2", "pyinstaller")
$nvidiaGpu = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match "NVIDIA" } | Select-Object -First 1
if ($nvidiaGpu) {
    Write-Host "       NVIDIA GPU detected: $($nvidiaGpu.Name)" -ForegroundColor Green
    Write-Host "       Adding CUDA libraries for local GPU testing..." -ForegroundColor Gray
    $packages += "nvidia-cublas-cu12"
    $packages += "nvidia-cudnn-cu12"
} else {
    Write-Host "       No NVIDIA GPU detected - CPU only." -ForegroundColor Yellow
}
& uv pip install @packages 2>&1 | ForEach-Object {
    $line = $_.ToString()
    if ($line -match "Downloading|Installed|Resolved|error") {
        Write-Host "       $line" -ForegroundColor Gray
    }
}
if ($LASTEXITCODE -ne 0) {
    Write-Err "Failed to install dependencies."
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Ok "All dependencies installed."

Write-Host ""
Write-Host "  =============================================" -ForegroundColor DarkCyan
Write-Host "       Developer setup complete" -ForegroundColor Green
Write-Host "  =============================================" -ForegroundColor DarkCyan
Write-Host ""
Write-Host "  Run the app from source:" -ForegroundColor White
Write-Host "       .venv\Scripts\pythonw.exe src\main.py" -ForegroundColor Yellow
Write-Host "  Build the distributable installer:" -ForegroundColor White
Write-Host "       installer\build.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "  FFmpeg is no longer required - audio and video are" -ForegroundColor Gray
Write-Host "  decoded with the bundled PyAV libraries." -ForegroundColor Gray
Write-Host ""
Read-Host "  Press Enter to close"
