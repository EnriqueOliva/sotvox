$ErrorActionPreference = "Stop"

$projectRoot = Split-Path $PSScriptRoot -Parent
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "Dev environment not found at $python. Run setup\setup.vbs first."
    exit 1
}

Write-Host "[1/2] Freezing the application with PyInstaller..." -ForegroundColor Cyan
& $python -m PyInstaller --noconfirm --clean `
    --distpath (Join-Path $PSScriptRoot "dist_app") `
    --workpath (Join-Path $PSScriptRoot "build") `
    (Join-Path $PSScriptRoot "sotvox.spec")
if ($LASTEXITCODE -ne 0) { Write-Error "PyInstaller failed."; exit 1 }

$frozenExe = Join-Path $PSScriptRoot "dist_app\Sotvox\Sotvox.exe"
if (-not (Test-Path $frozenExe)) { Write-Error "Frozen app missing: $frozenExe"; exit 1 }

$leaked = Get-ChildItem (Join-Path $PSScriptRoot "dist_app") -Recurse -File `
    -Include "*cudnn*", "*cublas*" -ErrorAction SilentlyContinue
if ($leaked) {
    Write-Error "CUDA libraries leaked into the bundle: $($leaked.Name -join ', ')"
    exit 1
}

$appSize = (Get-ChildItem (Join-Path $PSScriptRoot "dist_app") -Recurse -File |
    Measure-Object -Sum Length).Sum
Write-Host ("      Frozen app: {0:N1} MB" -f ($appSize / 1MB)) -ForegroundColor Gray

$candidates = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles}\Inno Setup 6\ISCC.exe",
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
)
$iscc = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) {
    $command = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($command) { $iscc = $command.Source }
}
if (-not $iscc) {
    Write-Error "Inno Setup not found. Install it with: winget install --id JRSoftware.InnoSetup -e"
    exit 1
}

Write-Host "[2/2] Building the installer with Inno Setup..." -ForegroundColor Cyan
& $iscc (Join-Path $PSScriptRoot "sotvox.iss")
if ($LASTEXITCODE -ne 0) { Write-Error "Inno Setup failed."; exit 1 }

$installer = Join-Path $projectRoot "Sotvox-Setup.exe"
if (Test-Path $installer) {
    $sizeMb = [math]::Round((Get-Item $installer).Length / 1MB, 2)
    Write-Host "Built: $installer ($sizeMb MB)" -ForegroundColor Green
} else {
    Write-Error "Inno Setup reported success but $installer is missing."
    exit 1
}
