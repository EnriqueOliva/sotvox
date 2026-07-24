$ErrorActionPreference = "Stop"

$candidates = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles}\Inno Setup 6\ISCC.exe",
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
)
$iscc = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) {
    $cmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($cmd) { $iscc = $cmd.Source }
}
if (-not $iscc) {
    Write-Error "Inno Setup (ISCC.exe) not found. Install it with: winget install --id JRSoftware.InnoSetup -e"
    exit 1
}

$iss = Join-Path $PSScriptRoot "sotvox.iss"
Write-Host "Compiling $iss with $iscc" -ForegroundColor Cyan
& $iscc $iss
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$out = Join-Path (Split-Path $PSScriptRoot -Parent) "Sotvox-Setup.exe"
if (Test-Path $out) {
    $sizeMb = [math]::Round((Get-Item $out).Length / 1MB, 2)
    Write-Host "Built: $out ($sizeMb MB)" -ForegroundColor Green
} else {
    Write-Error "Build reported success but $out is missing."
    exit 1
}
