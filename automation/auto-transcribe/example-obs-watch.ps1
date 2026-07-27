# Example: transcribe new OBS recordings with Sotvox, unattended.
#
# Finds the folder OBS is currently recording into, transcribes anything that
# does not have a transcript yet, and writes what happened to a log.
#
# Run with -DryRun to see what it would do without transcribing.

[CmdletBinding()]
param(
    [string]$OutputFolder   = (Join-Path ([Environment]::GetFolderPath('MyDocuments')) 'sotvox-transcripts'),
    [string]$FallbackFolder = (Join-Path $env:USERPROFILE 'Videos\OBS'),
    [string]$LogFile        = (Join-Path $env:LOCALAPPDATA 'Sotvox\logs\auto-transcribe.log'),
    [switch]$DryRun
)

$lockFile  = Join-Path $env:TEMP 'sotvox-auto-transcribe.lock'
$timeoutMs = 4 * 60 * 60 * 1000
$mediaExtensions = @('.mp4', '.mkv', '.avi', '.mov', '.webm', '.wmv', '.ts', '.flv',
                     '.mp3', '.wav', '.m4a', '.ogg', '.flac', '.wma', '.aac')

function Write-Log($message) {
    $line = "{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $message
    New-Item -ItemType Directory -Force -Path (Split-Path $LogFile) | Out-Null
    Add-Content -Path $LogFile -Value $line
}

# OBS keeps the recording folder in its own config, so switching profiles or
# output folders inside OBS is picked up without editing this script.
function Get-ObsRecordingFolder {
    $obsRoot = Join-Path $env:APPDATA 'obs-studio'
    if (-not (Test-Path $obsRoot)) { return $null }

    $profileDir = $null
    foreach ($configName in @('user.ini', 'global.ini')) {
        $configPath = Join-Path $obsRoot $configName
        if (-not (Test-Path $configPath)) { continue }
        $match = Select-String -Path $configPath -Pattern '^\s*ProfileDir\s*=\s*(.+)$' | Select-Object -First 1
        if ($match) { $profileDir = $match.Matches[0].Groups[1].Value.Trim(); break }
    }
    if (-not $profileDir) { return $null }

    $basicIni = Join-Path $obsRoot "basic\profiles\$profileDir\basic.ini"
    if (-not (Test-Path $basicIni)) { return $null }
    $content = Get-Content $basicIni

    $modeLine = $content | Select-String -Pattern '^\s*Mode\s*=\s*(.+)$' | Select-Object -First 1
    $mode = if ($modeLine) { $modeLine.Matches[0].Groups[1].Value.Trim() } else { 'Simple' }
    $key  = if ($mode -eq 'Advanced') { 'RecFilePath' } else { 'FilePath' }

    $pathLine = $content | Select-String -Pattern "^\s*$key\s*=\s*(.+)$" | Select-Object -First 1
    if (-not $pathLine) { return $null }

    $folder = $pathLine.Matches[0].Groups[1].Value.Trim().Replace('\\', '\').Replace('/', '\')
    if (Test-Path $folder) { return $folder }
    return $null
}

function Find-SotvoxExe {
    $candidates = @(
        (Join-Path $env:ProgramFiles 'Sotvox\Sotvox.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Sotvox\Sotvox.exe')
    )
    foreach ($hive in @('HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*',
                        'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*')) {
        $entry = Get-ItemProperty $hive -ErrorAction SilentlyContinue |
                 Where-Object { $_.DisplayName -eq 'Sotvox' } | Select-Object -First 1
        if ($entry -and $entry.InstallLocation) { $candidates += (Join-Path $entry.InstallLocation 'Sotvox.exe') }
    }
    return $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
}

# ------------------------------------------------------------------ main

if (Test-Path $lockFile) {
    $age = (Get-Date) - (Get-Item $lockFile).LastWriteTime
    if ($age.TotalHours -lt 6) { Write-Log 'already running, skipping'; return }
}
New-Item -ItemType File -Path $lockFile -Force | Out-Null

try {
    $recordingFolder = Get-ObsRecordingFolder
    if (-not $recordingFolder) {
        Write-Log "WARN  could not read the OBS settings, falling back to $FallbackFolder"
        if (-not (Test-Path $FallbackFolder)) { Write-Log 'WARN  fallback folder is missing too, stopping'; return }
        $recordingFolder = $FallbackFolder
    }

    $existing = @{}
    if (Test-Path $OutputFolder) {
        Get-ChildItem $OutputFolder -Filter *.txt -File | ForEach-Object { $existing[$_.BaseName] = $true }
    }
    $pending = Get-ChildItem $recordingFolder -File -ErrorAction SilentlyContinue |
        Where-Object { $mediaExtensions -contains $_.Extension.ToLower() -and -not $existing[$_.BaseName] }

    if (-not $pending) { Write-Log "nothing new in $recordingFolder"; return }
    Write-Log "$($pending.Count) new recording(s) in $recordingFolder"

    if (Get-Process Sotvox -ErrorAction SilentlyContinue) {
        Write-Log 'the Sotvox window is open, skipping so both do not use the GPU at once'
        return
    }

    $sotvox = Find-SotvoxExe
    if (-not $sotvox) { Write-Log 'Sotvox is not installed'; return }
    if ([version](Get-Item $sotvox).VersionInfo.FileVersion -lt [version]'1.3.0.0') {
        Write-Log 'Sotvox is older than 1.3.0 and has no batch mode, skipping'
        return
    }

    if ($DryRun) {
        $pending | ForEach-Object { Write-Log "DRYRUN would transcribe $($_.Name)" }
        return
    }

    $started = Get-Date
    $run = Start-Process $sotvox -PassThru -ArgumentList @(
        '--transcribe', "`"$recordingFolder`"", '--output', "`"$OutputFolder`"")
    if (-not $run.WaitForExit($timeoutMs)) {
        $run.Kill()
        Write-Log 'timed out and was stopped'
    } else {
        $minutes = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)
        Write-Log "finished in $minutes min (exit $($run.ExitCode)); per-file detail in Sotvox batch.txt"
    }
}
finally {
    Remove-Item $lockFile -Force -ErrorAction SilentlyContinue
}
