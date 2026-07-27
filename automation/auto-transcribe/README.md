# Headless batch mode

Sotvox can transcribe a whole folder without opening a window, so a scheduled task or a script can drive it.

```powershell
Sotvox.exe --transcribe "C:\Recordings"
```

Every supported file in that folder that does not already have a transcript gets transcribed. Nothing appears on screen.

Requires Sotvox **1.3.0** or newer.

## Options

| Flag | Default | What it does |
| --- | --- | --- |
| `--transcribe <folder>` | *required* | Folder to scan for audio and video |
| `--output <folder>` | `Documents\sotvox-transcripts` | Where the `.txt` files are written |
| `--model <name>` | `large-v3-turbo` | `large-v3`, `large-v3-turbo`, `medium`, `small`, `base`, `tiny` |
| `--language <name>` | `Auto-detect` | `Spanish`, `English`, `Portuguese`, `French`, `German`, `Italian`, `Japanese`, `Chinese`, `Korean` |
| `--device <name>` | `Auto` | `Auto` or `CPU` |

## It skips what it already did

A file is transcribed only when `<output>\<same name>.txt` does not exist.

So `lecture.mp4` is skipped once `lecture.txt` is there. No state file, no database — **you can run it as often as you like and it will only pick up what is new.** To redo one, delete its `.txt`.

## Knowing how it went

The command is silent, so it reports two ways.

**Exit code:**

| Code | Meaning |
| --- | --- |
| `0` | Everything transcribed, or there was nothing new |
| `1` | At least one file failed |
| `2` | The input folder does not exist |

**Log** — one line per file, at `%LOCALAPPDATA%\Sotvox\logs\batch.txt`:

```
[2026-07-27 00:31:08] 1 new file(s) to transcribe
[2026-07-27 00:31:08] Loading model large-v3-turbo on cuda
[2026-07-27 00:31:10]   lecture.mp4: en, 6s audio in 2s -> lecture.txt
[2026-07-27 00:31:10] Batch finished: 1 transcribed, 0 failed
```

Because it is a windowed program, PowerShell needs `-Wait` to block on it:

```powershell
$run = Start-Process "C:\Program Files\Sotvox\Sotvox.exe" -PassThru -Wait `
    -ArgumentList '--transcribe', '"C:\Recordings"'
if ($run.ExitCode -ne 0) { Write-Warning "Sotvox exit code $($run.ExitCode)" }
```

## Example: transcribe new OBS recordings at logon

[`example-obs-watch.ps1`](example-obs-watch.ps1) is a small script that finds your OBS recording folder, transcribes anything new, and logs the result. Register it to run after you sign in:

```powershell
$script  = "C:\path\to\example-obs-watch.ps1"
$action  = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`""
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$trigger.Delay = 'PT2M'

Register-ScheduledTask -TaskName 'SotvoxAutoTranscribe' -Action $action -Trigger $trigger `
    -Settings (New-ScheduledTaskSettingsSet -StartWhenAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Hours 5)) -Force
```

Run it with `-DryRun` first to see what it would pick up without transcribing anything.

## Worth knowing before you automate it

- **Give the task a generous time limit.** The default in Task Scheduler is well under an hour, and Windows will kill a long batch mid-run. The example above uses five hours.
- **Do not run it while the Sotvox window is open.** Both would load a model onto the same GPU. The example script checks for this.
- **The first run downloads the speech model** (about 1.5 GB for `large-v3-turbo`), once. Give the first run internet and time.
- **Write the log somewhere outside any repo you push**, or the automation will keep committing its own log file.
