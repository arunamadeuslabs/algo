<#
.SYNOPSIS
    Sets up Windows Task Scheduler to run trading algos automatically every trading day.

.DESCRIPTION
    Creates a scheduled task that:
    - Runs Monday to Friday at 09:10 AM
    - Launches both EMA Crossover + Sapphire paper trading algos
    - Auto-stops after market close (15:35)
    - Restarts crashed algos automatically

.NOTES
    Run this script ONCE as Administrator to set up the schedule.
    To remove: schtasks /Delete /TN "NiftyAlgoTrader" /F

.EXAMPLE
    # Run as Admin:
    powershell -ExecutionPolicy Bypass -File setup_scheduler.ps1
#>

$ErrorActionPreference = "Continue"

# ── Configuration ────────────────────────────────────────────
$TaskName = "NiftyAlgoTrader"
$PythonExe = "C:\Users\aprabhu\AppData\Local\Programs\Python\Python313\python.exe"
$LauncherScript = "C:\Users\aprabhu\work\algo\launcher.py"
$WorkDir = "C:\Users\aprabhu\work\algo"
$TriggerTime = "09:10"  # Start 5 min before market open

# ── Verify paths ─────────────────────────────────────────────
if (-not (Test-Path $PythonExe)) {
    Write-Host "ERROR: Python not found at $PythonExe" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $LauncherScript)) {
    Write-Host "ERROR: Launcher script not found at $LauncherScript" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  NIFTY ALGO TRADER - Task Scheduler Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Task Name:    $TaskName"
Write-Host "  Python:       $PythonExe"
Write-Host "  Launcher:     $LauncherScript"
Write-Host "  Schedule:     Mon-Fri at $TriggerTime"
Write-Host "  Auto-stop:    15:35 (after market close)"
Write-Host ""

# ── Remove existing task if present ──────────────────────────
$existing = schtasks /Query /TN $TaskName 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Removing existing task..." -ForegroundColor Yellow
    schtasks /Delete /TN $TaskName /F 2>&1 | Out-Null
}

# ── Create the scheduled task ────────────────────────────────
# Using schtasks for compatibility (no admin PowerShell cmdlets needed)

$Action = "`"$PythonExe`" `"$LauncherScript`""

# Create XML for proper weekly schedule (Mon-Fri)
$xml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Nifty Algo Trader - Runs EMA Crossover and Sapphire paper trading algos Mon-Fri at market open</Description>
  </RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>2026-02-17T09:10:00</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByWeek>
        <WeeksInterval>1</WeeksInterval>
        <DaysOfWeek>
          <Monday />
          <Tuesday />
          <Wednesday />
          <Thursday />
          <Friday />
        </DaysOfWeek>
      </ScheduleByWeek>
    </CalendarTrigger>
  </Triggers>
  <Principals>
    <Principal>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>true</RunOnlyIfNetworkAvailable>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <WakeToRun>true</WakeToRun>
    <ExecutionTimeLimit>PT7H</ExecutionTimeLimit>
    <Priority>5</Priority>
  </Settings>
  <Actions>
    <Exec>
      <Command>$PythonExe</Command>
      <Arguments>"$LauncherScript"</Arguments>
      <WorkingDirectory>$WorkDir</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"@

$xmlPath = Join-Path $env:TEMP "algo_task.xml"
$xml | Out-File -FilePath $xmlPath -Encoding Unicode

try {
    schtasks /Create /TN $TaskName /XML $xmlPath /F | Out-Null
    Write-Host "  Task created successfully!" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Failed to create task. Try running as Administrator." -ForegroundColor Red
    Write-Host "  $_" -ForegroundColor Red
    exit 1
} finally {
    Remove-Item $xmlPath -ErrorAction SilentlyContinue
}

# ── Verify ───────────────────────────────────────────────────
Write-Host ""
Write-Host "  Verifying..." -ForegroundColor Gray
schtasks /Query /TN $TaskName /FO LIST | Select-String "Task Name|Status|Next Run"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  SETUP COMPLETE" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Your algos will auto-start Mon-Fri at 09:10 AM."
Write-Host "  They will auto-stop at 15:35 PM after market close."
Write-Host ""
Write-Host "  Manual commands:" -ForegroundColor Yellow
Write-Host "    python launcher.py              # Start now"
Write-Host "    python launcher.py --status     # Check status"
Write-Host "    python launcher.py --stop       # Stop all"
Write-Host "    python launcher.py --sapphire   # Sapphire only"
Write-Host "    python launcher.py --ema        # EMA only"
Write-Host ""
Write-Host "  To remove scheduler:" -ForegroundColor Yellow
Write-Host "    schtasks /Delete /TN `"$TaskName`" /F"
Write-Host ""
