param(
    [string]$TaskName = "PolyMentor Daily Local Start",
    [string]$StartTime = "09:00",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$StartScript = Join-Path $ScriptDir "start-polymentor-local.ps1"

if (-not (Test-Path $StartScript)) {
    throw "Cannot find startup script at $StartScript"
}

$PowerShellPath = Join-Path $PSHOME "powershell.exe"
$TaskAction = "`"$PowerShellPath`" -NoProfile -ExecutionPolicy Bypass -File `"$StartScript`" -Port $Port"

schtasks.exe /Create `
    /TN $TaskName `
    /SC DAILY `
    /ST $StartTime `
    /TR $TaskAction `
    /F | Out-Host

Write-Host "Registered scheduled task '$TaskName' to run daily at $StartTime."
Write-Host "It starts PolyMentor locally on http://127.0.0.1:$Port when your laptop is on."

