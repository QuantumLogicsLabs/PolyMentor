param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$VenvPath = Join-Path $ProjectRoot ".venv"
$PythonPath = Join-Path $VenvPath "Scripts\python.exe"
$LogDir = Join-Path $ProjectRoot "logs"
$LogPath = Join-Path $LogDir "polymentor-local.log"
$HealthUrl = "http://${HostName}:${Port}/health"

function Write-Log {
    param([string]$Message)

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$timestamp] $Message" | Tee-Object -FilePath $LogPath -Append
}

function Test-PolyMentorHealth {
    try {
        $response = Invoke-RestMethod -Uri $HealthUrl -Method Get -TimeoutSec 5
        return $response.status -eq "ok"
    }
    catch {
        return $false
    }
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location $ProjectRoot

if (Test-PolyMentorHealth) {
    Write-Log "PolyMentor is already running at $HealthUrl."
    exit 0
}

if (-not (Test-Path $PythonPath)) {
    Write-Log "Creating Python virtual environment at $VenvPath."
    python -m venv $VenvPath *>> $LogPath
}

Write-Log "Installing/updating API dependencies."
& $PythonPath -m pip install --upgrade pip *>> $LogPath
& $PythonPath -m pip install -r (Join-Path $ProjectRoot "requirements-api.txt") *>> $LogPath

$UvicornArgs = "-m uvicorn src.api.app:app --host $HostName --port $Port"
$Command = "`"$PythonPath`" $UvicornArgs >> `"$LogPath`" 2>&1"

Write-Log "Starting PolyMentor at http://${HostName}:${Port}."
$process = Start-Process -FilePath "cmd.exe" `
    -ArgumentList "/c", $Command `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden `
    -PassThru

Start-Sleep -Seconds 5

if (Test-PolyMentorHealth) {
    Write-Log "PolyMentor started successfully. Process id: $($process.Id)."
    exit 0
}

Write-Log "PolyMentor did not pass the health check after startup. Check $LogPath."
exit 1

