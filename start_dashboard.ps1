param(
    [switch]$Stop,
    [switch]$Restart,
    [switch]$NoBrowser
)

$ErrorActionPreference = "SilentlyContinue"
$ProjectDir = $PSScriptRoot  # portable: the folder this script lives in
$Port = 8765
$Url = "http://127.0.0.1:$Port"
$LogFile = Join-Path $ProjectDir "server.log"

function Get-ServerPids {
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.OwningProcess -gt 4 }
    if ($conns) {
        return @($conns | Select-Object -ExpandProperty OwningProcess -Unique)
    }
    return @()
}

if ($Stop -or $Restart) {
    $pids = Get-ServerPids
    foreach ($p in $pids) {
        Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
    }
    if ($Stop) {
        Write-Host "AI Quota Hub durduruldu (Port $Port)." -ForegroundColor Yellow
        exit 0
    }
    Start-Sleep -Milliseconds 500
}

$existingPids = Get-ServerPids
if ($existingPids.Count -eq 0) {
    Write-Host "AI Quota Hub baslatiliyor ($Url)..." -ForegroundColor Cyan
    # Launch via WMI Win32_Process.Create so the daemon is outside any caller/agent Job Object
    # (JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE) and survives after the invoking shell exits.
    $pyExe = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $pyExe) { $pyExe = "python" }
    $cmdLine = "cmd.exe /c `"`"$pyExe`" -m uvicorn main:app --host 127.0.0.1 --port $Port > `"$LogFile`" 2> `"$ProjectDir\server.err.log`"`""
    $startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
    $null = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = $cmdLine
        CurrentDirectory = $ProjectDir
        ProcessStartupInformation = $startup
    }

    # Wait up to 5 seconds for port 8765 to listen
    for ($i = 0; $i -lt 25; $i++) {
        Start-Sleep -Milliseconds 200
        if ((Get-ServerPids).Count -gt 0) {
            break
        }
    }
} else {
    Write-Host "AI Quota Hub zaten calisiyor (PID: $($existingPids -join ', '))." -ForegroundColor Green
}

if (-not $NoBrowser) {
    Start-Process $Url
}
