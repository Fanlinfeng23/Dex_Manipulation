$ErrorActionPreference = 'Stop'
$appRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$cleanup = Join-Path $appRoot 'cleanup.ps1'
$python = Join-Path $appRoot '.venv\Scripts\python.exe'
. $cleanup

function Assert-True {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}
function Get-FreeTestPort {
    $probe = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
    $probe.Start()
    try { return $probe.LocalEndpoint.Port } finally { $probe.Stop() }
}
function Wait-TestService {
    param([int]$TestPort)
    $deadline = [datetime]::UtcNow.AddSeconds(25)
    do {
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$TestPort/api/health" -TimeoutSec 1
            if ($health.ok) { return }
        } catch {}
        Start-Sleep -Milliseconds 100
    } while ([datetime]::UtcNow -lt $deadline)
    throw "Test service did not start on $TestPort."
}

$firstPort = Get-FreeTestPort
$secondPort = Get-FreeTestPort
while ($secondPort -eq $firstPort) { $secondPort = Get-FreeTestPort }
Assert-True ($firstPort -ne 8765 -and $secondPort -ne 8765) 'The user service port is reserved; rerun the test.'
$fixtures = @()
$occupied = $null
try {
    foreach ($testPort in @($firstPort, $secondPort)) {
        $fixtures += Start-Process -FilePath $python -ArgumentList @('-m', 'quest_teleop', '--port', "$testPort") `
            -WorkingDirectory $appRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $appRoot "runtime/cleanup-test-$testPort.out.log") `
            -RedirectStandardError (Join-Path $appRoot "runtime/cleanup-test-$testPort.err.log")
        Wait-TestService $testPort
    }
    $before = @(Get-TeleopCleanupPlan -Processes @(Get-CimInstance Win32_Process -Filter "Name LIKE 'python%.exe'") -TargetPort $firstPort)
    $ownedIds = @($before | ForEach-Object { [int]$_.Process.ProcessId })
    Assert-True ($ownedIds.Count -ge 2) 'Expected server and control worker.'
    & powershell -NoProfile -ExecutionPolicy Bypass -File $cleanup -Port $firstPort -WhatIf
    Assert-True ($LASTEXITCODE -eq 0) 'WhatIf failed.'
    foreach ($ownedId in $ownedIds) {
        Assert-True ($null -ne (Get-Process -Id $ownedId -ErrorAction SilentlyContinue)) 'WhatIf stopped a process.'
    }
    & powershell -NoProfile -ExecutionPolicy Bypass -File $cleanup -Port $firstPort
    Assert-True ($LASTEXITCODE -eq 0) 'Cleanup failed.'
    foreach ($ownedId in $ownedIds) {
        Assert-True ($null -eq (Get-Process -Id $ownedId -ErrorAction SilentlyContinue)) "PID $ownedId remained after cleanup."
    }
    Wait-TestService $secondPort
    & powershell -NoProfile -ExecutionPolicy Bypass -File $cleanup -Port $firstPort
    Assert-True ($LASTEXITCODE -eq 0) 'Repeated cleanup must succeed.'

    $occupied = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
    $occupied.Start()
    Assert-True ($occupied.LocalEndpoint.Port -ne 8765) 'The user service port is reserved; rerun the test.'
    & powershell -NoProfile -ExecutionPolicy Bypass -File $cleanup -Port $occupied.LocalEndpoint.Port
    Assert-True ($LASTEXITCODE -eq 1) 'Unrelated occupied port must report incomplete cleanup.'
    Write-Host 'PASS: real server/worker cleanup, WhatIf, other-port preservation, repeated cleanup and unrelated listener preservation.'
} finally {
    if ($null -ne $occupied) { $occupied.Stop() }
    foreach ($testPort in @($firstPort, $secondPort)) {
        & powershell -NoProfile -ExecutionPolicy Bypass -File $cleanup -Port $testPort
    }
    foreach ($fixture in $fixtures) { $fixture.Dispose() }
}
