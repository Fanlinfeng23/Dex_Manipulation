$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '..\cleanup.ps1')

function Assert-True {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}
function New-FixtureProcess {
    param([int]$Id, [int]$Parent, [string]$Command, [string]$Name = 'python.exe', [int]$Created = 2)
    [pscustomobject]@{ ProcessId = $Id; ParentProcessId = $Parent; Name = $Name;
        CommandLine = $Command; CreationDate = ([datetime]'2026-01-01').AddSeconds($Created) }
}

$processes = @(
    (New-FixtureProcess 10 1 '"D:\project with spaces\.venv\Scripts\python.exe" -m quest_teleop' -Created 0),
    (New-FixtureProcess 20 10 '"D:\Python\python.exe" -m quest_teleop --port "8765"' -Created 1),
    (New-FixtureProcess 21 20 'python.exe -c "from multiprocessing.spawn import spawn_main; spawn_main(parent_pid=20, pipe_handle=920)" --multiprocessing-fork'),
    (New-FixtureProcess 30 1 'python.exe -u -X utf8 -m quest_teleop.keyboard --port=8765'),
    (New-FixtureProcess 31 30 'python.exe -m quest_teleop.keyboard --port 8765' -Created 3),
    (New-FixtureProcess 40 1 'python.exe -m quest_teleop --port 8766'),
    (New-FixtureProcess 41 40 'python.exe -c "from multiprocessing.spawn import spawn_main; spawn_main(parent_pid=40, pipe_handle=22)" --multiprocessing-fork' -Created 3),
    (New-FixtureProcess 50 1 'python.exe -c "print(''-m quest_teleop --port 8765'')"'),
    (New-FixtureProcess 51 1 'python.exe unrelated.py -m quest_teleop'),
    (New-FixtureProcess 52 1 'python.exe -m quest_teleop_other'),
    (New-FixtureProcess 53 1 'powershell.exe -m quest_teleop' -Name 'powershell.exe'),
    (New-FixtureProcess 54 20 'python.exe -m http.server'),
    (New-FixtureProcess 55 20 'adb.exe server nodaemon' -Name 'adb.exe'),
    (New-FixtureProcess 56 999 'python.exe -c "from multiprocessing.spawn import spawn_main; spawn_main(parent_pid=999, pipe_handle=22)" --multiprocessing-fork'),
    (New-FixtureProcess 57 20 'python.exe -c "from multiprocessing.spawn import spawn_main; spawn_main(parent_pid=20, pipe_handle=22)" --multiprocessing-fork' -Created 0),
    (New-FixtureProcess 58 1 'python.exe -m quest_teleop --port invalid'),
    (New-FixtureProcess 59 20 'python.exe -c "from multiprocessing.spawn import spawn_main; spawn_main(parent_pid=999, pipe_handle=22)" --multiprocessing-fork'),
    (New-FixtureProcess 60 1 'python.exe -m quest_teleop --port 8765 --port 8766'),
    (New-FixtureProcess 61 1 '')
)
$plan = @(Get-TeleopCleanupPlan -Processes $processes -TargetPort 8765)
$ids = @($plan | ForEach-Object { [int]$_.Process.ProcessId })
Assert-True (($ids | Sort-Object) -join ',' -eq '10,20,21,30,31') 'Wrong cleanup targets for 8765.'
Assert-True ([array]::IndexOf($ids, 21) -lt [array]::IndexOf($ids, 20)) 'Worker must stop before server.'
Assert-True ([array]::IndexOf($ids, 20) -lt [array]::IndexOf($ids, 10)) 'Real Python must stop before venv launcher.'
Assert-True ([array]::IndexOf($ids, 31) -lt [array]::IndexOf($ids, 30)) 'Keyboard real Python must stop before its launcher.'
$other = @(Get-TeleopCleanupPlan -Processes $processes -TargetPort 8766)
Assert-True ((@($other | ForEach-Object { $_.Process.ProcessId } | Sort-Object) -join ',') -eq '40,41,60') 'Wrong cleanup targets for custom port.'
Assert-True (@(Get-TeleopCleanupPlan -Processes $processes -TargetPort 18765).Count -eq 0) 'Empty plan expected.'
Write-Host 'PASS: default/custom ports, quoted paths, keyboard/server wrappers, worker ordering, PID reuse, unrelated processes and orphan exclusion.'
