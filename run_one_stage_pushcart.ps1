# Stage-one push-cart RL: frozen HOMIE low-level policy + a single high-level correction/scheduling policy.
# Wraps scripts/pushcart.py (assets, viser checks, MuJoCo sim2sim teleop, Isaac Lab smoke/training).
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunArgs
)
$ErrorActionPreference = 'Stop'
if (-not $RunArgs -or $RunArgs.Count -eq 0) {
    Write-Host 'Choose a command, for example:'
    Write-Host '  .\run_one_stage_pushcart.ps1 mj view    # View the latest trained checkpoint'
    Write-Host '  .\run_two_stage_viewer.ps1             # Same viewer, no subcommand needed'
    Write-Host '  .\run_one_stage_pushcart.ps1 train --num_envs 4096 --headless'
    Write-Host ''
    $RunArgs = @('--help')
}
$pythonPath = if ($env:HOMIE_PYTHON) { $env:HOMIE_PYTHON } else { 'E:\envs\RLisaac601\python.exe' }
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "Python not found: $pythonPath. Set HOMIE_PYTHON to your RLisaac601 python.exe." }
$envRoot = Split-Path -Parent $pythonPath
$env:PATH = "$envRoot;$envRoot\Library\bin;$envRoot\Scripts;$env:PATH"
$env:PYTHONUTF8 = '1'
$ErrorActionPreference = 'Continue'
& $pythonPath -u "$PSScriptRoot\scripts\pushcart.py" @RunArgs 2>&1 | ForEach-Object { $_.ToString() }
exit $LASTEXITCODE
