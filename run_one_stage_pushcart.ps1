# Stage-one push-cart RL: frozen HOMIE low-level policy + a single high-level correction/scheduling policy.
# Wraps scripts/pushcart.py (assets, viser checks, MuJoCo sim2sim teleop, Isaac Lab smoke/training).
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunArgs
)
$ErrorActionPreference = 'Stop'
$pythonPath = if ($env:HOMIE_PYTHON) { $env:HOMIE_PYTHON } else { 'E:\envs\RLisaac601\python.exe' }
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "Python not found: $pythonPath. Set HOMIE_PYTHON to your RLisaac601 python.exe." }
$envRoot = Split-Path -Parent $pythonPath
$env:PATH = "$envRoot;$envRoot\Library\bin;$envRoot\Scripts;$env:PATH"
$env:PYTHONUTF8 = '1'
$ErrorActionPreference = 'Continue'
& $pythonPath -u "$PSScriptRoot\scripts\pushcart.py" @RunArgs 2>&1 | ForEach-Object { $_.ToString() }
exit $LASTEXITCODE
