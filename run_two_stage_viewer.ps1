# Two-stage viewer: trained high-level checkpoint + frozen HOMIE, MuJoCo physics, viser rendering (CPU, no Isaac Sim).
# Default: newest model_*.pt of the newest run under logs/rsl_rl/homie_pushcart. Open http://localhost:8080.
#
#   .\run_two_stage_viewer.ps1                                   # manual command (gamepad / sliders), latest checkpoint
#   .\run_two_stage_viewer.ps1 --auto                            # cycle through all eval scenarios
#   .\run_two_stage_viewer.ps1 --scenario turn_left --level 17   # one scenario, explicit correction-scale level
#   .\run_two_stage_viewer.ps1 --run 2026-09-17_21-25-55_b4_homie60000 --checkpoint <run>\model_1200.pt
#   .\run_two_stage_viewer.ps1 --port 8090 --payload 25
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
& $pythonPath -u "$PSScriptRoot\scripts\pushcart.py" mj view @RunArgs 2>&1 | ForEach-Object { $_.ToString() }
exit $LASTEXITCODE
