# V5.1: offline accuracy test of the interaction (load) predictor on a trained checkpoint.
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Checkpoint = 'logs/rsl_rl/homie_pushcart_v51/2026-09-25_04-06-26_v51_interaction_privileged_20k/model_14600.pt',
    [int]$NumEnvs = 512,
    [int]$Steps = 1500,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ExtraArgs
)
$ErrorActionPreference = 'Stop'
$pythonPath = if ($env:HOMIE_PYTHON) { $env:HOMIE_PYTHON } else { 'E:\envs\RLisaac601\python.exe' }
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "Python not found: $pythonPath" }
$envRoot = Split-Path -Parent $pythonPath
$env:PATH = "$envRoot;$envRoot\Library\bin;$envRoot\Scripts;$env:PATH"
$env:PYTHONUTF8 = '1'
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
& $pythonPath -u "$PSScriptRoot\scripts\eval_pushcart_v51_force_predictor.py" `
    --checkpoint $Checkpoint --num_envs $NumEnvs --steps $Steps --headless @ExtraArgs 2>&1 | ForEach-Object { $_.ToString() }
exit $LASTEXITCODE
