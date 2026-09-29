# Existing bench Viser, with the V5.3 policy and selectable original/new scenes.
param(
    [string]$Checkpoint = 'latest',
    [ValidateSet('both','original','new')][string]$Scene = 'both',
    [int]$Port = 8080,
    [double]$Payload = 15.0,
    [double]$Speed = 0.3,
    [int]$Level = 13,
    [string]$PathName = 'straight'
)
$ErrorActionPreference = 'Stop'
$pythonPath = if ($env:HOMIE_PYTHON) { $env:HOMIE_PYTHON } else { 'E:\envs\RLisaac601\python.exe' }
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "Python not found: $pythonPath" }
$env:PYTHONUTF8 = '1'
$ErrorActionPreference = 'Continue'
& $pythonPath -u "$PSScriptRoot\scripts\view_v53_bench.py" --checkpoint $Checkpoint --scene $Scene --port $Port --payload $Payload --v_nom $Speed --level $Level --path $PathName 2>&1 | ForEach-Object { $_.ToString() }
exit $LASTEXITCODE
