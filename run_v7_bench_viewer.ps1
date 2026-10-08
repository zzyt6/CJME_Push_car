param(
    [string]$Checkpoint = 'latest',
    [string]$Run = '',
    [int]$Port = 8092,
    [double]$Payload = 15.0,
    [double]$Speed = 0.5,
    [int]$Level = 17,
    [string]$PathName = 'composite_course',
    [ValidateSet('auto', 'legacy', 'v7')][string]$Plant = 'auto'
)
$ErrorActionPreference = 'Stop'
$pythonPath = if ($env:HOMIE_PYTHON) { $env:HOMIE_PYTHON } else { 'E:\envs\RLisaac601\python.exe' }
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "Python not found: $pythonPath" }
$env:PYTHONUTF8 = '1'
$viewerArgs = @('--checkpoint', $Checkpoint, '--port', $Port, '--payload', $Payload, '--v_nom', $Speed, '--level', $Level, '--path', $PathName, '--plant', $Plant)
if ($Run) { $viewerArgs += @('--run', $Run) }
& $pythonPath -u "$PSScriptRoot\scripts\view_v7_bench.py" @viewerArgs
exit $LASTEXITCODE
