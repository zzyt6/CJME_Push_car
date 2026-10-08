# Live V7 view on the Isaac backend (Viser): open http://localhost:<Port> in a browser.
[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$Port = 8091,
    [string]$PathName = 'composite_course',
    [string]$Run = '',
    [string]$Checkpoint = '',
    [double]$Payload = 15.0,
    [double]$Speed = 0.5,
    [double]$Difficulty = 1.0
)
$ErrorActionPreference = 'Stop'
$pythonPath = if ($env:HOMIE_PYTHON) { $env:HOMIE_PYTHON } else { 'E:\envs\RLisaac601\python.exe' }
$env:PYTHONUTF8 = '1'
$viewerArgs = @('--headless', '--device', 'cpu', '--port', $Port, '--path', $PathName, '--payload', $Payload, '--v_nom', $Speed, '--difficulty', $Difficulty)
if ($Run) { $viewerArgs += @('--run', $Run) }
if ($Checkpoint) { $viewerArgs += @('--checkpoint', $Checkpoint) }
& $pythonPath -u "$PSScriptRoot\scripts\view_v7_isaac.py" @viewerArgs
exit $LASTEXITCODE
