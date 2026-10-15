# Teleop of HOMIE walking + the v2 balance model on the Isaac backend (Viser): open http://localhost:<Port> in a browser.
# Xbox gamepad (XInput) or sliders set the velocity; below |command| 0.05 the balance model takes over. -Demo drives it by itself.
[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$Port = 8092,
    [string]$Run = '',
    [string]$Checkpoint = '',
    [switch]$Demo
)
$ErrorActionPreference = 'Stop'
$pythonPath = if ($env:HOMIE_PYTHON) { $env:HOMIE_PYTHON } else { 'E:\envs\RLisaac601\python.exe' }
$envRoot = Split-Path -Parent $pythonPath
$env:PATH = "$envRoot;$envRoot\Library\bin;$envRoot\Scripts;$env:PATH"
$env:PYTHONUTF8 = '1'
$viewerArgs = @('--headless', '--device', 'cpu', '--port', $Port)
if ($Run) { $viewerArgs += @('--run', $Run) }
if ($Checkpoint) { $viewerArgs += @('--checkpoint', $Checkpoint) }
if ($Demo) { $viewerArgs += '--demo' }
# Windows PowerShell 5 treats native stderr warnings as errors when redirected.
# The process exit code, not Kit's warning stream, determines success.
$ErrorActionPreference = 'Continue'
& $pythonPath -u "$PSScriptRoot\scripts\view_homie_v2_balance.py" @viewerArgs 2>&1 | ForEach-Object { $_.ToString() }
exit $LASTEXITCODE
