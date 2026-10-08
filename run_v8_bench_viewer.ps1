# V8 teacher on the MuJoCo bench in Viser (V7 plant, fine-tuned HOMIE velocity estimator, leg residual).
#   .\run_v8_bench_viewer.ps1                                  newest V8 run, newest checkpoint, 25 kg
#   .\run_v8_bench_viewer.ps1 -Checkpoint <model_N.pt> -Payload 12.5
#   .\run_v8_bench_viewer.ps1 -NoLegResidual                   ablation: leg residual forced to zero
param(
    [string]$Checkpoint = 'latest',
    [string]$Run = '',
    [int]$Port = 8093,
    [double]$Payload = 25.0,
    [double]$Speed = 0.5,
    [int]$Level = 17,
    [string]$PathName = 'composite_course',
    [switch]$NoLegResidual
)
$ErrorActionPreference = 'Stop'
$pythonPath = if ($env:HOMIE_PYTHON) { $env:HOMIE_PYTHON } else { 'E:\envs\RLisaac601\python.exe' }
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "Python not found: $pythonPath" }
$env:PYTHONUTF8 = '1'
$viewerArgs = @('--version', 'v8', '--checkpoint', $Checkpoint, '--port', $Port, '--payload', $Payload, '--v_nom', $Speed,
                '--level', $Level, '--path', $PathName)
if ($Run) { $viewerArgs += @('--run', $Run) }
if ($NoLegResidual) { $viewerArgs += '--no_leg_residual' }
& $pythonPath -u "$PSScriptRoot\scripts\view_v7_bench.py" @viewerArgs
exit $LASTEXITCODE
