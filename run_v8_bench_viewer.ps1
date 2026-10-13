# V8 on the MuJoCo bench in Viser (V7 plant, fine-tuned HOMIE velocity estimator, leg residual).
#   .\run_v8_bench_viewer.ps1                                  V8.1 final student (deployable), 25 kg
#   .\run_v8_bench_viewer.ps1 -Payload 0                       same, empty cart
#   .\run_v8_bench_viewer.ps1 -V80                             V8.0 final student (comparison)
#   .\run_v8_bench_viewer.ps1 -Teacher                         V8.1 teacher (privileged truth, simulation only)
#   .\run_v8_bench_viewer.ps1 -Checkpoint <model_N.pt>         any V8 checkpoint
#   .\run_v8_bench_viewer.ps1 -NoLegResidual                   ablation: leg residual forced to zero
# The vx / vy / wz plots show command / actual / the V8.1 student's velocity-head estimate; the status line shows the
# student's robot-velocity estimate against the MuJoCo truth. Open http://localhost:<Port> in a browser.
param(
    [string]$Checkpoint = 'latest',
    [string]$Run = '',
    [int]$Port = 8093,
    [double]$Payload = 25.0,
    [double]$Speed = 0.5,
    [int]$Level = 17,
    [string]$PathName = 'composite_course',
    [switch]$V80,
    [switch]$Teacher,
    [switch]$NoLegResidual
)
$ErrorActionPreference = 'Stop'
$pythonPath = if ($env:HOMIE_PYTHON) { $env:HOMIE_PYTHON } else { 'E:\envs\RLisaac601\python.exe' }
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "Python not found: $pythonPath" }
$env:PYTHONUTF8 = '1'
if ($V80) { $Checkpoint = 'v8.0' }
if ($Teacher) { $Checkpoint = "$PSScriptRoot\logs\rsl_rl\homie_pushcart_v8\2026-10-12_20-52-19_v8_1_roa_teacher_8k\model_6000.pt" }
$viewerArgs = @('--version', 'v8', '--checkpoint', $Checkpoint, '--port', $Port, '--payload', $Payload, '--v_nom', $Speed,
                '--level', $Level, '--path', $PathName)
if ($Run) { $viewerArgs += @('--run', $Run) }
if ($NoLegResidual) { $viewerArgs += '--no_leg_residual' }
& $pythonPath -u "$PSScriptRoot\scripts\view_v7_bench.py" @viewerArgs
exit $LASTEXITCODE
