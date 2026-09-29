# Push-cart control benchmark (MuJoCo + viser, CPU). Same paths / steps / disturbances / high-level PID for every policy.
#
#   .\run_pushcart_bench.ps1 path   --policies passthrough two_stage:latest --view                 # large S, live payload/speed
#   .\run_pushcart_bench.ps1 path   --policies v53_original:latest v53_new:latest --view --level 13
#   .\run_pushcart_bench.ps1 path   --paths s_curve right_angle --mirror --seeds 0 1 2            # subset, right-turn copies
#   .\run_pushcart_bench.ps1 step   --policies passthrough two_stage:latest                       # experiment 2
#   .\run_pushcart_bench.ps1 robust --policies passthrough two_stage:latest --factors payload_kg kick
#   .\run_pushcart_bench.ps1 report --out_dir logs\bench\<dir>                                        # regenerate tables / figures
#
# Policies: passthrough | robot_only | two_stage[:<model_N.pt>|:latest] | plugin:<module>:<Class>[:<arg>]
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
& $pythonPath -u "$PSScriptRoot\scripts\pushcart_bench.py" @RunArgs 2>&1 | ForEach-Object { $_.ToString() }
exit $LASTEXITCODE
