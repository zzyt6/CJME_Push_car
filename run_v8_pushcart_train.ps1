# V8 teacher training (phase 1). Leg residual on by default; pass `env.task.v8.leg_residual.enabled=false` (Hydra) for
# the ablation.
[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$NumEnvs = 4096,
    [int]$MaxIterations = 20000,
    [int]$Level = 0,
    [int]$Seed = 1,
    [string]$RunName = 'v8_teacher',
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ExtraArgs
)
$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\run_v8_pushcart.ps1" train `
    --num_envs $NumEnvs --max_iterations $MaxIterations --level $Level `
    --seed $Seed --run_name $RunName --headless @ExtraArgs
exit $LASTEXITCODE
