# From scratch: train high-level Actor/encoder/Critic and the per-hand force estimator, keep HOMIE/HIM frozen.
[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$NumEnvs = 4096,
    [int]$MaxIterations = 20000,
    [int]$Level = 0,
    [int]$Seed = 1,
    [string]$RunName = 'v52_hand_estimator_20k',
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ExtraArgs
)
$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\run_v52_pushcart.ps1" train `
    --num_envs $NumEnvs --max_iterations $MaxIterations --level $Level `
    --seed $Seed --run_name $RunName --headless @ExtraArgs
exit $LASTEXITCODE
