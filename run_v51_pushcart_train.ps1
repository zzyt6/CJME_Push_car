# From scratch: train high-level Actor/encoder/Critic, keep HOMIE/HIM frozen.
[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$NumEnvs = 4096,
    [int]$MaxIterations = 20000,
    [int]$Level = 0,
    [int]$Seed = 1,
    [string]$RunName = 'v51_interaction_privileged_20k',
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ExtraArgs
)
$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\run_v51_pushcart.ps1" train `
    --num_envs $NumEnvs --max_iterations $MaxIterations --level $Level `
    --seed $Seed --run_name $RunName --headless @ExtraArgs
exit $LASTEXITCODE
