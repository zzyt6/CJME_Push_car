# New V6 training; V3.2 weights have a different observation schema and cannot be resumed as V6.
[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$NumEnvs = 4096,
    [int]$MaxIterations = 20000,
    [int]$Level = 0,
    [int]$Seed = 1,
    [string]$RunName = 'v6_v32_cart_him_hand_estimators_20k',
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ExtraArgs
)
$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\run_v6_pushcart.ps1" train `
    --num_envs $NumEnvs --max_iterations $MaxIterations --level $Level `
    --seed $Seed --run_name $RunName --headless @ExtraArgs
exit $LASTEXITCODE
