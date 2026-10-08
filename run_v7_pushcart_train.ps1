# New V7 training (V7 assets must be built: run_v7_pushcart.ps1 build all, then build usd).
[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$NumEnvs = 4096,
    [int]$MaxIterations = 20000,
    [int]$Level = 0,
    [int]$Seed = 1,
    [string]$RunName = 'v7_slip_grasp_coulomb_cart_20k',
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ExtraArgs
)
$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\run_v7_pushcart.ps1" train `
    --num_envs $NumEnvs --max_iterations $MaxIterations --level $Level `
    --seed $Seed --run_name $RunName --headless @ExtraArgs
exit $LASTEXITCODE
