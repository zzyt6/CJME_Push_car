# V5 reaction recovery training from scratch. Uses isolated homie_pushcart_v5 logs and assets.
param(
    [int]$NumEnvs = 4096,
    [int]$MaxIterations = 20000,
    [int]$Level = 0,
    [string]$RunName = 'v5_reaction_recovery_scratch',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\run_v5_pushcart.ps1" train `
    --quality_profile v5 `
    --level $Level `
    --num_envs $NumEnvs `
    --max_iterations $MaxIterations `
    --run_name $RunName `
    --headless `
    @ExtraArgs
exit $LASTEXITCODE
