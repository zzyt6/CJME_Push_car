# V4 push-cart training from scratch. Keeps the canonical homie_pushcart experiment directory.
param(
    [int]$NumEnvs = 4096,
    [int]$MaxIterations = 20000,
    [int]$Level = 0,
    [string]$RunName = 'v4_speed_quality_scratch',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\run_one_stage_pushcart.ps1" train `
    --quality_profile v4 `
    --level $Level `
    --num_envs $NumEnvs `
    --max_iterations $MaxIterations `
    --run_name $RunName `
    --headless `
    @ExtraArgs
exit $LASTEXITCODE
