# Separate first-stage V5 task baseline. Original V5 training launcher is unchanged.
param(
    [int]$NumEnvs = 128,
    [int]$MaxIterations = 120,
    [int]$Level = 5,
    [string]$RunName = 'v5_task_baseline',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\run_v5_pushcart.ps1" train `
    --quality_profile v5 --level $Level --num_envs $NumEnvs `
    --max_iterations $MaxIterations --run_name $RunName --headless `
    env.task.curriculum.enabled=false `
    env.task.commands.vx_range_max='[-0.3,0.3]' `
    env.task.commands.wz_max=0.2 `
    env.task.randomization.cart_force_pulse=false `
    env.task.randomization.wheel_stall_prob_per_s=0.0 `
    env.task.randomization.robot_push=false `
    env.task.rewards.scales.task_wrench=0.0 `
    env.task.rewards.scales.reaction_motion=0.0 `
    env.task.rewards.scales.centroidal_recovery=0.0 `
    env.task.rewards.scales.support_feasibility=0.0 `
    env.task.rewards.scales.force_waste=0.0 `
    env.task.rewards.scales.force_budget=-0.5 `
    @ExtraArgs
exit $LASTEXITCODE
