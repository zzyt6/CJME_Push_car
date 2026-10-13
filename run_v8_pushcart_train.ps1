# V8.1 teacher training (step 1 of 2): ROA teacher with the V8.1 defaults -- upper body explicit PD with Unitree's
# real-robot gains, arm-drive randomisation 0.8-1.2, lambda 0 -> 0.1 over iterations 3000-5000, adaptation module
# fitted every update, velocity head. The released student used model_6000 of an 8000-iteration run.
# Step 2 (RMA phase 2, student-driven, adaptation module only):
#   .\run_v8_pushcart.ps1 adapt --checkpoint <run>\model_6000.pt --num_envs 2048 --max_iterations 300 --headless
# docs/push_cart_v8_teacher_student_plan.md, section 12.5.
[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$NumEnvs = 4096,
    [int]$MaxIterations = 8000,
    [int]$Level = 0,
    [int]$Seed = 1,
    [string]$RunName = 'v8_1_roa_teacher_8k',
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ExtraArgs
)
$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\run_v8_pushcart.ps1" train `
    --num_envs $NumEnvs --max_iterations $MaxIterations --level $Level `
    --seed $Seed --run_name $RunName --headless @ExtraArgs
exit $LASTEXITCODE
