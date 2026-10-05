# Independent HOMIE hand-force research task; currently baseline-equivalent, executed by NVIDIA's unified train.py.
# Omitted num_envs/max_iterations use the original 4096 / 100000 defaults.
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$TrainArgs
)
& "$PSScriptRoot\run_homie_hand_force.ps1" train @TrainArgs
exit $LASTEXITCODE
