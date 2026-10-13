# HOMIE v2 (port-vs-official fixes, HomieRL/homie_isaaclab_v2/README.md), executed by NVIDIA's unified train.py.
# Omitted num_envs/max_iterations use the original 4096 / 100000 defaults.
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$TrainArgs
)
& "$PSScriptRoot\run_homie_v2.ps1" train @TrainArgs
exit $LASTEXITCODE
