# HOMIE v2 balance (stand-still) model, fine-tuned from the walking model_60000 (HomieRL/homie_isaaclab_v2/README.md).
# Record the walk-to-stop switch states first: .\run_homie_v2.ps1 bank --headless
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$TrainArgs
)
if ($null -eq $TrainArgs) { $TrainArgs = @() }
$checkpoint = "$PSScriptRoot\logs\rsl_rl\homie_g1\2026-09-14_15-10-07_my_policy\model_60000.pt"
if ($TrainArgs -notcontains '--checkpoint') { $TrainArgs = @('--checkpoint', $checkpoint) + $TrainArgs }
& "$PSScriptRoot\run_homie_v2.ps1" train --task balance @TrainArgs
exit $LASTEXITCODE
