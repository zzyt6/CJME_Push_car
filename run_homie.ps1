param(
    [string]$Mode = 'doctor',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunArgs
)
$ErrorActionPreference = 'Stop'
$pythonPath = if ($env:HOMIE_PYTHON) { $env:HOMIE_PYTHON } else { 'E:\envs\RLisaac601\python.exe' }
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "Python not found: $pythonPath. Set HOMIE_PYTHON to your RLisaac601 python.exe." }
# Set the DLL search paths for this process only; no global environment edits.
$envRoot = Split-Path -Parent $pythonPath
$env:PATH = "$envRoot;$envRoot\Library\bin;$envRoot\Scripts;$env:PATH"
$env:PYTHONUTF8 = '1'
# Windows PowerShell 5 treats native stderr warnings as errors when redirected.
# The process exit code, not Kit's warning stream, determines success.
$ErrorActionPreference = 'Continue'
& $pythonPath -u "$PSScriptRoot\scripts\homie.py" $Mode @RunArgs 2>&1 | ForEach-Object { $_.ToString() }
exit $LASTEXITCODE
