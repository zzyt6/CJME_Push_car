# V8: teacher-student on the V7.1 plant.
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$RunArgs)
$ErrorActionPreference = 'Stop'
$pythonPath = if ($env:HOMIE_PYTHON) { $env:HOMIE_PYTHON } else { 'E:\envs\RLisaac601\python.exe' }
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "Python not found: $pythonPath" }
$envRoot = Split-Path -Parent $pythonPath
$env:PATH = "$envRoot;$envRoot\Library\bin;$envRoot\Scripts;$env:PATH"
$env:PYTHONUTF8 = '1'
if (-not $RunArgs) { $RunArgs = @('--help') }
$ErrorActionPreference = 'Continue'
& $pythonPath -u "$PSScriptRoot\scripts\pushcart_v8.py" @RunArgs 2>&1 | ForEach-Object { $_.ToString() }
exit $LASTEXITCODE
