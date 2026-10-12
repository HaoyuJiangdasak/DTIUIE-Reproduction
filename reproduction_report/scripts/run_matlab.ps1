param(
    [string]$MatlabExecutable = 'D:\Program Files\MATLAB\R2025b\bin\matlab.EXE'
)
$ErrorActionPreference = 'Stop'
$reportRoot = Split-Path -Parent $PSScriptRoot
$outputDirectory = Join-Path $reportRoot 'tools\matlab'
$plotScript = Join-Path $PSScriptRoot 'plot_matlab.m'
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
if (-not (Test-Path -LiteralPath $MatlabExecutable)) { throw "MATLAB executable missing: $MatlabExecutable" }
if (-not (Test-Path -LiteralPath (Join-Path $reportRoot 'data\epochs.csv'))) { throw 'data/epochs.csv is required.' }
if (-not (Test-Path -LiteralPath (Join-Path $reportRoot 'data\gpu_samples.csv'))) { throw 'data/gpu_samples.csv is required.' }
if (-not (Test-Path -LiteralPath (Join-Path $reportRoot 'data\summary.json'))) { throw 'data/summary.json is required.' }
$normalizedScriptPath = $plotScript.Replace('\', '/').Replace("'", "''")
$batchExpression = "run('$normalizedScriptPath')"
$stdoutPath = Join-Path $outputDirectory 'matlab_stdout.log'
$stderrPath = Join-Path $outputDirectory 'matlab_stderr.log'
$startedAt = Get-Date
$arguments = @('-batch', ('"' + $batchExpression + '"'))
$process = Start-Process -FilePath $MatlabExecutable -ArgumentList $arguments -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
$status = [ordered]@{
    executable = $MatlabExecutable
    batch = $batchExpression
    started_local = $startedAt.ToString('o')
    finished_local = (Get-Date).ToString('o')
    exit_code = $process.ExitCode
    stdout = $stdoutPath
    stderr = $stderrPath
}
$status | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $outputDirectory 'matlab_process_status.json') -Encoding UTF8
Get-Content -LiteralPath $stdoutPath
if ((Get-Item -LiteralPath $stderrPath).Length -gt 0) { Get-Content -LiteralPath $stderrPath }
if ($process.ExitCode -ne 0) { throw "MATLAB returned exit code $($process.ExitCode)." }
