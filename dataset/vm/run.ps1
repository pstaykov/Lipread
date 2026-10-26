# Start (or resume) mining in a hidden background process:
#   powershell -ExecutionPolicy Bypass -File dataset\vm\run.ps1 [-Workers 14] [-Shard 1/2]
# It keeps running when the Remote Desktop window is closed (disconnect), but not when you sign out.
param([int]$Workers = [Math]::Max(1, [int]$env:NUMBER_OF_PROCESSORS - 2), [int]$Prep = 5, [string]$Shard = "1/2")
Set-Location (Resolve-Path "$PSScriptRoot\..\..")
$pidFile = "bundestag_data\clips\miner.pid"
if ((Test-Path $pidFile) -and (Get-Process -Id (Get-Content $pidFile) -ErrorAction SilentlyContinue)) {
    Write-Host "already running (pid $(Get-Content $pidFile))"
    exit 1
}
$log = "bundestag_data\clips\mine_clips_vm.log"
$cmd = "set GLOG_minloglevel=2&& set TF_CPP_MIN_LOG_LEVEL=3&& set PYTHONIOENCODING=utf-8&& " +
       ".venv-mine\Scripts\python.exe -u dataset\mine_clips.py --workers $Workers --prep $Prep --shard $Shard >> $log 2>&1"
Start-Process cmd.exe -ArgumentList "/c $cmd" -WindowStyle Hidden -WorkingDirectory (Get-Location).Path
Start-Sleep 15
Get-Content $log -Tail 3 -Encoding UTF8
Write-Host "running with $Workers workers, $Prep prep threads; progress: status.ps1, stop: stop.ps1"
