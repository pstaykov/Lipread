# Stop the miner and its worker processes; interrupted speeches are retried on the next run.ps1.
Set-Location (Resolve-Path "$PSScriptRoot\..\..")
taskkill /PID (Get-Content bundestag_data\clips\miner.pid) /T /F
