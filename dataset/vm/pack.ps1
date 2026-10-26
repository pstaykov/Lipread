# Pack the mined clips for the main PC (stop first, or the last few speeches are missing) -> clips_vm.tar
Set-Location (Resolve-Path "$PSScriptRoot\..\..")
tar -cf clips_vm.tar --exclude=_tmp --exclude=miner.pid -C bundestag_data clips
Get-Item clips_vm.tar | Select-Object Name, @{n = "GB"; e = { [Math]::Round($_.Length / 1GB, 1) } }
