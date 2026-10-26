Set-Location (Resolve-Path "$PSScriptRoot\..\..")
$c = "bundestag_data\clips"
Get-Content "$c\mine_clips_vm.log" -Tail 300 | Select-String '^\[|speeches to go|stopped|Traceback|Error' | Select-Object -Last 5
$n = { param($f) if (Test-Path $f) { (Get-Content $f | Measure-Object -Line).Lines } else { 0 } }   # file stays readable while the miner appends
Write-Host "speeches: $(& $n "$c\clips_index.jsonl")  clips: $(& $n "$c\clips.jsonl")"
$drive = Get-PSDrive (Get-Location).Drive.Name
$os = Get-CimInstance Win32_OperatingSystem
$cpu = (Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average).Average
Write-Host ("disk free {0:N0} GB, RAM free {1:N1} GB, CPU {2}%" -f ($drive.Free / 1GB), ($os.FreePhysicalMemory / 1MB), $cpu)
