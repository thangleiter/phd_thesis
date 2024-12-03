Write-Host "Activating environment"
# We run without a profile to keep shell startup time at a minimum. The conda hook still needs to be invoked.
(& "$env:localappdata\miniforge3\Scripts\conda.exe" "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate tectonic

Write-Host "Building document"
& "tectonic.exe" -X watch --exec "build --keep-intermediates"

Write-Host "Viewing document"
Start-Process $env:localappdata\SumatraPDF\sumatrapdf.exe -ArgumentList @("-reuse-instance", "build\default\default.pdf")
