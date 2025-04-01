$Env:MAMBA_EXE="$Env:LOCALAPPDATA\miniforge3\Library\bin\mamba.exe"
$Env:MAMBA_ROOT_PREFIX="$Env:LOCALAPPDATA\miniforge3"

Write-Host "Activating environment"
# We run without a profile to keep shell startup time at a minimum. The conda hook still needs to be invoked.
(& $Env:MAMBA_EXE 'shell' 'hook' -s 'powershell' -r $Env:MAMBA_ROOT_PREFIX) | Out-String | Invoke-Expression
mamba activate tectonic

Write-Host "Building document"
& "tectonic.exe" -X build --keep-intermediates

Write-Host "Viewing document"
Start-Process $Env:LOCALAPPDATA\SumatraPDF\sumatrapdf.exe -ArgumentList @("-reuse-instance", "build\default\default.pdf")
