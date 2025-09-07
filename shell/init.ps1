Write-Host "Installing git hooks."
git config core.hooksPath .githooks

Write-Host "Initializing git submodules."
git submodule update --init

Write-Host "Running post-checkout hook"

# Get the root directory of the git repository.
$repoRoot = git rev-parse --show-toplevel 2>$null
if (-not $repoRoot) {
    Write-Host "Error: Not inside a Git repository."
    exit 1
}

# Build the full path to the post-checkout hook script.
$postCheckoutScript = Join-Path $repoRoot ".githooks\post-checkout"

# Ensure the script exists and is executable.
if (-not (Test-Path $postCheckoutScript)) {
    Write-Host "Error: post-checkout hook not found at $postCheckoutScript"
    exit 1
}

# Run the post-checkout hook. The call operator (&) is used to invoke the script.
& $postCheckoutScript
if ($LASTEXITCODE -ne 0) {
    Write-Host "Something went wrong."
    exit 1
}

Write-Host "Success!"
Write-Host "Compile with 'latexmk -r windows.latexmkrc -pv'"
exit 0

