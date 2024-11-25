echo "Activating environment"
# We run without a profile to keep shell startup time at a minimum. The conda hook still needs to be invoked.
eval "$($HOME/miniforge3/bin/conda shell.zsh hook)"
conda activate tectonic

echo "Building document"
$HOME/Code/tectonic/target/release/tectonic -X build --keep-intermediates

echo "Viewing document"
open build/default/default.pdf
