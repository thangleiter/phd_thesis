#!/bin/sh

echo "Installing git hooks."
git config core.hooksPath .githooks
echo "Initializing git submodules."
git submodule update --init
echo "Running post-checkout hook"
if ! "$(git rev-parse --show-toplevel)"/.githooks/post-checkout; then
    echo "Something went wrong."
    exit 1
fi

echo "Success!"
echo "Compile with 'tectonic -X build --keep-intermediates --open'"

exit 0

