echo "Building document"
tectonic -X watch --exec "build --keep-intermediates"

echo "Viewing document"
nohup xdg-open "$HOME/Physik/Promotion/thesis/build/default/default.pdf" &> /dev/null &
# Need to sleep a tiny amount of time so that evince has time to start before the shell exits
sleep 1e-3