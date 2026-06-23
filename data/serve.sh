#!/bin/bash
# Launch the RAGnosis dataset explorer:  ./serve.sh
# Serves this folder locally and opens the visualizer, which then
# auto-loads the dataset/ folder — no manual file selection needed.
cd "$(dirname "$0")" || exit 1
PORT="${PORT:-8777}"

# free the port if a previous run is still around
lsof -ti tcp:"$PORT" 2>/dev/null | xargs kill 2>/dev/null

URL="http://localhost:$PORT/index.html"
echo "Serving $(pwd) at $URL"
python3 -m http.server "$PORT" >/dev/null 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
sleep 1

# open the browser (macOS: open, Linux: xdg-open)
if command -v open >/dev/null 2>&1; then open "$URL"
elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL"
else echo "Open $URL in your browser."; fi

echo "Explorer running. Press Ctrl-C to stop the server."
wait $SRV
