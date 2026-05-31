#!/usr/bin/env bash
# Launch the Face Blur app so it's reachable from other devices on your Wi-Fi
# (e.g. your phone). Prints the exact URL to open on the phone.
set -e
cd "$(dirname "$0")"

# Activate venv if present
[ -d .venv ] && source .venv/bin/activate

# Detect this machine's LAN IP (macOS Wi-Fi en0, then en1; Linux fallback)
IP=$(ipconfig getifaddr en0 2>/dev/null \
  || ipconfig getifaddr en1 2>/dev/null \
  || hostname -I 2>/dev/null | awk '{print $1}' \
  || echo "")

echo ""
echo "  Face Blur is starting…"
echo "  ────────────────────────────────────────────"
echo "  On THIS computer:   http://localhost:8000"
if [ -n "$IP" ]; then
  echo "  On your PHONE/tablet (same Wi-Fi): http://$IP:8000"
else
  echo "  On your phone: http://<this-computer-ip>:8000"
fi
echo "  ────────────────────────────────────────────"
echo "  (Ctrl+C to stop)"
echo ""

# --host 0.0.0.0 makes it reachable from other devices on the network.
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
