#!/bin/zsh
# Install (or with --remove, uninstall) a launchd job that runs scripts/eod_push.sh every day at 23:30.
set -eu
DIR="$(cd "$(dirname "$0")/.." && pwd)"
PLIST="$HOME/Library/LaunchAgents/com.marketbugs.eodpush.plist"
launchctl bootout "gui/$(id -u)/com.marketbugs.eodpush" 2>/dev/null || true
if [ "${1:-}" = "--remove" ]; then rm -f "$PLIST"; echo "removed"; exit 0; fi
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.marketbugs.eodpush</string>
  <key>ProgramArguments</key><array><string>$DIR/scripts/eod_push.sh</string></array>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>23</integer><key>Minute</key><integer>30</integer></dict>
  <key>StandardErrorPath</key><string>$DIR/artifacts/eod_push.err</string>
</dict></plist>
EOF
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed: runs $DIR/scripts/eod_push.sh daily at 23:30 (log: artifacts/eod_push.log)"
