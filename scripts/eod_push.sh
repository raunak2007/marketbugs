#!/bin/zsh
# End-of-day push: commit completed work and push to origin.
# Safety: skips (and logs why) if tests or schema validation fail, if any staged file contains the host home path,
# if a pilot/build/baseline is still running (outputs may be partial), or if the staged change is over 50 MB.
# Schedule (macOS, 23:30 daily):  scripts/install_eod_push.sh
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=artifacts/eod_push.log
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

if pgrep -f "src.agents.pilot|src.gen.build_marketbugs|src.baselines.ochiai" >/dev/null; then
  say "SKIP: experiment still running (outputs may be partial)"; exit 0; fi
.venv/bin/python -m pytest -q tests >/dev/null 2>&1 || { say "SKIP: unit tests failing"; exit 0; }
make -s validate >/dev/null 2>&1 || { say "SKIP: schema validation failing"; exit 0; }

git add -A
if git diff --cached --quiet; then say "nothing new to commit"; git push -q origin HEAD 2>>"$LOG"; exit 0; fi
files=$(git diff --cached --name-only)
leaks=$(echo "$files" | xargs git grep --cached -l -F "$HOME" -- 2>/dev/null)
if [ -n "$leaks" ]; then
  say "SKIP: staged files contain host home path: $(echo $leaks | tr '\n' ' ')"; git reset -q; exit 0; fi
size=$(echo "$files" | xargs du -ck 2>/dev/null | tail -1 | cut -f1)
if [ "${size:-0}" -gt 51200 ]; then say "SKIP: staged change ${size} KB > 50 MB"; git reset -q; exit 0; fi

n=$(echo "$files" | wc -l | tr -d ' ')
git commit -q -m "EOD $(date +%F): ${n} files" && git push -q origin HEAD 2>>"$LOG" \
  && say "pushed $(git rev-parse --short HEAD)" || say "FAIL: commit/push error"
