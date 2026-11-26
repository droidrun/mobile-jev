#!/usr/bin/env bash
# Reinstall base mobilerun + mobilerun-agents into a local dev venv.
#
#   scripts/reinstall.sh                 # .venv, base from ../mobilerun
#   scripts/reinstall.sh .venv ~/code/mobilerun
#
# Base is installed as a normal package (your base checkout stays untouched).
# mobilerun-agents is installed editable, so the editor and demos use src/ directly.
set -euo pipefail

cd "$(dirname "$0")/.."
venv="${1:-.venv}"
base="${2:-../mobilerun}"

command -v uv >/dev/null || { echo "uv not found: brew install uv" >&2; exit 1; }
[ -d "$base" ] || { echo "base checkout not found: $base" >&2; exit 1; }

if [ ! -x "$venv/bin/python" ]; then
  echo "creating $venv (Python 3.13)"
  uv venv --python 3.13 "$venv"
fi
py="$venv/bin/python"

if [ -f "$base/.gitmodules" ] && [ -z "$(ls -A "$base/skills" 2>/dev/null)" ]; then
  echo "initialising base submodules"
  git -C "$base" submodule update --init
fi

echo "uninstalling mobilerun + mobilerun-agents"
uv pip uninstall --python "$py" mobilerun mobilerun-agents 2>/dev/null || true

branch="$(git -C "$base" branch --show-current 2>/dev/null || echo '?')"
commit="$(git -C "$base" log -1 --format='%h %s' 2>/dev/null || echo '?')"
dirty="$(git -C "$base" status --short 2>/dev/null | wc -l | tr -d ' ')"
echo "installing base from $base  [$branch @ $commit, $dirty uncommitted files]"
# Fresh build of the working tree, incl. uncommitted changes (no stale uv build cache).
uv pip install --python "$py" --reinstall-package mobilerun --refresh-package mobilerun --no-cache "$base"

echo "installing mobilerun-agents (editable) with dev + langfuse extras"
uv pip install --python "$py" -e ".[dev,langfuse]"

"$py" - <<'PY'
import mobilerun, mobilerun_agents
print("mobilerun        ->", mobilerun.__file__)
print("mobilerun_agents ->", mobilerun_agents.__file__)
PY
echo "done. In VS Code: 'Python: Select Interpreter' -> $venv/bin/python"
