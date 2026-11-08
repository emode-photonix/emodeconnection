#!/bin/sh
# Run the suite a second time with every import spelled with the old name,
# emodeconnection, so the shim carries the whole client API (#121).
# Usage: sh tests/run_under_old_name.sh [pytest args]
set -eu
here=$(cd "$(dirname "$0")" && pwd)
copy="$here/../tests_old_name"
rm -rf "$copy"
mkdir "$copy"
cp -R "$here"/. "$copy"/
rm -f "$copy/test_shim.py" "$copy/run_under_old_name.sh"
trap 'rm -rf "$copy"' EXIT
for f in "$copy"/test_*.py; do
  perl -pi -e 's/^(\s*)import emode\.emode as /$1import emodeconnection.emodeconnection as /;
                s/^(\s*)from emode\.emode import/$1from emodeconnection.emodeconnection import/;
                s/^(\s*)(from|import) emode\b/$1$2 emodeconnection/' "$f"
done
python -m pytest "$copy" "$@"
