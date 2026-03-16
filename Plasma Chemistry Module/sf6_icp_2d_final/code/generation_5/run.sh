#!/bin/bash
# Run Gen-5 from this directory.
# Usage: bash run.sh [options]
#
# This script adds the shared_modules to the Python path so that
# the generation driver can import chemistry, mesh, solvers, etc.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR/../shared_modules:$PYTHONPATH"

python3 "$SCRIPT_DIR/main_gen5.py" "$@"
