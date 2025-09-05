#!/bin/bash
set -e

CMD=$1
shift

case "$CMD" in
  prepare)
    echo "[MIREX] BMACE prepare step (no action needed)"
    ;;
  do_chord_identification)
    INPUT=$1
    OUTPUT=$2
    shift 2
    echo "[MIREX] Running BMACE chord identification on $INPUT -> $OUTPUT"
    echo "Running python3 test_best.py ..."
    python3 test_best.py --input "$INPUT" --output "$OUTPUT" "$@" 2>&1 | tee run.log
    ;;
  *)
    echo "Unknown command: $CMD"
    exit 1
    ;;
esac