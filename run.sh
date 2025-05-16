#!/bin/bash
cd "$(dirname "$0")"
$(poetry env activate)
while true; do
  echo "Starting whisper-dictation..."
  python whisper-dictation.py -m small --ptt
  echo "Finished"
done
