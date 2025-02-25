#!/bin/bash
cd "$(dirname "$0")"
while true; do
  echo "Starting whisper-dictation..."
  python whisper-dictation.py -m base --ptt
  echo "whisper-dictation crashed, restarting in 3 seconds..."
  sleep 1
done
