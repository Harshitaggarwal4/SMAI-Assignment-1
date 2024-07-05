#!/bin/bash
# chatGPT code starts
if [ $# -ne 1 ]; then
    echo "Usage: $0 filename"
    exit 1
fi
filename="$1"
if [ ! -f "$filename" ]; then
    echo "Error: File not found!"
    exit 1
fi
python3 1.py "$filename"
# chatGPT code ends