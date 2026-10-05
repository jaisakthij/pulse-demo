#!/bin/bash
# PULSE Streamlit Launcher for bash/git-bash

cd "$(dirname "$0")"

echo "Starting PULSE Longitudinal Multimodal Demo..."
echo

# Find Python
if command -v python &>/dev/null; then
    PY=python
elif [ -x /c/Users/jaisa/AppData/Local/hermes/tools/python*/python ]; then
    PY=$(echo /c/Users/jaisa/AppData/Local/hermes/tools/python*/python | head -1)
else
    echo "ERROR: Python not found"
    exit 1
fi

echo "Using Python: $PY"
echo

# Check dependencies
$PY -c "import streamlit, numpy, pandas, torch" 2>/dev/null || {
    echo "Installing dependencies..."
    $PY -m pip install -q -r requirements.txt
}

echo
echo "========================================"
echo "  PULSE App Starting"
echo "  URL: http://127.0.0.1:8504"
echo "========================================"
echo
echo "Press Ctrl+C to stop the server"
echo

$PY -m streamlit run app.py --server.address 127.0.0.1 --server.port 8504
