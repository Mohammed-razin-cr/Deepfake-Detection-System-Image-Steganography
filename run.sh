#!/bin/bash

# Deepfake Detection System - Unix/Linux Startup Script

echo ""
echo "================================================"
echo "  Deepfake Detection System - Startup Script"
echo "================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.10+ from https://www.python.org"
    exit 1
fi

echo "[1/5] Checking Python version..."
python3 --version

echo ""
echo "[2/5] Checking virtual environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created!"
else
    echo "Virtual environment already exists"
fi

echo ""
echo "[3/5] Activating virtual environment..."
source venv/bin/activate

echo ""
echo "[4/5] Installing/updating dependencies..."
pip install -q -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error installing dependencies"
    exit 1
fi
echo "Dependencies installed successfully!"

echo ""
echo "[5/5] Starting Flask application..."
echo ""
echo "================================================"
echo "  Application starting on http://localhost:5000"
echo "  Press Ctrl+C to stop the server"
echo "================================================"
echo ""

cd web_app
python app.py

