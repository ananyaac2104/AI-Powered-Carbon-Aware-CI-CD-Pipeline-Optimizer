#!/bin/bash

# Green-Ops Setup Script for Mac
# ==============================

echo "🌱 Starting Green-Ops environment setup..."

# 1. Check for Python 3
if ! command -v python3 &> /dev/null
then
    echo "❌ Error: python3 could not be found. Please install Python 3.9+."
    exit 1
fi

# 2. Clean up old venv if broken
if [ -d "venv" ]; then
    echo "🧹 Cleaning up existing venv..."
    rm -rf venv
fi

# 3. Create Virtual Environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# 4. Activate Venv
source venv/bin/activate

# 5. Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# 6. Install requirements
echo "📥 Installing dependencies (this may take a minute)..."
pip install -r requirements.txt

# 7. Add requests (if not in requirements)
pip install requests PyGithub

echo "✅ Setup complete! Run your project using:"
echo "   source venv/bin/activate"
echo "   python greenops_run_master.py"
