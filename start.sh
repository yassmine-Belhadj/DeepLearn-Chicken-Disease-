#!/bin/bash
z
echo "➡️ Running training script (main.py)..."
python main.py

echo "✅ Training completed. Starting Flask API..."
python app.py
