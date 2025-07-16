#!/bin/bash
# Ensure virtual environment is activated (assumes already active)
echo "Using existing virtual environment: $VIRTUAL_ENV"

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
